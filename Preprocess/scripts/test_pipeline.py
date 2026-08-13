#!/usr/bin/env python3
"""
test_pipeline.py — Standalone test for GNINA docking + ProLIF (TYR56 pi-pi only).

This is ONE self-contained script. The only external file you need is your dock.py.

Usage:
    python Preprocess/scripts/test_pipeline.py '<SMILES>'
    python Preprocess/scripts/test_pipeline.py --check
    python Preprocess/scripts/test_pipeline.py --debug              # prolif dump on existing output
    python Preprocess/scripts/test_pipeline.py --debug '<SMILES>'   # dock + full prolif dump

Env vars:
    DOCK_PY      path to your dock.py
    RECEPTOR_PDB path to receptor.pdb
    OUTPUT_DIR   output folder (default: pipeline_output)
    TYR_RESIDUE  TYR residue number (default: 56)
"""
from __future__ import annotations

import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


# ============================================================
# ProLIF v1/v2 compat (inlined — no extra files needed)
# ============================================================

def _prolif_version(plf) -> str:
    return getattr(plf, "__version__", "unknown")


def _make_fingerprint(plf, count: bool = True):
    try:
        return plf.Fingerprint(count=count)
    except TypeError:
        return plf.Fingerprint()


def _get_ifp_from_fingerprint(fp, frame: int = 0):
    ifp = fp.ifp
    if isinstance(ifp, dict) and ifp and all(isinstance(k, int) for k in ifp):
        return ifp[frame]
    return ifp


def _run_fingerprint(fp, ligand, protein, residues: Optional[List[str]] = None, frame: int = 0):
    if hasattr(fp, "generate"):
        sig = inspect.signature(fp.generate)
        kwargs: dict[str, Any] = {}
        if residues is not None and "residues" in sig.parameters:
            kwargs["residues"] = residues
        if "metadata" in sig.parameters:
            kwargs["metadata"] = True
        try:
            return fp.generate(ligand, protein, **kwargs)
        except TypeError:
            pass

    if hasattr(fp, "run_from_iterable"):
        if residues is not None:
            fp.run_from_iterable([ligand], protein, residues=residues)
        else:
            fp.run_from_iterable([ligand], protein)
        return _get_ifp_from_fingerprint(fp, frame=frame)

    if residues is not None:
        try:
            fp.run(ligand, protein, residues=residues)
        except TypeError:
            fp.run(ligand, protein)
    else:
        fp.run(ligand, protein)
    return fp.ifp


def _iter_ifp_pairs(ifp) -> Iterator[Tuple[Any, Any, dict]]:
    for key, ix_dict in ifp.items():
        if isinstance(key, tuple) and len(key) == 2:
            yield key[0], key[1], ix_dict


def _ifp_to_dataframe(plf, fp, ifp):
    if hasattr(plf, "to_dataframe"):
        interactions = getattr(fp, "interactions", None)
        if interactions is not None:
            return plf.to_dataframe({0: ifp}, interactions)
        return plf.to_dataframe({0: ifp})
    if hasattr(fp, "to_dataframe"):
        return fp.to_dataframe(ifp)
    raise TypeError("No compatible to_dataframe API found")


def _count_interactions(ix_dict, predicate) -> int:
    total = 0
    for name, metadata in ix_dict.items():
        if not predicate(name) or metadata is None:
            continue
        if isinstance(metadata, (list, tuple)):
            total += len(metadata)
        elif isinstance(metadata, dict):
            total += max(len(metadata), 1) if metadata else 0
        else:
            total += 1
    return total


def _tyr_residue_ids(resid: int, chains: str = "AB") -> List[str]:
    return [f"TYR{resid}.{c}" for c in chains]

# ============================================================
# CONFIG — edit these paths for your machine
# ============================================================
DOCK_PY = os.environ.get(
    "DOCK_PY",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dock.py"),
)
RECEPTOR_PDB = os.environ.get(
    "RECEPTOR_PDB",
    "/home/genai/navneet/iict/pdl1/docking_TL_dataset/receptor.pdb",
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "pipeline_output")
TYR_RESIDUE = int(os.environ.get("TYR_RESIDUE", "56"))


# ============================================================
# PREFLIGHT — catch missing files early
# ============================================================

def check_files() -> None:
    errors = []
    if not os.path.isfile(DOCK_PY):
        errors.append(f"  dock.py NOT FOUND: {DOCK_PY}")
    if not os.path.isfile(RECEPTOR_PDB):
        errors.append(f"  receptor.pdb NOT FOUND: {RECEPTOR_PDB}")
    if errors:
        print("FILE CHECK FAILED:\n" + "\n".join(errors))
        print("\nFix: set env vars before running:")
        print('  export DOCK_PY="/full/path/to/dock.py"')
        print('  export RECEPTOR_PDB="/full/path/to/receptor.pdb"')
        sys.exit(1)
    print("FILE CHECK OK:")
    print(f"  dock.py:      {DOCK_PY}")
    print(f"  receptor.pdb: {RECEPTOR_PDB}")


# ============================================================
# REWARD HELPERS
# ============================================================

def affinity_to_reward(affinity: float) -> float:
    if affinity <= -12.0:
        return 1.0
    if affinity <= -10.0:
        return 0.5
    return 0.0


def tyr_count_to_reward(count: int) -> float:
    if count >= 2:
        return 1.0
    if count == 1:
        return 0.5
    return 0.0


def _interaction_name(name) -> str:
    """ProLIF keys can be class objects e.g. PiStacking, not strings."""
    if hasattr(name, "__name__"):
        return name.__name__
    return str(name)


def _residue_info(prot_res) -> dict:
    """Extract residue attributes for debugging."""
    info = {"str": str(prot_res)}
    for attr in ("resname", "resid", "chain", "segid", "icode"):
        if hasattr(prot_res, attr):
            info[attr] = getattr(prot_res, attr)
    return info


def _is_pi_pi_stacking(name) -> bool:
    n = _interaction_name(name).lower().replace("-", "").replace("_", "")
    if "pication" in n or "cationpi" in n:
        return False
    return n == "pistacking" or ("pi" in n and "stack" in n)


def _is_tyr_residue(prot_res, resid: int) -> bool:
    info = _residue_info(prot_res)
    if info.get("resname", "").upper() == "TYR":
        try:
            if int(info["resid"]) == resid:
                return True
        except (ValueError, TypeError):
            pass
    s = info["str"].upper()
    if "TYR" not in s:
        return False
    if re.search(rf"TYR[^\d]*{resid}(?:[^\d]|$)", s):
        return True
    if re.search(rf"(?:^|[^\d]){resid}[^\d]*TYR", s):
        return True
    return f"TYR{resid}" in s.replace(" ", "").replace(".", "").replace(":", "")


def inspect_pdb_residue(receptor_pdb: str, resid: int) -> None:
    """Print what residue name is at PDB residue number `resid`."""
    print(f"\n--- PDB residue {resid} in {receptor_pdb} ---")
    found = []
    with open(receptor_pdb) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resnum = line[22:26].strip()
            try:
                if int(resnum) != resid:
                    continue
            except ValueError:
                continue
            resname = line[17:20].strip()
            chain = line[21].strip()
            found.append(f"  chain={chain} resname={resname} resnum={resnum}")
    if found:
        print("\n".join(dict.fromkeys(found)))  # unique lines
    else:
        print(f"  No ATOM/HETATM record at residue number {resid}")


# ============================================================
# DOCKING
# ============================================================

def run(cmd: list) -> None:
    print("\n>> " + " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def _count_aromatic_rings(mol) -> int:
    if mol is None:
        return 0
    try:
        ri = mol.GetRingInfo()
        return sum(
            1 for ring in ri.AtomRings()
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        )
    except Exception:
        return 0


def _prepare_docked_ligand(pose_mol, smiles: str) -> Chem.Mol:
    """
    GNINA/OpenBabel SDFs often lack bond orders and aromaticity.
    Copy them from the SMILES template onto the docked 3D coordinates.
    """
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    pose = Chem.Mol(pose_mol)
    try:
        fixed = AllChem.AssignBondOrdersFromTemplate(template, pose)
    except (ValueError, RuntimeError):
        pose_h = Chem.RemoveHs(pose)
        template_h = Chem.RemoveHs(template)
        fixed_h = AllChem.AssignBondOrdersFromTemplate(template_h, pose_h)
        fixed = Chem.AddHs(fixed_h, addCoords=True)

    Chem.SanitizeMol(fixed)
    return fixed


def _load_input_smiles(output_dir: str | Path) -> str:
    csv_path = Path(output_dir) / "input.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No {csv_path} — re-run with: python test_pipeline.py --debug '<SMILES>'"
        )
    return clean_smiles(str(pd.read_csv(csv_path)["SMILES"].iloc[0]))


def clean_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES.")
    return Chem.MolToSmiles(mol, canonical=True)


def extract_affinity(log_file: str) -> float:
    text = Path(log_file).read_text()
    m = re.search(r"^\s*1\s+(-?\d+(?:\.\d+)?)", text, re.MULTILINE)
    if m:
        return float(m.group(1))
    m = re.search(r"Docking Score:\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    raise ValueError(f"Could not parse affinity from: {log_file}")


def run_docking(smiles: str, output_dir: Path) -> tuple[str, str]:
    input_csv = output_dir / "input.csv"
    pd.DataFrame({"SMILES": [smiles]}).to_csv(input_csv, index=False)
    run([sys.executable, DOCK_PY, str(input_csv), str(output_dir) + "/"])

    log_file = output_dir / "mol0_log.txt"
    out_sdf = output_dir / "mol0_out.sdf"
    if not log_file.exists():
        raise FileNotFoundError(f"GNINA log missing: {log_file}")
    if not out_sdf.exists():
        raise FileNotFoundError(f"GNINA output missing: {out_sdf}")
    return str(log_file), str(out_sdf)


# ============================================================
# PROLIF
# ============================================================

def load_protein_for_prolif(receptor_pdb: str):
    """Load receptor for ProLIF. Returns (plf.Molecule, plf_module)."""
    import prolif as plf

    if not os.path.isfile(receptor_pdb):
        raise FileNotFoundError(f"Receptor file not found: {receptor_pdb}")

    # MDAnalysis first — keeps TYR phenyl rings aromatic for pi-stacking
    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
        tmp_path = tmp.name
        with open(receptor_pdb) as f:
            for line in f:
                if not line.startswith("CONECT"):
                    tmp.write(line)
        tmp.close()

        import MDAnalysis as mda
        u = mda.Universe(tmp_path)
        ag = u.select_atoms("protein and not resname HOH WAT TIP3 SOL")
        if len(ag) == 0:
            ag = u.atoms
        ag.guess_bonds()
        return plf.Molecule.from_mda(ag), plf

    except Exception:
        pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    try:
        rdmol = Chem.MolFromPDBFile(receptor_pdb, removeHs=False, sanitize=True)
        if rdmol is not None:
            return plf.Molecule.from_rdkit(rdmol), plf
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load receptor '{receptor_pdb}': {exc}"
        ) from exc

    raise RuntimeError(f"Cannot load receptor '{receptor_pdb}'")


def get_best_pose(out_sdf: str, smiles: str) -> Chem.Mol:
    if not os.path.isfile(out_sdf):
        raise FileNotFoundError(f"Docked SDF not found: {out_sdf}")
    for mol in Chem.SDMolSupplier(out_sdf, removeHs=False):
        if mol is not None:
            return _prepare_docked_ligand(mol, smiles)
    raise ValueError(f"No valid pose in {out_sdf}")


def run_prolif(
    receptor_pdb: str,
    out_sdf: str,
    smiles: str,
    tyr_resid: int = TYR_RESIDUE,
):
    """Run ProLIF, return (fp, ifp, protein, ligand, plf_module)."""
    protein, plf = load_protein_for_prolif(receptor_pdb)
    ligand = plf.Molecule.from_rdkit(get_best_pose(out_sdf, smiles))
    fp = _make_fingerprint(plf, count=True)
    residues = _tyr_residue_ids(tyr_resid)
    ifp = _run_fingerprint(fp, ligand, protein, residues=residues)
    return fp, ifp, protein, ligand, plf


def count_tyr56_pi_stacking(ifp, tyr_resid: int = TYR_RESIDUE) -> tuple[int, list]:
    """Count pi-pi stacking at TYR{resid} using count=True fingerprint."""
    details = []
    total = 0
    for lig_res, prot_res, ix_dict in _iter_ifp_pairs(ifp):
        if not _is_tyr_residue(prot_res, tyr_resid):
            continue
        n = _count_interactions(ix_dict, _is_pi_pi_stacking)
        if n > 0:
            total += n
            details.append({
                "ligand": str(lig_res),
                "protein": str(prot_res),
                "interaction": "PiStacking",
                "count": n,
            })
    return total, details


def debug_prolif(
    receptor_pdb: str,
    out_sdf: str,
    smiles: str,
    tyr_resid: int = TYR_RESIDUE,
) -> None:
    """Dump EVERY ProLIF interaction — use this when count is unexpectedly 0."""
    print("\n" + "=" * 70)
    print("PROLIF DEBUG DUMP")
    print("=" * 70)

    inspect_pdb_residue(receptor_pdb, tyr_resid)

    raw_pose = next(m for m in Chem.SDMolSupplier(out_sdf, removeHs=False) if m is not None)
    n_raw = _count_aromatic_rings(raw_pose)
    fixed_pose = _prepare_docked_ligand(raw_pose, smiles)
    n_fixed = _count_aromatic_rings(fixed_pose)
    print(f"\n--- Aromaticity fix (GNINA SDF → SMILES template) ---")
    print(f"  Raw GNINA pose aromatic rings:    {n_raw}")
    print(f"  After SMILES bond-order fix:      {n_fixed}")
    if n_raw == 0 and n_fixed > 0:
        print("  (Pi-stacking needs aromatic rings — raw SDF had none)")

    fp, ifp, protein, ligand, plf = run_prolif(receptor_pdb, out_sdf, smiles, tyr_resid)
    print(f"\nProLIF version: {_prolif_version(plf)}")
    print(f"Fingerprint count=True (detects multiple stacks per residue)")

    # Try dataframe export if available
    try:
        df = _ifp_to_dataframe(plf, fp, ifp)
        print(f"\n--- ProLIF dataframe ({len(df)} rows) ---")
        print(df.to_string())
    except Exception as exc:
        print(f"\n(DataFrame export unavailable: {exc})")

    pairs = list(_iter_ifp_pairs(ifp))
    print(f"\n--- ALL interactions ({len(pairs)} residue pairs) ---")
    tyr_all = []
    tyr_target = []
    pi_all = []

    for lig_res, prot_res, ix_dict in pairs:
        pinfo = _residue_info(prot_res)
        is_tyr = "TYR" in pinfo["str"].upper() or str(pinfo.get("resname", "")).upper() == "TYR"
        is_target = _is_tyr_residue(prot_res, tyr_resid)

        for name, metadata in ix_dict.items():
            iname = _interaction_name(name)
            row = {
                "ligand": str(lig_res),
                "protein": pinfo["str"],
                "protein_resid": pinfo.get("resid"),
                "protein_resname": pinfo.get("resname"),
                "interaction": iname,
                "metadata": repr(metadata),
                "is_pi": _is_pi_pi_stacking(name),
            }
            if is_tyr:
                tyr_all.append(row)
            if is_target:
                tyr_target.append(row)
            if row["is_pi"]:
                pi_all.append(row)

            print(f"  {row['protein']:20s} | {iname:20s} | meta={row['metadata'][:40]}")

    print(f"\n--- TYR residues only ({len(tyr_all)} interactions) ---")
    for r in tyr_all:
        print(f"  resid={r['protein_resid']} {r['protein']:20s} | {r['interaction']}")

    print(f"\n--- TYR{tyr_resid} only ({len(tyr_target)} interactions) ---")
    for r in tyr_target:
        flag = " [PI-PI]" if r["is_pi"] else ""
        print(f"  {r['interaction']}{flag}  meta={r['metadata']}")

    print(f"\n--- Pi-pi stacking anywhere ({len(pi_all)} total) ---")
    for r in pi_all:
        print(f"  resid={r['protein_resid']} {r['protein']} | {r['interaction']}")

    pi_at_target_count, pi_details = count_tyr56_pi_stacking(ifp, tyr_resid)
    print(f"\n--- RESULT: TYR{tyr_resid} pi-pi stacking count = {pi_at_target_count} ---")
    for d in pi_details:
        print(f"  {d['protein']} x{d['count']}")

    # Check multiple poses
    print(f"\n--- Checking all poses in {out_sdf} ---")
    supplier = Chem.SDMolSupplier(out_sdf, removeHs=False)
    residues = _tyr_residue_ids(tyr_resid)
    for pose_i, mol in enumerate(supplier):
        if mol is None:
            continue
        lig = plf.Molecule.from_rdkit(_prepare_docked_ligand(mol, smiles))
        fp2 = _make_fingerprint(plf, count=True)
        ifp2 = _run_fingerprint(fp2, lig, protein, residues=residues)
        count, _ = count_tyr56_pi_stacking(ifp2, tyr_resid)
        print(f"  Pose {pose_i + 1}: TYR{tyr_resid} pi-pi count = {count}")


def analyze_tyr56_pi_stacking(
    receptor_pdb: str,
    out_sdf: str,
    smiles: str,
    tyr_resid: int = TYR_RESIDUE,
):
    _, ifp, _, _, _ = run_prolif(receptor_pdb, out_sdf, smiles, tyr_resid)
    count, interactions = count_tyr56_pi_stacking(ifp, tyr_resid)

    all_tyr_residues = set()
    for _, prot_res, ix_dict in _iter_ifp_pairs(ifp):
        if "TYR" in str(prot_res):
            info = _residue_info(prot_res)
            all_tyr_residues.add(f"{info.get('resname','?')}{info.get('resid','?')} ({prot_res})")

    return count, interactions, sorted(all_tyr_residues)


# ============================================================
# MAIN
# ============================================================

def print_files(output_dir: Path) -> None:
    print("\nOUTPUT FILES:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(output_dir)}  ({f.stat().st_size/1024:.1f} KB)")


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--check":
        check_files()
        return

    # --debug on existing pipeline_output (skip docking)
    if sys.argv[1:2] == ["--debug"] and len(sys.argv) == 2:
        if not os.path.isfile(RECEPTOR_PDB):
            print(f"RECEPTOR_PDB not found: {RECEPTOR_PDB}")
            sys.exit(1)
        out_sdf = Path(OUTPUT_DIR) / "mol0_out.sdf"
        if not out_sdf.exists():
            print(f"No docked output at {out_sdf}")
            print("Run docking first, or: python test_pipeline.py --debug '<SMILES>'")
            sys.exit(1)
        smiles = _load_input_smiles(OUTPUT_DIR)
        debug_prolif(RECEPTOR_PDB, str(out_sdf), smiles)
        return

    if len(sys.argv) == 3 and sys.argv[1] == "--debug":
        check_files()
        smiles = clean_smiles(sys.argv[2])
        output_dir = Path(OUTPUT_DIR)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        print("=== DOCKING ===")
        _, out_sdf = run_docking(smiles, output_dir)
        debug_prolif(RECEPTOR_PDB, out_sdf, smiles)
        return

    if len(sys.argv) != 2:
        print("Usage:")
        print("  python test_pipeline.py '<SMILES>'")
        print("  python test_pipeline.py --check")
        print("  python test_pipeline.py --debug              # debug existing pipeline_output/")
        print("  python test_pipeline.py --debug '<SMILES>'   # dock then debug")
        sys.exit(1)

    check_files()

    smiles = clean_smiles(sys.argv[1])
    print(f"\nSMILES: {smiles}")

    output_dir = Path(OUTPUT_DIR)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Step 1: GNINA
    print("\n=== STEP 1: GNINA (dock.py) ===")
    log_file, out_sdf = run_docking(smiles, output_dir)
    affinity = extract_affinity(log_file)
    dock_reward = affinity_to_reward(affinity)
    print_files(output_dir)
    print(f"\nAffinity:       {affinity:.2f} kcal/mol")
    print(f"Docking reward: {dock_reward}")

    # Step 2: ProLIF
    print(f"\n=== STEP 2: ProLIF (TYR{TYR_RESIDUE} pi-pi stacking only) ===")
    try:
        pi_count, details, all_tyrs = analyze_tyr56_pi_stacking(RECEPTOR_PDB, out_sdf, smiles)
        tyr_reward = tyr_count_to_reward(pi_count)
        print(f"TYR{TYR_RESIDUE} pi-pi stacking: {pi_count}")
        print(f"TYR reward:       {tyr_reward}")
        if not details:
            print(f"  No pi-pi stacking at TYR{TYR_RESIDUE}.")
            if all_tyrs:
                print(f"  (Other TYR residues seen by ProLIF: {', '.join(all_tyrs)})")
        for i, ix in enumerate(details, 1):
            print(f"  {i}. {ix['protein']} — {ix['interaction']}")
    except Exception as exc:
        print(f"PROLIF FAILED: {exc}")
        pi_count, tyr_reward = 0, 0.0

    summary = output_dir / "test_summary.txt"
    summary.write_text(
        f"SMILES: {smiles}\n"
        f"Affinity: {affinity:.2f} kcal/mol\n"
        f"Docking reward: {dock_reward}\n"
        f"TYR{TYR_RESIDUE} pi-pi stacking: {pi_count}\n"
        f"TYR reward: {tyr_reward}\n"
    )
    print(f"\nSummary saved: {summary}")
    print("DONE.")


if __name__ == "__main__":
    main()
