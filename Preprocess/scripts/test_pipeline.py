#!/usr/bin/env python3
"""
test_pipeline.py — Standalone test for GNINA docking + ProLIF (TYR56 pi-pi only).

Usage:
    python Preprocess/scripts/test_pipeline.py '<SMILES>'
    python Preprocess/scripts/test_pipeline.py --check   # verify paths only

Env vars:
    DOCK_PY      path to your dock.py
    RECEPTOR_PDB path to receptor.pdb
    OUTPUT_DIR   output folder (default: pipeline_output)
    TYR_RESIDUE  TYR residue number (default: 56)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from rdkit import Chem

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


def _is_pi_pi_stacking(name: str) -> bool:
    n = name.lower().replace("-", "").replace("_", "")
    if "pication" in n or "cationpi" in n:
        return False
    return n == "pistacking" or ("pi" in n and "stack" in n)


def _is_tyr_residue(prot_res, resid: int) -> bool:
    if hasattr(prot_res, "resname") and hasattr(prot_res, "resid"):
        try:
            return str(prot_res.resname).upper() == "TYR" and int(prot_res.resid) == resid
        except (ValueError, TypeError):
            pass
    s = str(prot_res).upper()
    if "TYR" not in s:
        return False
    if re.search(rf"TYR[^\d]*{resid}(?:[^\d]|$)", s):
        return True
    if re.search(rf"(?:^|[^\d]){resid}[^\d]*TYR", s):
        return True
    return f"TYR{resid}" in s.replace(" ", "").replace(".", "").replace(":", "")


# ============================================================
# DOCKING
# ============================================================

def run(cmd: list) -> None:
    print("\n>> " + " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


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
    """Load receptor for ProLIF. Returns plf.Molecule."""
    import MDAnalysis as mda
    import prolif as plf

    if not os.path.isfile(receptor_pdb):
        raise FileNotFoundError(f"Receptor file not found: {receptor_pdb}")

    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
        tmp_path = tmp.name
        with open(receptor_pdb) as f:
            for line in f:
                if not line.startswith("CONECT"):
                    tmp.write(line)
        tmp.close()

        u = mda.Universe(tmp_path)
        ag = u.select_atoms("protein and not resname HOH WAT TIP3 SOL")
        if len(ag) == 0:
            ag = u.atoms
        ag.guess_bonds()
        return plf.Molecule.from_mda(ag), plf

    except Exception as exc_mda:
        try:
            rdmol = Chem.MolFromPDBFile(receptor_pdb, removeHs=False, sanitize=False)
            if rdmol is None:
                raise RuntimeError(
                    f"Cannot load receptor '{receptor_pdb}'.\n"
                    f"  MDAnalysis error: {exc_mda}\n"
                    f"  RDKit also returned None."
                )
            return plf.Molecule.from_rdkit(rdmol), plf
        except Exception as exc_rdkit:
            raise RuntimeError(
                f"Cannot load receptor '{receptor_pdb}'.\n"
                f"  MDAnalysis: {exc_mda}\n"
                f"  RDKit:      {exc_rdkit}"
            ) from exc_rdkit
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_best_pose(out_sdf: str) -> Chem.Mol:
    if not os.path.isfile(out_sdf):
        raise FileNotFoundError(f"Docked SDF not found: {out_sdf}")
    for mol in Chem.SDMolSupplier(out_sdf, removeHs=False):
        if mol is not None:
            return mol
    raise ValueError(f"No valid pose in {out_sdf}")


def analyze_tyr56_pi_stacking(receptor_pdb: str, out_sdf: str, tyr_resid: int = TYR_RESIDUE):
    protein, plf = load_protein_for_prolif(receptor_pdb)
    ligand = plf.Molecule.from_rdkit(get_best_pose(out_sdf))

    fp = plf.Fingerprint()
    fp.run(ligand, protein)

    interactions = []
    all_tyr_residues = set()

    for (lig_res, prot_res), ix_dict in fp.ifp.items():
        if "TYR" in str(prot_res):
            all_tyr_residues.add(str(prot_res))
        if not _is_tyr_residue(prot_res, tyr_resid):
            continue
        for name, metadata in ix_dict.items():
            if not metadata:
                continue
            if not _is_pi_pi_stacking(str(name)):
                continue
            interactions.append({
                "ligand": str(lig_res),
                "protein": str(prot_res),
                "interaction": str(name),
            })

    return len(interactions), interactions, sorted(all_tyr_residues)


# ============================================================
# MAIN
# ============================================================

def print_files(output_dir: Path) -> None:
    print("\nOUTPUT FILES:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(output_dir)}  ({f.stat().st_size/1024:.1f} KB)")


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--check":
        check_files()
        return

    if len(sys.argv) != 2:
        print("Usage:")
        print("  python test_pipeline.py '<SMILES>'")
        print("  python test_pipeline.py --check")
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
        pi_count, details, all_tyrs = analyze_tyr56_pi_stacking(RECEPTOR_PDB, out_sdf)
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
