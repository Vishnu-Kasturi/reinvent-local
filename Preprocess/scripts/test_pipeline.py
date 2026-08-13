#!/usr/bin/env python3
"""
test_pipeline.py — Standalone test for GNINA docking + ProLIF.

Uses YOUR dock.py for docking, then runs ProLIF on mol0_out.sdf (pose 1).
No other project files required except dock.py.

Usage:
    python Preprocess/scripts/test_pipeline.py '<SMILES>'

Optional env vars:
    DOCK_PY      path to your dock.py
    RECEPTOR_PDB path to receptor.pdb (for ProLIF)
    OUTPUT_DIR   output folder (default: pipeline_output)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import MDAnalysis as mda
import pandas as pd
import prolif as plf
from rdkit import Chem

# ============================================================
# CONFIG — edit or set env vars
# ============================================================
DOCK_PY = os.environ.get(
    "DOCK_PY",
    "/home/genai/Vishnu/psearch-master/reinvent-local-main/Preprocess/scripts/dock.py",
)
RECEPTOR_PDB = os.environ.get(
    "RECEPTOR_PDB",
    "/home/genai/navneet/iict/pdl1/docking_TL_dataset/receptor.pdb",
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "pipeline_output")
TYR_RESIDUE = int(os.environ.get("TYR_RESIDUE", "56"))  # only this TYR residue


def _is_pi_pi_stacking(interaction_name: str) -> bool:
    """True only for pi-pi stacking (not cation-pi, H-bond, etc.)."""
    n = interaction_name.lower().replace("-", "").replace("_", "")
    if "pication" in n or "cationpi" in n:
        return False
    return n == "pistacking" or ("pi" in n and "stack" in n)


def _is_tyr_residue(prot_res, resid: int) -> bool:
    """True if protein residue is TYR with the given residue number."""
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
# HELPERS
# ============================================================

def run(cmd: list) -> None:
    print("\n>> " + " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def clean_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES.")
    return Chem.MolToSmiles(mol, canonical=True)


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


def extract_affinity(log_file: str) -> float:
    text = Path(log_file).read_text()
    m = re.search(r"^\s*1\s+(-?\d+(?:\.\d+)?)", text, re.MULTILINE)
    if m:
        return float(m.group(1))
    m = re.search(r"Docking Score:\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    raise ValueError("Could not parse affinity from GNINA log.")


def get_best_pose(out_sdf: str) -> Chem.Mol:
    for mol in Chem.SDMolSupplier(out_sdf, removeHs=False):
        if mol is not None:
            return mol
    raise ValueError(f"No valid pose in {out_sdf}")


def load_protein_for_prolif(receptor_pdb: str) -> plf.Molecule:
    """Load receptor for ProLIF. Strips partial CONECT records that cause valence errors."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    with open(receptor_pdb) as f:
        for line in f:
            if not line.startswith("CONECT"):
                tmp.write(line)
    tmp.close()
    try:
        u = mda.Universe(tmp.name)
        ag = u.select_atoms("protein and not resname HOH WAT TIP3 SOL")
        if len(ag) == 0:
            ag = u.atoms
        ag.guess_bonds()
        return plf.Molecule.from_mda(ag)
    except Exception:
        rdmol = Chem.MolFromPDBFile(receptor_pdb, removeHs=False, sanitize=False)
        if rdmol is None:
            raise RuntimeError(f"Cannot load receptor: {receptor_pdb}")
        return plf.Molecule.from_rdkit(rdmol)
    finally:
        os.unlink(tmp.name)


def analyze_tyr56_pi_stacking(
    receptor_pdb: str,
    out_sdf: str,
    tyr_resid: int = TYR_RESIDUE,
) -> tuple[int, list]:
    """Count pi-pi stacking at a specific TYR residue only (default: TYR56)."""
    protein = load_protein_for_prolif(receptor_pdb)
    ligand = plf.Molecule.from_rdkit(get_best_pose(out_sdf))

    fp = plf.Fingerprint()
    fp.run(ligand, protein)

    interactions = []
    for (lig_res, prot_res), ix_dict in fp.ifp.items():
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

    return len(interactions), interactions


# ============================================================
# DOCKING (your dock.py)
# ============================================================

def run_docking(smiles: str, output_dir: Path) -> tuple[str, str]:
    input_csv = output_dir / "input.csv"
    pd.DataFrame({"SMILES": [smiles]}).to_csv(input_csv, index=False)
    run(["python", DOCK_PY, str(input_csv), str(output_dir) + "/"])

    log_file = output_dir / "mol0_log.txt"
    out_sdf = output_dir / "mol0_out.sdf"
    if not log_file.exists():
        raise FileNotFoundError(f"Missing: {log_file}")
    if not out_sdf.exists():
        raise FileNotFoundError(f"Missing: {out_sdf}")
    return str(log_file), str(out_sdf)


# ============================================================
# PRINT / SAVE
# ============================================================

def print_files(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            tag = ""
            if f.name == "mol0_out.sdf":
                tag = "  <- GNINA poses (pose 1 = best)"
            elif f.name == "mol0_log.txt":
                tag = "  <- GNINA log"
            print(f"  {f.relative_to(output_dir)}  ({f.stat().st_size/1024:.1f} KB){tag}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python test_pipeline.py '<SMILES>'")
        sys.exit(1)

    smiles = clean_smiles(sys.argv[1])
    print("SMILES:", smiles)

    output_dir = Path(OUTPUT_DIR)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Step 1: dock
    print("\n" + "=" * 60)
    print("STEP 1 — GNINA (dock.py)")
    print("=" * 60)
    log_file, out_sdf = run_docking(smiles, output_dir)
    affinity = extract_affinity(log_file)
    dock_reward = affinity_to_reward(affinity)
    print_files(output_dir)
    print(f"\n  Affinity:        {affinity:.2f} kcal/mol")
    print(f"  Docking reward:  {dock_reward}  (<=−12→1.0, −12..−10→0.5, >−10→0.0)")

    # Step 2: prolif — TYR56 pi-pi stacking ONLY
    print("\n" + "=" * 60)
    print(f"STEP 2 — ProLIF (TYR{TYR_RESIDUE} pi-pi stacking only, pose 1)")
    print("=" * 60)
    try:
        pi_count, details = analyze_tyr56_pi_stacking(RECEPTOR_PDB, out_sdf)
        tyr_reward = tyr_count_to_reward(pi_count)
        print(f"  TYR{TYR_RESIDUE} pi-pi stacking: {pi_count}")
        print(f"  TYR reward:                  {tyr_reward}  (>=2→1.0, 1→0.5, 0→0.0)")
        if not details:
            print(f"    No pi-pi stacking at TYR{TYR_RESIDUE}.")
        for i, ix in enumerate(details, 1):
            print(f"    {i}. {ix['protein']} — {ix['interaction']}")
    except Exception as exc:
        print(f"  [!] ProLIF failed: {exc}")
        pi_count, tyr_reward = 0, 0.0

    summary = output_dir / "test_summary.txt"
    summary.write_text(
        f"SMILES: {smiles}\n"
        f"Affinity: {affinity:.2f} kcal/mol\n"
        f"Docking reward: {dock_reward}\n"
        f"TYR{TYR_RESIDUE} pi-pi stacking count: {pi_count}\n"
        f"TYR reward: {tyr_reward}\n"
    )
    print(f"\nSummary: {summary}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
