#!/usr/bin/env python3
"""
test_pipeline.py — Single-molecule docking + ProLIF interaction analysis.

Usage:
    python Preprocess/scripts/test_pipeline.py '<SMILES>'

Environment variables (or edit defaults below):
    GNINA_EXECUTABLE   path to gnina binary
    RECEPTOR_PDB       receptor PDB file
    REF_LIGAND_PDB     reference ligand for autobox
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

# Allow importing sibling scripts
_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent.parent
sys.path.insert(0, str(_SCRIPTS))

from prolif_utils import (  # noqa: E402
    get_pi_stacking,
    get_tyr_interactions,
    load_ligand_for_prolif,
    load_protein_for_prolif,
    run_prolif,
)

# ============================================================
# CONFIGURATION — override via env vars
# ============================================================
DOCK_PY = str(_SCRIPTS / "dock.py")
RECEPTOR_PDB = os.environ.get(
    "RECEPTOR_PDB",
    "/home/genai/navneet/iict/pdl1/docking_TL_dataset/receptor.pdb",
)
REF_LIGAND_PDB = os.environ.get(
    "REF_LIGAND_PDB",
    "/home/genai/navneet/iict/pdl1/docking_TL_dataset/ref_ligand.pdb",
)
OUTPUT_DIR = "pipeline_output"


def run(cmd: list) -> None:
    print("\n>> " + " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def clean_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES — RDKit could not parse input.")
    return Chem.MolToSmiles(mol, canonical=True)


def create_input_csv(smiles: str, output_dir: Path) -> str:
    path = output_dir / "input.csv"
    pd.DataFrame({"SMILES": [smiles]}).to_csv(path, index=False)
    return str(path)


def run_docking(smiles: str, output_dir: Path) -> tuple[str, str]:
    input_csv = create_input_csv(smiles, output_dir)
    run([
        "python", DOCK_PY, input_csv, str(output_dir) + "/",
        "--receptor", RECEPTOR_PDB,
        "--autobox_ligand", REF_LIGAND_PDB,
        "--cnn_scoring", "none",
    ])

    log_file = output_dir / "mol0_log.txt"
    out_sdf = output_dir / "mol0_out.sdf"
    if not log_file.exists():
        raise FileNotFoundError(f"Docking log not found: {log_file}")
    if not out_sdf.exists():
        raise FileNotFoundError(f"Docked SDF not found: {out_sdf}")
    return str(log_file), str(out_sdf)


def extract_best_docking_score(log_file: str) -> float:
    text = Path(log_file).read_text()
    m = re.search(r"^\s*1\s+(-?\d+(?:\.\d+)?)", text, re.MULTILINE)
    if m:
        return float(m.group(1))
    m = re.search(r"Docking Score:\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    raise ValueError("Could not extract docking score from gnina log.")


def analyze_interactions(receptor_pdb: str, docked_sdf: str):
    """Run ProLIF with robust protein loading."""
    print("\nLoading receptor for ProLIF...")
    protein = load_protein_for_prolif(receptor_pdb)
    print(f"  Protein loaded: {protein.n_residues} residues")

    print("Loading best gnina pose...")
    ligand = load_ligand_for_prolif(docked_sdf)

    import prolif as plf
    fp = plf.Fingerprint()
    print("Running ProLIF interaction analysis...")
    fp.run(ligand, protein)
    return fp


def print_results(docking_score, tyr_interactions, pi_pi_interactions):
    print("\n" + "=" * 60)
    print("DOCKING RESULT")
    print("=" * 60)
    print(f"Best docking score: {docking_score:.2f} kcal/mol")

    print("\n" + "=" * 60)
    print("TYROSINE INTERACTIONS")
    print("=" * 60)
    if not tyr_interactions:
        print("No TYR interactions detected.")
    else:
        print(f"Total TYR interactions: {len(tyr_interactions)}\n")
        for i, ix in enumerate(tyr_interactions, 1):
            print(f"{i}. Protein: {ix['protein']}")
            print(f"   Interaction: {ix['interaction']}")

    print("\n" + "=" * 60)
    print("TYR PI-PI STACKING")
    print("=" * 60)
    if not pi_pi_interactions:
        print("No TYR Pi-Pi stacking detected.")
    else:
        print(f"TYR Pi-Pi stacking interactions: {len(pi_pi_interactions)}\n")
        for i, ix in enumerate(pi_pi_interactions, 1):
            print(f"{i}. Protein: {ix['protein']}")
            print(f"   Interaction: {ix['interaction']}")


def save_results(output_dir, smiles, docking_score, tyr_interactions, pi_pi_interactions):
    path = Path(output_dir) / "interaction_results.txt"
    with open(path, "w") as f:
        f.write("=" * 60 + "\nINPUT SMILES\n" + "=" * 60 + "\n")
        f.write(smiles + "\n\n")
        f.write("=" * 60 + "\nDOCKING RESULT\n" + "=" * 60 + "\n")
        f.write(f"Best docking score: {docking_score:.2f} kcal/mol\n\n")
        f.write("=" * 60 + "\nALL TYR INTERACTIONS\n" + "=" * 60 + "\n")
        if not tyr_interactions:
            f.write("No TYR interactions detected.\n")
        else:
            for ix in tyr_interactions:
                f.write(f"Protein: {ix['protein']}\nInteraction: {ix['interaction']}\n\n")
        f.write("=" * 60 + "\nTYR PI-PI STACKING\n" + "=" * 60 + "\n")
        if not pi_pi_interactions:
            f.write("No TYR Pi-Pi stacking detected.\n")
        else:
            for ix in pi_pi_interactions:
                f.write(f"Protein: {ix['protein']}\nInteraction: {ix['interaction']}\n\n")
    print(f"\nResults saved to: {path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_pipeline.py '<SMILES>'")
        sys.exit(1)

    raw_smiles = sys.argv[1]
    print("=" * 60 + "\nINPUT SMILES\n" + "=" * 60)
    print(raw_smiles)

    smiles = clean_smiles(raw_smiles)
    print("\n" + "=" * 60 + "\nCLEAN SMILES\n" + "=" * 60)
    print(smiles)

    output_dir = Path(OUTPUT_DIR)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    log_file, out_sdf = run_docking(smiles, output_dir)
    docking_score = extract_best_docking_score(log_file)
    print(f"\nBest docking score: {docking_score:.2f} kcal/mol")

    fp = analyze_interactions(RECEPTOR_PDB, out_sdf)
    tyr_ix = get_tyr_interactions(fp)
    pi_pi = get_pi_stacking(tyr_ix)

    print_results(docking_score, tyr_ix, pi_pi)
    save_results(output_dir, smiles, docking_score, tyr_ix, pi_pi)
    print("\n" + "=" * 60 + "\nPIPELINE FINISHED SUCCESSFULLY\n" + "=" * 60)


if __name__ == "__main__":
    main()
