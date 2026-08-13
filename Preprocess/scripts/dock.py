#!/usr/bin/env python3
"""
dock.py — Batch Gnina docking from a SMILES CSV.

Matches the original workflow:
  python dock.py input.csv output_dir/

Each row produces:
  output_dir/mol{N}.sdf       — 3D ligand
  output_dir/mol{N}_out.sdf   — docked poses
  output_dir/mol{N}_log.txt   — gnina log (parse affinity from here)

Paths are configurable via CLI flags or environment variables:
  GNINA_EXECUTABLE, RECEPTOR_PDB, REF_LIGAND_PDB
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

_AFFINITY_RE = re.compile(
    r"^\s*1\s+(-?\d+(?:\.\d+)?)", re.MULTILINE
)


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(">> " + " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=check)


def embed_ligand(smiles: str, mol_path: str, sdf_path: str) -> bool:
    """Embed SMILES to 3D and write .mol + .sdf."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[!] Invalid SMILES: {smiles}")
        return False

    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xF00D) != 0:
        print(f"[!] Embedding failed: {smiles}")
        return False
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass

    with open(mol_path, "w") as f:
        f.write(Chem.MolToMolBlock(mol))

    # Prefer obabel if available (matches original workflow), else RDKit SDWriter
    obabel = shutil.which("obabel")
    if obabel:
        _run([obabel, "-imol", mol_path, "-osdf", "-O", sdf_path])
    else:
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()

    return os.path.exists(sdf_path)


def dock_ligand(
    sdf_path: str,
    out_sdf: str,
    log_path: str,
    receptor: str,
    autobox_ligand: str,
    gnina: str,
    cnn_scoring: str = "none",
) -> bool:
    """Run gnina on a single ligand SDF."""
    cmd = [
        gnina,
        "-r", receptor,
        "--ligand", sdf_path,
        "--autobox_ligand", autobox_ligand,
        "--cnn_scoring", cnn_scoring,
        "--out", out_sdf,
        "--log", log_path,
    ]
    try:
        _run(cmd)
        return os.path.exists(log_path)
    except subprocess.CalledProcessError as exc:
        print(f"[!] gnina failed (rc={exc.returncode})")
        return False


def extract_affinity(log_path: str) -> float | None:
    """Parse best (mode 1) affinity from gnina log."""
    text = Path(log_path).read_text()
    m = _AFFINITY_RE.search(text)
    if m:
        return float(m.group(1))
    m = re.search(r"Docking Score:\s*(-?\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def dock_one(
    smiles: str,
    index: int,
    output_dir: Path,
    receptor: str,
    autobox_ligand: str,
    gnina: str,
    cnn_scoring: str,
) -> dict:
    """Dock a single SMILES. Returns result dict."""
    fname = f"mol{index}"
    mol_path = str(output_dir / f"{fname}.mol")
    sdf_path = str(output_dir / f"{fname}.sdf")
    out_sdf = str(output_dir / f"{fname}_out.sdf")
    log_path = str(output_dir / f"{fname}_log.txt")

    print(f"\n[{index}] SMILES: {smiles}")

    if not embed_ligand(smiles, mol_path, sdf_path):
        return {"index": index, "smiles": smiles, "success": False, "error": "embed failed"}

    if not dock_ligand(sdf_path, out_sdf, log_path, receptor, autobox_ligand, gnina, cnn_scoring):
        return {"index": index, "smiles": smiles, "success": False, "error": "gnina failed"}

    affinity = extract_affinity(log_path)
    print(f"    affinity = {affinity} kcal/mol" if affinity else "    [!] could not parse affinity")
    print(f"    log: {log_path}")

    return {
        "index": index,
        "smiles": smiles,
        "success": True,
        "affinity": affinity,
        "log_file": log_path,
        "out_sdf": out_sdf,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch Gnina docking from SMILES CSV")
    parser.add_argument("input_csv", help="CSV with a SMILES column")
    parser.add_argument("output_dir", help="Directory for docking outputs")
    parser.add_argument("--receptor", default=os.environ.get("RECEPTOR_PDB"),
                        help="Receptor PDB (env: RECEPTOR_PDB)")
    parser.add_argument("--autobox_ligand", default=os.environ.get("REF_LIGAND_PDB"),
                        help="Reference ligand for autobox (env: REF_LIGAND_PDB)")
    parser.add_argument("--gnina", default=os.environ.get("GNINA_EXECUTABLE", "gnina"),
                        help="Gnina executable (env: GNINA_EXECUTABLE)")
    parser.add_argument("--cnn_scoring", default="none",
                        choices=["none", "rescore", "refinement", "all"])
    parser.add_argument("--smiles_col", default="SMILES", help="SMILES column name")
    parser.add_argument("--start", type=int, default=0, help="Start row index")
    parser.add_argument("--end", type=int, default=None, help="End row index (exclusive)")
    args = parser.parse_args()

    if not args.receptor:
        sys.exit("[!] --receptor or RECEPTOR_PDB env var required")
    if not args.autobox_ligand:
        sys.exit("[!] --autobox_ligand or REF_LIGAND_PDB env var required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if args.smiles_col not in df.columns:
        sys.exit(f"[!] Column '{args.smiles_col}' not found. Columns: {list(df.columns)}")

    end = args.end if args.end is not None else len(df)
    results = []

    for index in range(args.start, end):
        smiles = df[args.smiles_col][index]
        if pd.isna(smiles) or not str(smiles).strip():
            continue
        res = dock_one(
            str(smiles).strip(), index, output_dir,
            args.receptor, args.autobox_ligand, args.gnina, args.cnn_scoring,
        )
        results.append(res)

    summary_path = output_dir / "docking_summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\n[+] Summary: {summary_path} ({len(results)} molecules)")


if __name__ == "__main__":
    main()
