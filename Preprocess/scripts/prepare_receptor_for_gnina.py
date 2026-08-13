#!/usr/bin/env python3
"""
prepare_receptor_for_gnina.py — Convert receptor PDB to PDBQT for Gnina docking.

Gnina requires receptors in PDBQT format. This script tries multiple backends
in order: obabel (Open Babel), meeko, then mk_prepare_receptor (AutoDockTools).

Usage:
    python Preprocess/scripts/prepare_receptor_for_gnina.py \\
        --input receptor.pdb \\
        --output docking/receptor.pdbqt

    # Also compute grid center from a co-crystallized ligand:
    python Preprocess/scripts/prepare_receptor_for_gnina.py \\
        --input receptor.pdb \\
        --output docking/receptor.pdbqt \\
        --ref_ligand co_crystal_ligand.sdf \\
        --write_grid docking/grid.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from rdkit import Chem
from rdkit.Chem import AllChem


def _try_obabel(input_path: str, output_path: str) -> bool:
    obabel = shutil.which("obabel")
    if not obabel:
        return False
    cmd = [obabel, input_path, "-O", output_path, "-xr", "-p", "7.4", "--partialcharge", "gasteiger"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except subprocess.CalledProcessError:
        return False


def _try_meeko(input_path: str, output_path: str) -> bool:
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from meeko import RDKitMolCreate
    except ImportError:
        return False

    try:
        if input_path.endswith(".pdbqt"):
            shutil.copy(input_path, output_path)
            return True

        # meeko receptor prep via subprocess
        mk_rec = shutil.which("mk_prepare_receptor.py") or shutil.which("mk_prepare_receptor")
        if mk_rec:
            subprocess.run([mk_rec, "-i", input_path, "-o", output_path], check=True, capture_output=True)
            return os.path.exists(output_path)
        return False
    except Exception:
        return False


def _try_mk_prepare_receptor(input_path: str, output_path: str) -> bool:
    mk_rec = shutil.which("mk_prepare_receptor.py") or shutil.which("mk_prepare_receptor")
    if not mk_rec:
        return False
    try:
        subprocess.run([mk_rec, "-i", input_path, "-o", output_path], check=True, capture_output=True)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except subprocess.CalledProcessError:
        return False


def prepare_receptor(input_path: str, output_path: str) -> str:
    """Convert receptor to PDBQT. Returns the method used."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if input_path.endswith(".pdbqt"):
        shutil.copy(input_path, output_path)
        return "copy"

    for name, fn in [
        ("obabel", _try_obabel),
        ("meeko", _try_meeko),
        ("mk_prepare_receptor", _try_mk_prepare_receptor),
    ]:
        if fn(input_path, output_path):
            print(f"[+] Receptor prepared via {name}: {output_path}")
            return name

    raise RuntimeError(
        "Could not convert receptor to PDBQT. Install one of:\n"
        "  conda install -c conda-forge openbabel   (obabel)\n"
        "  pip install meeko                        (mk_prepare_receptor.py)\n"
        "Or provide a pre-prepared .pdbqt file directly."
    )


def compute_grid_from_ligand(ligand_path: str, padding: float = 4.0) -> dict:
    """Compute docking grid center and size from a reference ligand's bounding box."""
    if ligand_path.endswith(".sdf") or ligand_path.endswith(".mol"):
        suppl = Chem.SDMolSupplier(ligand_path, removeHs=False)
        mol = next((m for m in suppl if m is not None), None)
    else:
        mol = Chem.MolFromMolFile(ligand_path, removeHs=False)

    if mol is None:
        mol = Chem.MolFromSmiles(open(ligand_path).read().strip())
        if mol:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

    if mol is None:
        raise ValueError(f"Could not read ligand: {ligand_path}")

    conf = mol.GetConformer()
    coords = conf.GetPositions()
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = ((mins + maxs) / 2).tolist()
    size = (maxs - mins + 2 * padding).tolist()

    return {
        "center_x": round(center[0], 2),
        "center_y": round(center[1], 2),
        "center_z": round(center[2], 2),
        "size_x": round(max(size[0], 15.0), 2),
        "size_y": round(max(size[1], 15.0), 2),
        "size_z": round(max(size[2], 15.0), 2),
        "autobox_ligand": os.path.abspath(ligand_path),
        "autobox_add": padding,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare receptor PDBQT for Gnina docking")
    parser.add_argument("--input", required=True, help="Input receptor (.pdb or .pdbqt)")
    parser.add_argument("--output", required=True, help="Output receptor (.pdbqt)")
    parser.add_argument("--ref_ligand", default=None, help="Co-crystallized ligand for grid box")
    parser.add_argument("--write_grid", default=None, help="Write grid parameters to JSON file")
    parser.add_argument("--padding", type=float, default=4.0, help="Padding around ligand (Å)")
    args = parser.parse_args()

    method = prepare_receptor(args.input, args.output)
    print(f"[+] Method: {method}")

    if args.ref_ligand:
        grid = compute_grid_from_ligand(args.ref_ligand, padding=args.padding)
        print(f"[+] Grid center: ({grid['center_x']}, {grid['center_y']}, {grid['center_z']})")
        print(f"[+] Grid size:   ({grid['size_x']}, {grid['size_y']}, {grid['size_z']}) Å")
        if args.write_grid:
            os.makedirs(os.path.dirname(os.path.abspath(args.write_grid)), exist_ok=True)
            with open(args.write_grid, "w") as f:
                json.dump(grid, f, indent=2)
            print(f"[+] Grid written to {args.write_grid}")


if __name__ == "__main__":
    main()
