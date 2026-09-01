"""Shared SMILES I/O and RDKit property helpers for vendor scripts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import MolLogP, MolWt, NumHAcceptors, NumHDonors, TPSA
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

SMILES_NAMES = ("smiles", "canonical_smiles", "input_smiles", "SMILES")
PROP_COLS = ("MW", "LogP", "HBD", "HBA", "TPSA", "Rot")

# Reference ranges (for distribution reports — not applied as filters here)
REFERENCE_RANGES = {
    "MW": [0, 500],
    "LogP": [-5, 5],
    "HBD": [0, 5],
    "HBA": [0, 10],
    "TPSA": [0, 200],
    "Rot": [0, 10],
}


def read_smiles(csv_path: Path, dedupe: bool = False) -> list[str]:
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    first = text.splitlines()[0] if text else ""
    sep = ";" if first.count(";") > first.count(",") else ","
    header = pd.read_csv(csv_path, sep=sep, nrows=0)
    col = next((c for c in header.columns if c.strip().lower() in {n.lower() for n in SMILES_NAMES}), None)
    if not col:
        raise ValueError(f"No SMILES column in {csv_path}. Columns: {list(header.columns)}")
    s = pd.read_csv(csv_path, sep=sep, usecols=[col], dtype=str)[col]
    smiles = s.dropna().astype(str).str.strip().loc[lambda x: x != ""].tolist()
    if dedupe:
        smiles = list(dict.fromkeys(smiles))
    return smiles


def calc_properties(mol: Chem.Mol) -> dict[str, float]:
    return {
        "MW": MolWt(mol),
        "LogP": MolLogP(mol),
        "HBD": float(NumHDonors(mol)),
        "HBA": float(NumHAcceptors(mol)),
        "TPSA": TPSA(mol),
        "Rot": float(CalcNumRotatableBonds(mol)),
    }


def properties_for_smiles(smiles: str) -> dict[str, float] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return calc_properties(mol)
