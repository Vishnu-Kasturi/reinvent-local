"""Shared helpers for vendor/filter_csv.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

PROP_COLS = ("MW", "LogP", "HBD", "HBA", "TPSA", "Rot")

SMILES_NAMES = ("smiles", "canonical_smiles", "input_smiles", "SMILES")

# Canonical output names → possible input column names (first match wins)
SCORE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "pIC50": ("pIC50", "pic50", "PD1PDL1pIC50", "PD1PDL1pIC50 (raw)"),
    "Solubility": ("Solubility", "solubility", "logS", "PD1PDL1Sol", "PD1PDL1Sol (raw)"),
    "Tyrosine_PiStacking": (
        "Tyrosine_PiStacking",
        "tyr_pi_stacking (TyrInteractionReward)",
        "TyrInteractionCount_raw",
        "TyrInteractionCount_raw (raw)",
    ),
    "Docking_Score": (
        "Docking_Score",
        "DockingAffinity_raw",
        "DockingAffinity_raw (raw)",
        "docking_score",
        "DockingReward (raw)",
    ),
}


def _detect_sep(path: Path) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
    if header.count(";") > header.count(",") and header.count(";") > header.count("\t"):
        return ";"
    if header.count("\t") > header.count(","):
        return "\t"
    return ","


def find_smiles_col(columns) -> str:
    lower = {str(c).strip().lower(): c for c in columns}
    for name in SMILES_NAMES:
        if name in columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise ValueError(f"No SMILES column. Columns: {list(columns)}")


def resolve_score_columns(columns) -> dict[str, str]:
    """Map canonical score name → actual column name in ``columns``."""
    lower = {str(c).strip().lower(): c for c in columns}
    found: dict[str, str] = {}
    for canonical, aliases in SCORE_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in columns:
                found[canonical] = alias
                break
            if alias.lower() in lower:
                found[canonical] = lower[alias.lower()]
                break
    return found


def load_input_table(csv_path: Path, dedupe: bool = False) -> tuple[pd.DataFrame, str]:
    """Load CSV; normalize SMILES column name to ``SMILES``."""
    sep = _detect_sep(csv_path)
    df = pd.read_csv(csv_path, sep=sep, dtype=str, keep_default_na=False)
    smi_col = find_smiles_col(df.columns)
    df = df.copy()
    df["SMILES"] = df[smi_col].astype(str).str.strip()
    df = df[df["SMILES"].str.len() > 0]
    df = df[df["SMILES"].str.lower() != "nan"]
    if dedupe:
        df = df.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df, smi_col


def read_smiles(csv_path: Path, dedupe: bool = False) -> list[str]:
    df, _ = load_input_table(csv_path, dedupe=dedupe)
    return df["SMILES"].tolist()


def calc_properties(mol: Chem.Mol) -> dict[str, float]:
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "Rot": float(Lipinski.NumRotatableBonds(mol)),
    }
