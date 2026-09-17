#!/usr/bin/env python3
"""
Read a CSV with SMILES + docking score, add pIC50 and solubility predictions.

Usage:
  cd CVAE_RL_finetuning   # or vendor/rl_infer
  python enrich_csv_scores.py input.csv
  python enrich_csv_scores.py input.csv --out output.csv

Finds SMILES and docking columns automatically (case-insensitive).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from pic50_scorer import readModel as readPic50Model, calculateScore as calculatePic50
from sol_scorer import readModel as readSolModel, calculateScore as calculateSolubility

SMILES_NAMES = ("smiles", "canonical_smiles", "input_smiles", "SMILES")
DOCKING_NAMES = (
    "docking_score",
    "docking",
    "dock",
    "dockingreward",
    "dockingaffinity_raw",
    "dockingaffinity",
    "affinity",
    "gnina_affinity",
    "Docking_Score",
    "Docking_Score",
)


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise SystemExit(f"No {label} column found. Columns: {list(df.columns)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Add pIC50 and solubility to a SMILES + docking CSV")
    p.add_argument("input_csv", help="Input CSV with SMILES and docking score")
    p.add_argument("--out", default=None, help="Output CSV (default: <input>_enriched.csv)")
    p.add_argument("--smiles-col", default=None, help="SMILES column name (auto-detect if omitted)")
    p.add_argument("--docking-col", default=None, help="Docking column name (auto-detect if omitted)")
    args = p.parse_args()

    inp = Path(args.input_csv).expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"File not found: {inp}")

    df = pd.read_csv(inp)
    smiles_col = args.smiles_col or _find_col(df, SMILES_NAMES, "SMILES")
    docking_col = args.docking_col or _find_col(df, DOCKING_NAMES, "docking")

    print("Loading QSAR models...")
    readPic50Model()
    readSolModel()

    rows = []
    for _, row in df.iterrows():
        smi = str(row[smiles_col]).strip()
        if not smi or smi.lower() == "nan":
            continue
        try:
            dock = float(row[docking_col])
        except (TypeError, ValueError):
            dock = np.nan

        pic50 = calculatePic50(smi)
        sol = calculateSolubility(smi)

        rows.append({
            "SMILES": smi,
            "pIC50": pic50 if math.isfinite(pic50) else np.nan,
            "Solubility": sol if math.isfinite(sol) else np.nan,
            "Docking_Score": dock if math.isfinite(dock) else np.nan,
        })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out).expanduser().resolve() if args.out else inp.with_name(f"{inp.stem}_enriched.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[+] {len(out_df)} rows -> {out_path}")
    if len(out_df):
        print(out_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
