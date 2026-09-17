#!/usr/bin/env python3
"""
Read a CSV with SMILES + docking score, add pIC50 and solubility predictions.

Edit the paths below, then run:
  python enrich_csv_scores.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── EDIT THESE ────────────────────────────────────────────────────────────────
INPUT_CSV = "results/linkinvent_rl.csv"      # input CSV path
OUTPUT_CSV = "results/linkinvent_enriched.csv"  # output CSV path
SMILES_COL = "SMILES"                        # column name for SMILES in input CSV
DOCKING_COL = "Docking_Score"                # column name for docking score (or DockingReward, affinity, etc.)
# ────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from pic50_scorer import readModel as readPic50Model, calculateScore as calculatePic50
from sol_scorer import readModel as readSolModel, calculateScore as calculateSolubility


def main() -> None:
    inp = Path(INPUT_CSV).expanduser().resolve()
    out_path = Path(OUTPUT_CSV).expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"File not found: {inp}")

    df = pd.read_csv(inp)
    if SMILES_COL not in df.columns:
        raise SystemExit(f"SMILES column '{SMILES_COL}' not found. Columns: {list(df.columns)}")
    if DOCKING_COL not in df.columns:
        raise SystemExit(f"Docking column '{DOCKING_COL}' not found. Columns: {list(df.columns)}")

    print("Loading QSAR models...")
    readPic50Model()
    readSolModel()

    rows = []
    for _, row in df.iterrows():
        smi = str(row[SMILES_COL]).strip()
        if not smi or smi.lower() == "nan":
            continue
        try:
            dock = float(row[DOCKING_COL])
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"[+] {len(out_df)} rows -> {out_path}")
    if len(out_df):
        print(out_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
