#!/usr/bin/env python3
"""
Enrich a REINVENT / LinkInvent output CSV with pIC50 and solubility predictions.

For reinvent-local-main pipeline (NOT RBDD / vendor/rl_infer).

Edit paths below, then from repo root:
  python Preprocess/scripts/enrich_csv_scores.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# ── EDIT THESE (paths relative to reinvent-local-main repo root) ─────────────
INPUT_CSV = "results/linkinvent_2_1.csv"
OUTPUT_CSV = "results/linkinvent_2_1_enriched.csv"
SMILES_COL = "SMILES"
DOCKING_COL = "DockingAffinity_raw"   # or Docking_Score, DockingReward, affinity, etc.
# ─────────────────────────────────────────────────────────────────────────────

PIC50_MODEL = "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_model.ubj"
PIC50_SCALER = "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_scaler.pkl"
SOL_MODEL = "Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj"
SOL_SCALER = "Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl"

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "REINVENT4"))

from reinvent_plugins.components.pd1_pdl1_features import compute_features


def _load_booster(path: Path) -> xgb.Booster:
    bst = xgb.Booster()
    bst.load_model(str(path))
    return bst


def _predict_pic50(smiles: list[str], model: xgb.Booster, scaler: str) -> list[float]:
    X, mask = compute_features(smiles, scaler)
    X = X[:, :2415]
    preds = model.predict(xgb.DMatrix(X))
    out = []
    for i, raw in enumerate(preds):
        if mask[i] and np.isfinite(raw):
            out.append(float(raw))
        else:
            out.append(np.nan)
    return out


def _predict_sol(smiles: list[str], model: xgb.Booster, scaler: str) -> list[float]:
    X, mask = compute_features(smiles, scaler)
    preds = model.predict(xgb.DMatrix(X))
    out = []
    for i, raw in enumerate(preds):
        if mask[i] and np.isfinite(raw):
            out.append(float(raw))
        else:
            out.append(np.nan)
    return out


def main() -> None:
    inp = (_REPO_ROOT / INPUT_CSV).resolve()
    out_path = (_REPO_ROOT / OUTPUT_CSV).resolve()

    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    df = pd.read_csv(inp)
    if SMILES_COL not in df.columns:
        raise SystemExit(f"SMILES column '{SMILES_COL}' not found. Columns: {list(df.columns)}")
    if DOCKING_COL not in df.columns:
        raise SystemExit(f"Docking column '{DOCKING_COL}' not found. Columns: {list(df.columns)}")

    smiles = df[SMILES_COL].astype(str).tolist()
    docking = pd.to_numeric(df[DOCKING_COL], errors="coerce")

    pic50_model = _load_booster(_REPO_ROOT / PIC50_MODEL)
    sol_model = _load_booster(_REPO_ROOT / SOL_MODEL)

    print(f"Loaded {len(df)} rows from {inp}")
    print("Predicting pIC50 and solubility...")

    pic50 = _predict_pic50(smiles, pic50_model, str(_REPO_ROOT / PIC50_SCALER))
    sol = _predict_sol(smiles, sol_model, str(_REPO_ROOT / SOL_SCALER))

    out_df = pd.DataFrame({
        "SMILES": smiles,
        "pIC50": pic50,
        "Solubility": sol,
        "Docking_Score": docking,
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[+] Wrote {len(out_df)} rows -> {out_path}")
    print(out_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
