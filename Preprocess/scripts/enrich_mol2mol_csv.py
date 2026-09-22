#!/usr/bin/env python3
"""
Enrich a Mol2Mol REINVENT RL summary CSV with pIC50 and solubility predictions.

Keeps scaffold-hop / docking / tyrosine / physchem reward columns from the RL run
and appends model-predicted pIC50 and Solubility.

Edit paths below, then from repo root:
  python Preprocess/scripts/enrich_mol2mol_csv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# ── EDIT THESE ────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/genai/Vishnu/psearch-master/reinvent-local-main")

INPUT_CSV = BASE_DIR / "results" / "mol2mol_scaffold_hop_1.csv"
OUTPUT_CSV = BASE_DIR / "iict_libinvent" / "mol2mol_scaffold_hop_1_enriched.csv"

XGB_DEVICE = "cuda:0"  # use "cpu" if no GPU
SMILES_COL = "SMILES"
# ─────────────────────────────────────────────────────────────────────────────

PIC50_MODEL = BASE_DIR / "Preprocess" / "final_acc" / "pd1_pdl1_pic50_final_acc_model.ubj"
PIC50_SCALER = BASE_DIR / "Preprocess" / "final_acc" / "pd1_pdl1_pic50_final_acc_scaler.pkl"
SOL_MODEL = BASE_DIR / "Preprocess" / "final_acc" / "pd1_pdl1_sol_final_acc_model.ubj"
SOL_SCALER = BASE_DIR / "Preprocess" / "final_acc" / "pd1_pdl1_sol_final_acc_scaler.pkl"

REINVENT4_DIR = BASE_DIR / "REINVENT4"

# Columns from pd1_pdl1_mol2mol_scaffold_hop_dock_tyr.toml scoring endpoints only.
# Missing columns are skipped. pIC50 / Solubility are appended after SMILES.
OUTPUT_COLUMNS = [
    "SMILES",
    "ScaffoldHop",
    "ScaffoldHop (raw)",
    "DockingReward",
    "DockingReward (raw)",
    "DockingAffinity_raw",
    "DockingAffinity_raw (raw)",
    "TyrInteractionReward",
    "TyrInteractionReward (raw)",
    "TyrInteractionCount_raw",
    "TyrInteractionCount_raw (raw)",
    "tyr_pi_stacking (TyrInteractionReward)",
    "LowCsp3",
    "LowCsp3 (raw)",
    "LowRotBonds",
    "LowRotBonds (raw)",
    "AromaticRings_2_4",
    "AromaticRings_2_4 (raw)",
    "MultiRing",
    "MultiRing (raw)",
    "pIC50",
    "Solubility",
]

sys.path.insert(0, str(REINVENT4_DIR))
try:
    from reinvent_plugins.components.nophyschem_features import (
        compute_features as compute_pic50_features,
    )
    from reinvent_plugins.components.pd1_pdl1_features import (
        compute_features as compute_sol_features,
    )
except ImportError as exc:
    raise RuntimeError(
        f"Could not import REINVENT4 feature modules from {REINVENT4_DIR}"
    ) from exc


def _resolve_smiles_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    matches = [c for c in df.columns if c.lower() == preferred.lower()]
    if not matches:
        raise ValueError(f"SMILES column not found. Available columns: {list(df.columns)}")
    return matches[0]


def _load_booster(model_path: Path, device: str) -> xgb.Booster:
    booster = xgb.Booster({"device": device})
    booster.load_model(str(model_path))
    return booster


def _predict(
    smiles: list[str],
    model: xgb.Booster,
    scaler_path: Path,
    feature_fn,
) -> list[float]:
    features, mask = feature_fn(smiles, str(scaler_path))
    preds = model.predict(xgb.DMatrix(features, nthread=-1))
    return [
        round(float(preds[i]), 4) if mask[i] and np.isfinite(preds[i]) else np.nan
        for i in range(len(smiles))
    ]


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    print("=================================================================")
    print("  Mol2Mol RL CSV enrichment (pIC50 + Solubility inference)       ")
    print("=================================================================\n")
    print(f"--> Loading dataset from: {INPUT_CSV}")

    df_in = pd.read_csv(INPUT_CSV, on_bad_lines="skip", engine="python")
    smiles_col = _resolve_smiles_col(df_in, SMILES_COL)

    df_in = df_in.dropna(subset=[smiles_col]).copy()
    df_in.reset_index(drop=True, inplace=True)
    smiles_list = df_in[smiles_col].astype(str).tolist()
    print(f"--> Found {len(smiles_list)} valid rows after cleaning.")

    print("--> Loading XGBoost models...")
    pic50_model = _load_booster(PIC50_MODEL, XGB_DEVICE)
    sol_model = _load_booster(SOL_MODEL, XGB_DEVICE)

    print("--> Predicting pIC50...")
    pic50_values = _predict(smiles_list, pic50_model, PIC50_SCALER, compute_pic50_features)

    print("--> Predicting solubility (logS)...")
    sol_values = _predict(smiles_list, sol_model, SOL_SCALER, compute_sol_features)

    print("--> Assembling output...")
    scoring_cols = [c for c in OUTPUT_COLUMNS if c not in ("SMILES", "pIC50", "Solubility")]
    out = pd.DataFrame({"SMILES": smiles_list})
    for col in scoring_cols:
        if col in df_in.columns:
            out[col] = df_in[col].values

    out["pIC50"] = pic50_values
    out["Solubility"] = sol_values

    missing = [c for c in scoring_cols if c not in df_in.columns]
    if missing:
        print(f"--> Note: TOML scoring columns not in input (skipped): {missing}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSuccessfully wrote {len(out)} rows -> {OUTPUT_CSV}")
    print(f"Columns: {list(out.columns)}")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
