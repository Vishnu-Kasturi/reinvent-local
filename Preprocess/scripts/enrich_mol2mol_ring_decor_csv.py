#!/usr/bin/env python3
"""
Enrich a Mol2Mol ring-decor RL summary CSV for select_balanced_mol2mol_v2.py.

Reads REINVENT output from pd1_pdl1_mol2mol_ring_decor_dock_tyr.toml (5 rewards),
keeps TOML score columns, adds XGBoost pIC50 + Solubility, and maps:
  Docking_Score       <- DockingAffinity_raw (raw)  [kcal/mol]
  Tyrosine_PiStacking <- tyr_pi_stacking (TyrInteractionReward) or Tyr count raw

Edit CONFIG, then from repo root:
  python Preprocess/scripts/enrich_mol2mol_ring_decor_csv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── CONFIG ───────────────────────────────────────────────────────────────────
RUN_NAME = "mol2mol_ring_decor"
INPUT_CSV = _REPO_ROOT / "results" / f"{RUN_NAME}_1.csv"
OUTPUT_CSV = _REPO_ROOT / "iict_libinvent" / f"{RUN_NAME}_enriched.csv"
SMILES_COL = "SMILES"
XGB_DEVICE = "cuda:0"
# ─────────────────────────────────────────────────────────────────────────────

PIC50_MODEL = _REPO_ROOT / "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_model.ubj"
PIC50_SCALER = _REPO_ROOT / "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_scaler.pkl"
SOL_MODEL = _REPO_ROOT / "Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj"
SOL_SCALER = _REPO_ROOT / "Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl"

# Ring-decor TOML endpoint names (transformed + raw where REINVENT writes them)
TOML_SCORE_COLUMNS = [
    "NearLead",
    "NearLead (raw)",
    "DockingReward",
    "DockingReward (raw)",
    "DockingAffinity_raw",
    "DockingAffinity_raw (raw)",
    "TyrInteractionReward",
    "TyrInteractionReward (raw)",
    "TyrInteractionCount_raw",
    "TyrInteractionCount_raw (raw)",
    "tyr_pi_stacking (TyrInteractionReward)",
    "AromaticRings",
    "AromaticRings (raw)",
    "RingHetero",
    "RingHetero (raw)",
]

DOCKING_SCORE_ALIASES = [
    "DockingAffinity_raw (raw)",
    "DockingAffinity_raw",
    "Docking_Score",
    "affinity",
    "DockingReward (raw)",
]

TYR_STACK_ALIASES = [
    "tyr_pi_stacking (TyrInteractionReward)",
    "Tyrosine_PiStacking",
    "TyrInteractionCount_raw (raw)",
    "TyrInteractionCount_raw",
    "TyrInteractionReward (raw)",
]


def _resolve_smiles_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for c in df.columns:
        if c.lower() == preferred.lower():
            return c
    raise ValueError(f"SMILES column not found. Columns: {list(df.columns)}")


def _first_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _load_booster(path: Path, device: str) -> xgb.Booster:
    booster = xgb.Booster({"device": device})
    booster.load_model(str(path))
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
    sys.path.insert(0, str(_REPO_ROOT / "REINVENT4"))
    from reinvent_plugins.components.nophyschem_features import (
        compute_features as compute_pic50_features,
    )
    from reinvent_plugins.components.pd1_pdl1_features import (
        compute_features as compute_sol_features,
    )

    if not INPUT_CSV.is_file():
        raise SystemExit(f"Input not found: {INPUT_CSV}")

    df_in = pd.read_csv(INPUT_CSV, on_bad_lines="skip", engine="python")
    smi_col = _resolve_smiles_col(df_in, SMILES_COL)
    work = df_in.dropna(subset=[smi_col]).copy()
    smiles_list = work[smi_col].astype(str).tolist()
    print(f"Loaded {len(smiles_list)} rows from {INPUT_CSV}")

    pic50_model = _load_booster(PIC50_MODEL, XGB_DEVICE)
    sol_model = _load_booster(SOL_MODEL, XGB_DEVICE)
    pic50_values = _predict(smiles_list, pic50_model, PIC50_SCALER, compute_pic50_features)
    sol_values = _predict(smiles_list, sol_model, SOL_SCALER, compute_sol_features)

    out = pd.DataFrame({"SMILES": smiles_list})
    for col in TOML_SCORE_COLUMNS:
        if col in work.columns:
            out[col] = work[col].values

    out["pIC50"] = pic50_values
    out["Solubility"] = sol_values

    dock_col = _first_column(work, DOCKING_SCORE_ALIASES)
    if dock_col is None:
        raise SystemExit(
            "No docking affinity column found. Re-run RL with DockingAffinity_raw (weight 0) "
            f"in the TOML, or add one of: {DOCKING_SCORE_ALIASES}"
        )
    out["Docking_Score"] = pd.to_numeric(work[dock_col], errors="coerce").values
    if dock_col not in ("DockingAffinity_raw (raw)", "DockingAffinity_raw"):
        print(f"  note: Docking_Score taken from '{dock_col}' (not kcal/mol affinity)")

    tyr_col = _first_column(work, TYR_STACK_ALIASES)
    if tyr_col is None:
        raise SystemExit(
            "No tyrosine stacking column found. Expected "
            "'tyr_pi_stacking (TyrInteractionReward)' or TyrInteractionCount_raw (raw)."
        )
    out["Tyrosine_PiStacking"] = pd.to_numeric(work[tyr_col], errors="coerce").values
    if tyr_col != "tyr_pi_stacking (TyrInteractionReward)":
        print(f"  note: Tyrosine_PiStacking taken from '{tyr_col}'")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"[+] Wrote {len(out)} rows -> {OUTPUT_CSV}")
    print(f"    Columns: {list(out.columns)}")


if __name__ == "__main__":
    main()
