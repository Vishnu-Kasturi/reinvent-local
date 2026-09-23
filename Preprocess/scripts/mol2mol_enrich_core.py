"""Shared Mol2Mol RL CSV enrichment (XGBoost pIC50 + solubility)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import RDLogger

from mol2mol_rl_columns import (
    MOL2MOL_INFERENCE_COLUMNS,
    MOL2MOL_OUTPUT_COLUMNS,
    MOL2MOL_RL_SCORING_COLUMNS,
)

RDLogger.DisableLog("rdApp.*")


def resolve_smiles_col(df: pd.DataFrame, preferred: str = "SMILES") -> str:
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


def enrich_mol2mol_dataframe(
    df_in: pd.DataFrame,
    base_dir: Path,
    reinvent4_dir: Path,
    smiles_col: str = "SMILES",
    xgb_device: str = "cuda:0",
) -> pd.DataFrame:
    """Repredict pIC50 + Solubility and merge with TOML scoring columns from input."""
    sys.path.insert(0, str(reinvent4_dir))
    from reinvent_plugins.components.nophyschem_features import (
        compute_features as compute_pic50_features,
    )
    from reinvent_plugins.components.pd1_pdl1_features import (
        compute_features as compute_sol_features,
    )

    smi_col = resolve_smiles_col(df_in, smiles_col)
    work = df_in.dropna(subset=[smi_col]).copy()
    smiles_list = work[smi_col].astype(str).tolist()

    pic50_model_path = base_dir / "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_model.ubj"
    pic50_scaler_path = base_dir / "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_scaler.pkl"
    sol_model_path = base_dir / "Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj"
    sol_scaler_path = base_dir / "Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl"

    pic50_model = _load_booster(pic50_model_path, xgb_device)
    sol_model = _load_booster(sol_model_path, xgb_device)

    pic50_values = _predict(smiles_list, pic50_model, pic50_scaler_path, compute_pic50_features)
    sol_values = _predict(smiles_list, sol_model, sol_scaler_path, compute_sol_features)

    out = pd.DataFrame({"SMILES": smiles_list})
    for col in MOL2MOL_RL_SCORING_COLUMNS:
        if col in work.columns:
            out[col] = work[col].values

    for col, values in zip(MOL2MOL_INFERENCE_COLUMNS, (pic50_values, sol_values)):
        out[col] = values

    # Stable column order; keep any extra input columns only if already in OUTPUT list
    ordered = [c for c in MOL2MOL_OUTPUT_COLUMNS if c in out.columns]
    return out[ordered]


def enrich_mol2mol_csv_file(
    input_csv: Path,
    output_csv: Path,
    base_dir: Path,
    reinvent4_dir: Path | None = None,
    smiles_col: str = "SMILES",
    xgb_device: str = "cuda:0",
) -> pd.DataFrame:
    if reinvent4_dir is None:
        reinvent4_dir = base_dir / "REINVENT4"
    df_in = pd.read_csv(input_csv, on_bad_lines="skip", engine="python")
    out = enrich_mol2mol_dataframe(
        df_in,
        base_dir=base_dir,
        reinvent4_dir=reinvent4_dir,
        smiles_col=smiles_col,
        xgb_device=xgb_device,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
