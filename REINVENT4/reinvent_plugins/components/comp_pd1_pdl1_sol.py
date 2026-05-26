"""
PD1-PDL1 Solubility Scoring Component for REINVENT4 (final_acc)
----------------------------------------------------------------
Uses the best PD1-PDL1 solubility XGBoost model (R² = 0.7771 on scaffold-stratified split).
Model: Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj
Features: 2427-dim (200 RDKit + 2048 ECFP4 + 167 MACCS + 12 physchem)

Normalization:
  The PD1-PDL1 solubility dataset Y ranges from about -13.17 to +2.14 (logS).
    0.0 → Y ≤ -13.17  (highly insoluble)
    1.0 → Y ≥  +2.14  (highly soluble)

TOML usage
----------
[[stage.scoring.component]]
[stage.scoring.component.PD1PDL1Sol]
[[stage.scoring.component.PD1PDL1Sol.endpoint]]
name               = "PD1PDL1Sol"
weight             = 1.0
params.model_path  = ["Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj"]
params.scaler_path = ["Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl"]
[[stage.scoring.component.PD1PDL1Sol.endpoint]]
name               = "PD1PDL1Sol_raw"
weight             = 0.0
params.model_path  = ["Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj"]
params.scaler_path = ["Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl"]
"""
from __future__ import annotations

__all__ = ["PD1PDL1Sol"]

import logging
import os
import traceback
from typing import List

import numpy as np
import xgboost as xgb
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from .pd1_pdl1_features import compute_features, EXPECTED_FEATURE_DIM

logger = logging.getLogger("reinvent")

# Normalization calibrated from PD1-PDL1 solubility training set
SOL_MIN   = -13.17
SOL_MAX   =   2.14
SOL_RANGE = SOL_MAX - SOL_MIN  # ~15.31


@add_tag("__parameters")
@dataclass
class Parameters:
    model_path: List[str]
    scaler_path: List[str]


@add_tag("__component")
class PD1PDL1Sol:
    """
    REINVENT4 scoring component: XGBoost PD1-PDL1 aqueous solubility predictor.
    Best model: R² = 0.7771, RMSE = 0.9472 (scaffold-stratified split).

    Returns TWO endpoints:
      [0] Normalized [0,1] reward  →  'PD1PDL1Sol'     (weight=1.0)
      [1] Raw log S value          →  'PD1PDL1Sol_raw'  (weight=0.0, logging only)
    """

    def __init__(self, params: Parameters):
        self.model_path  = params.model_path[0]
        self.scaler_path = params.scaler_path[0]
        self.number_of_endpoints = 2

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[PD1PDL1Sol] Model not found: {self.model_path}")
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
        logger.info(f"[PD1PDL1Sol] Loaded model ({EXPECTED_FEATURE_DIM} features): {self.model_path}")

        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"[PD1PDL1Sol] Scaler not found: {self.scaler_path}")
        logger.info(f"[PD1PDL1Sol] Loaded scaler: {self.scaler_path}")

    def __call__(self, smilies: List[str]) -> ComponentResults:
        n = len(smilies)
        try:
            X, valid_mask = compute_features(smilies, self.scaler_path)

            if X.shape[1] != EXPECTED_FEATURE_DIM:
                logger.error(f"[PD1PDL1Sol] Feature dim mismatch: {X.shape[1]} vs {EXPECTED_FEATURE_DIM}")
                return ComponentResults([np.full(n, np.nan, dtype=np.float32)])

            nan_rows = np.where(~np.isfinite(X).all(axis=1))[0]
            if len(nan_rows) > 0:
                X[nan_rows] = 0.0
                for idx in nan_rows:
                    valid_mask[idx] = False

            dmatrix = xgb.DMatrix(X)
            raw_preds = self.model.predict(dmatrix)

            scores_norm = np.full(n, np.nan, dtype=np.float32)
            scores_raw  = np.full(n, np.nan, dtype=np.float32)
            n_valid = 0

            for i in range(n):
                if not valid_mask[i]:
                    continue
                raw = float(raw_preds[i])
                if not np.isfinite(raw):
                    continue
                scores_norm[i] = float(np.clip((raw - SOL_MIN) / SOL_RANGE, 0.0, 1.0))
                scores_raw[i]  = raw
                n_valid += 1

            valid_scores = scores_norm[np.isfinite(scores_norm)]
            valid_raw    = scores_raw[np.isfinite(scores_raw)]
            if len(valid_scores) > 0:
                logger.info(
                    f"[PD1PDL1Sol] Batch: {n} | valid={n_valid} | "
                    f"mean_norm={valid_scores.mean():.4f} | mean_logS={valid_raw.mean():.3f}"
                )
            else:
                logger.warning(f"[PD1PDL1Sol] ALL {n} molecules invalid. Returning NaN.")

            return ComponentResults([scores_norm, scores_raw])

        except Exception as exc:
            logger.error(f"[PD1PDL1Sol] Error: {exc}\n{traceback.format_exc()}")
            return ComponentResults([np.full(n, np.nan, dtype=np.float32)])
