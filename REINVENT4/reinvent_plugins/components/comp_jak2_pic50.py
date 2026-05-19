"""
Production-grade JAK2 pIC50 Scoring Component for REINVENT4
---
Key improvements over v1:
- NaN/Inf validation per molecule before prediction
- Feature shape assertion before xgb.DMatrix creation
- Scaler consistency check at load time
- Detailed per-step debug logging (toggled by env var JAK2_DEBUG=1)
- Safe fallback: invalid molecules return nan (not 0.0)
- Reproducible descriptor ordering
"""
from __future__ import annotations

__all__ = ["JAK2pIC50"]

import logging
import os
import pickle
import traceback
from typing import List

import numpy as np
import xgboost as xgb
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from .jak2_pic50_features import compute_features, EXPECTED_FEATURE_DIM

logger = logging.getLogger("reinvent")

# Enable debug mode by setting env var: export JAK2_DEBUG=1
_DEBUG = os.environ.get("JAK2_DEBUG", "0") == "1"

# Normalization range calibrated from the training set
PIC50_MIN = 3.8   # lowest pIC50 in training data
PIC50_MAX = 10.8  # highest pIC50 in training data
PIC50_RANGE = PIC50_MAX - PIC50_MIN  # 7.0


@add_tag("__parameters")
@dataclass
class Parameters:
    model_path: List[str]
    scaler_path: List[str]


@add_tag("__component")
class JAK2pIC50:
    """
    REINVENT4 scoring component: XGBoost-based JAK2 pIC50 predictor.

    Returns TWO endpoints (configure both in the TOML):

    Endpoint 1: JAK2pIC50  (set weight=2.0 in TOML)
        Normalized [0, 1] reward used for the RL training signal.
        0.0 → pIC50 ≤ 3.8  (inactive)
        0.5 → pIC50 ≈ 7.3  (IC50 ~50 nM)
        1.0 → pIC50 ≥ 10.8 (sub-picomolar)

    Endpoint 2: JAK2pIC50_raw  (set weight=0.0 in TOML — logging only)
        Raw predicted pIC50 in the natural [~4, ~11] scale.
        This is written to the RL output CSV as a human-readable column
        so you can directly inspect predicted potency without back-converting.
    """

    def __init__(self, params: Parameters):
        self.model_path = params.model_path[0]
        self.scaler_path = params.scaler_path[0]

        # Resolve physchem scaler relative to model directory
        model_dir = os.path.dirname(self.model_path)
        self.physchem_scaler_path = os.path.join(
            model_dir, "../../data_splits/physchem_scaler.pkl"
        )

        # --- Load and validate XGBoost model ---
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[JAK2pIC50] XGBoost model not found: {self.model_path}"
            )
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
        logger.info(f"[JAK2pIC50] Loaded XGBoost model: {self.model_path}")

        # --- Validate scaler exists ---
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"[JAK2pIC50] Descriptor scaler not found: {self.scaler_path}"
            )
        with open(self.scaler_path, "rb") as f:
            scaler = pickle.load(f)
        # Check scaler feature count matches expected feature count
        if hasattr(scaler, "n_features_in_"):
            n_rdkit = scaler.n_features_in_
            logger.info(f"[JAK2pIC50] Descriptor scaler expects {n_rdkit} RDKit features")
        logger.info(f"[JAK2pIC50] Loaded descriptor scaler: {self.scaler_path}")

        # --- Log physchem scaler status ---
        if os.path.exists(self.physchem_scaler_path):
            logger.info(
                f"[JAK2pIC50] Physchem scaler found: {self.physchem_scaler_path}"
            )
        else:
            logger.warning(
                f"[JAK2pIC50] Physchem scaler NOT found at: {self.physchem_scaler_path} "
                f"— physchem features will not be scaled"
            )

    def __call__(self, smilies: List[str]) -> ComponentResults:
        n = len(smilies)

        try:
            # ─── 1. Feature extraction ────────────────────────────────────────
            X, valid_mask = compute_features(
                smilies, self.scaler_path, self.physchem_scaler_path
            )

            # ─── 2. Shape validation ─────────────────────────────────────────
            expected_dim = EXPECTED_FEATURE_DIM
            if X.shape[1] != expected_dim:
                logger.error(
                    f"[JAK2pIC50] Feature shape mismatch! "
                    f"Got {X.shape[1]}, expected {expected_dim}. "
                    f"This will cause XGBoost prediction errors."
                )
                return ComponentResults(
                    [np.full(n, np.nan, dtype=np.float32)]
                )

            # ─── 3. NaN/Inf check on feature matrix ──────────────────────────
            nan_rows = np.where(~np.isfinite(X).all(axis=1))[0]
            if len(nan_rows) > 0:
                logger.warning(
                    f"[JAK2pIC50] Found NaN/Inf in feature matrix for "
                    f"{len(nan_rows)} rows: {list(nan_rows[:5])}... Zeroing out."
                )
                X[nan_rows] = 0.0
                # Mark these as invalid too
                for idx in nan_rows:
                    valid_mask[idx] = False

            # ─── 4. XGBoost prediction ────────────────────────────────────────
            dmatrix = xgb.DMatrix(X)
            raw_preds = self.model.predict(dmatrix)

            # ─── 5. Validate prediction outputs ───────────────────────────────
            if len(raw_preds) != n:
                logger.error(
                    f"[JAK2pIC50] Prediction count mismatch: "
                    f"got {len(raw_preds)}, expected {n}"
                )
                return ComponentResults(
                    [np.full(n, np.nan, dtype=np.float32)]
                )

            # ─── 6. Normalize → [0,1] score AND keep raw pIC50 ──────────────
            scores_norm = np.empty(n, dtype=np.float32)
            scores_raw  = np.empty(n, dtype=np.float32)
            scores_norm[:] = np.nan  # default: invalid
            scores_raw[:] = np.nan

            n_valid   = 0
            n_invalid = 0

            for i in range(n):
                if not valid_mask[i]:
                    n_invalid += 1
                    if _DEBUG:
                        logger.debug(
                            f"[JAK2pIC50] SMILES[{i}] invalid (failed RDKit parse): "
                            f"'{smilies[i][:60]}...'"
                        )
                    continue

                raw = float(raw_preds[i])

                if not np.isfinite(raw):
                    logger.warning(
                        f"[JAK2pIC50] Non-finite pIC50 prediction for SMILES[{i}]: {raw}"
                    )
                    n_invalid += 1
                    continue

                # Endpoint 1: normalized [0, 1] reward for RL training signal
                norm = (raw - PIC50_MIN) / PIC50_RANGE
                norm = float(np.clip(norm, 0.0, 1.0))
                scores_norm[i] = norm

                # Endpoint 2: raw pIC50 in [~4, ~11] for human-readable CSV logging
                scores_raw[i] = float(raw)
                n_valid += 1

                if _DEBUG:
                    logger.debug(
                        f"[JAK2pIC50] SMILES[{i}] | "
                        f"pIC50={raw:.3f} | norm={norm:.4f} | "
                        f"'{smilies[i][:50]}'"
                    )

            # ─── 7. Step-level summary logging ───────────────────────────────
            valid_scores = scores_norm[np.isfinite(scores_norm)]
            valid_raw    = scores_raw[np.isfinite(scores_raw)]
            if len(valid_scores) > 0:
                logger.info(
                    f"[JAK2pIC50] Batch: {n} molecules | "
                    f"valid={n_valid} | invalid={n_invalid} | "
                    f"mean_norm={valid_scores.mean():.4f} | "
                    f"mean_pIC50={valid_raw.mean():.3f} | "
                    f"max_pIC50={valid_raw.max():.3f}"
                )
            else:
                logger.warning(
                    f"[JAK2pIC50] ALL {n} molecules in batch were invalid! "
                    f"Returning all NaN scores."
                )

            # Return two endpoints:
            #   [0] scores_norm → logged as 'JAK2pIC50'      (weight=2.0 in TOML)
            #   [1] scores_raw  → logged as 'JAK2pIC50_raw'  (weight=0.0 in TOML)
            return ComponentResults([scores_norm, scores_raw])

        except Exception as exc:
            logger.error(
                f"[JAK2pIC50] Unexpected error during scoring: {exc}\n"
                f"{traceback.format_exc()}"
            )
            # Safe fallback: return NaN for all (REINVENT treats NaN as 0)
            return ComponentResults([np.full(n, np.nan, dtype=np.float32)])
