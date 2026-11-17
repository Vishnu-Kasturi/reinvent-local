"""PD1-PDL1 solubility (logS) XGBoost scorer for CVAE_RL-style rewards."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import xgboost as xgb

_INFER_DIR = Path(__file__).resolve().parent
if str(_INFER_DIR) not in sys.path:
    sys.path.insert(0, str(_INFER_DIR))

from qsar_features import get_compute_features

_VENDOR_DIR = _INFER_DIR.parent
SOL_MODEL = _VENDOR_DIR / "Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj"
SOL_SCALER = _VENDOR_DIR / "Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl"

SOL_MIN = -13.17
SOL_MAX = 2.14
SOL_SCALE = 3.0


class SolPredictor:
    """Load once, predict many (pass instance to get_reward)."""

    def __init__(
        self,
        model_path: str | Path = SOL_MODEL,
        scaler_path: str | Path = SOL_SCALER,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Sol model not found: {self.model_path}")
        if not self.scaler_path.is_file():
            raise FileNotFoundError(f"Sol scaler not found: {self.scaler_path}")

        self._compute_features = get_compute_features()
        self._model = xgb.Booster()
        self._model.load_model(str(self.model_path))

    def predict(self, smiles: str) -> float:
        X, mask = self._compute_features([smiles], str(self.scaler_path))
        if not mask[0]:
            return float("nan")
        pred = float(self._model.predict(xgb.DMatrix(X))[0])
        return pred if math.isfinite(pred) else float("nan")


def predictSolubility(smiles: str, predictor: SolPredictor) -> float:
    return predictor.predict(smiles)


def calculateScore(smiles: str, predictor: SolPredictor) -> float:
    """
    Solubility reward on shifted scale so exp() behaves like SA:
      exp((logS - SOL_MIN) / SOL_SCALE)
    """
    logs = predictSolubility(smiles, predictor)
    if not math.isfinite(logs):
        return 0.0
    shifted = logs - SOL_MIN
    return float(math.exp(shifted / SOL_SCALE))


def normalizeSolubility(logs: float) -> float:
    """Map raw logS to [0, 1] using training-set calibration."""
    if not math.isfinite(logs):
        return 0.0
    return float(max(0.0, min(1.0, (logs - SOL_MIN) / (SOL_MAX - SOL_MIN))))
