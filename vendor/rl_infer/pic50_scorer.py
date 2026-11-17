"""PD1-PDL1 pIC50 XGBoost scorer for CVAE_RL-style rewards."""

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
PIC50_MODEL = _VENDOR_DIR / "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_model.ubj"
PIC50_SCALER = _VENDOR_DIR / "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_scaler.pkl"

PIC50_MIN = 4.01
PIC50_MAX = 11.0
PIC50_SCALE = 3.0  # exp(pIC50 / PIC50_SCALE), same style as SA in RL.py


class Pic50Predictor:
    """Load once, predict many (pass instance to get_reward)."""

    def __init__(
        self,
        model_path: str | Path = PIC50_MODEL,
        scaler_path: str | Path = PIC50_SCALER,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"pIC50 model not found: {self.model_path}")
        if not self.scaler_path.is_file():
            raise FileNotFoundError(f"pIC50 scaler not found: {self.scaler_path}")

        self._compute_features = get_compute_features()
        self._model = xgb.Booster()
        self._model.load_model(str(self.model_path))

    def predict(self, smiles: str) -> float:
        X, mask = self._compute_features([smiles], str(self.scaler_path))
        if not mask[0]:
            return float("nan")
        X = X[:, :2415]
        pred = float(self._model.predict(xgb.DMatrix(X))[0])
        return pred if math.isfinite(pred) else float("nan")


def predictPic50(smiles: str, predictor: Pic50Predictor) -> float:
    return predictor.predict(smiles)


def calculateScore(smiles: str, predictor: Pic50Predictor) -> float:
    """pIC50 reward: exp(pIC50 / PIC50_SCALE)."""
    pic50 = predictPic50(smiles, predictor)
    if not math.isfinite(pic50):
        return 0.0
    return float(math.exp(pic50 / PIC50_SCALE))


def normalizePic50(pic50: float) -> float:
    """Map raw pIC50 to [0, 1] using training-set calibration."""
    if not math.isfinite(pic50):
        return 0.0
    return float(max(0.0, min(1.0, (pic50 - PIC50_MIN) / (PIC50_MAX - PIC50_MIN))))
