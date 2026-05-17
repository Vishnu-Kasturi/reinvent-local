from __future__ import annotations

__all__ = ["JAK2pIC50"]
from typing import List
import logging
import os

import numpy as np
import xgboost as xgb
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from .jak2_pic50_features import compute_features

logger = logging.getLogger("reinvent")

PIC50_MIN = 3.8
PIC50_MAX = 10.8

@add_tag("__parameters")
@dataclass
class Parameters:
    model_path: List[str]
    scaler_path: List[str]

@add_tag("__component")
class JAK2pIC50:
    def __init__(self, params: Parameters):
        self.model_path = params.model_path[0]
        self.scaler_path = params.scaler_path[0]
        self.physchem_scaler_path = os.path.join(os.path.dirname(self.model_path), "../../data_splits/physchem_scaler.pkl")
        
        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)

    def __call__(self, smilies: List[str]) -> ComponentResults:
        X, valid_mask = compute_features(smilies, self.scaler_path, self.physchem_scaler_path)
        
        # Make predictions
        dmatrix = xgb.DMatrix(X)
        preds = self.model.predict(dmatrix)
        
        # Normalize and mask invalid
        scores = []
        for i, pred in enumerate(preds):
            if not valid_mask[i]:
                scores.append(np.nan)
            else:
                norm_score = (pred - PIC50_MIN) / (PIC50_MAX - PIC50_MIN)
                norm_score = max(0.0, min(1.0, float(norm_score)))
                scores.append(norm_score)
                
        return ComponentResults([np.array(scores, dtype=np.float32)])
