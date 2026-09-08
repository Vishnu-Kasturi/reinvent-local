import math
import os.path as op
import sys

import xgboost as xgb

_bst = None
_scaler_path = None
_compute_features = None

SOL_MIN = -13.17


def readModel(
    model_path=None,
    scaler_path=None,
):
    global _bst, _scaler_path, _compute_features
    base = op.dirname(__file__)
    vendor = op.dirname(base)
    if model_path is None:
        model_path = op.join(vendor, "Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj")
    if scaler_path is None:
        scaler_path = op.join(vendor, "Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl")

    sys.path.insert(0, vendor)
    from pd1_pdl1_features import compute_features

    _compute_features = compute_features
    _scaler_path = scaler_path
    _bst = xgb.Booster()
    _bst.load_model(model_path)


def calculateScore(smiles):
    if _bst is None:
        readModel()

    X, mask = _compute_features([smiles], _scaler_path)
    if not mask[0]:
        return float("nan")
    pred = float(_bst.predict(xgb.DMatrix(X))[0])
    if not math.isfinite(pred):
        return float("nan")
    return pred
