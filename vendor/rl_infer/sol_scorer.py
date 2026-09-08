import math
import os.path as op

import xgboost as xgb

from pd1_pdl1_features import compute_features

_bst = None
_scaler_path = None
_RL_INFER_DIR = op.dirname(op.abspath(__file__))
_VENDOR_DIR = op.dirname(_RL_INFER_DIR)

SOL_MIN = -13.17


def _default_model_path():
    for root in (_RL_INFER_DIR, _VENDOR_DIR):
        p = op.join(root, "Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj")
        if op.isfile(p):
            return p
    return op.join(_VENDOR_DIR, "Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj")


def _default_scaler_path():
    for root in (_RL_INFER_DIR, _VENDOR_DIR):
        p = op.join(root, "Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl")
        if op.isfile(p):
            return p
    return op.join(_VENDOR_DIR, "Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl")


def readModel(model_path=None, scaler_path=None):
    global _bst, _scaler_path
    if model_path is None:
        model_path = _default_model_path()
    if scaler_path is None:
        scaler_path = _default_scaler_path()

    _scaler_path = scaler_path
    _bst = xgb.Booster()
    _bst.load_model(model_path)


def calculateScore(smiles):
    if _bst is None:
        readModel()

    X, mask = compute_features([smiles], _scaler_path)
    if not mask[0]:
        return float("nan")
    pred = float(_bst.predict(xgb.DMatrix(X))[0])
    if not math.isfinite(pred):
        return float("nan")
    return pred
