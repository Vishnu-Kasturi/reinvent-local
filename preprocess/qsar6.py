"""
XGBoost pIC50 Predictor v2
===========================
Features:
  - 11 pre-scaled physicochemical descriptors
  - 167 MACCS keys
  - 2048-bit ECFP4  (Morgan r=2)
  - 2048-bit ECFP6  (Morgan r=3)
  - ~200 RDKit 2D descriptors (topological, electronic, constitutional)

Fixes vs v1:
  - Stronger regularisation to close train/val gap
  - Richer feature set for better generalisation
  - Optionally stack LightGBM + XGB (--stack flag)

Usage
-----
  python xgb_pic50_v2.py
  python xgb_pic50_v2.py --stack          # stacked ensemble

Dependencies
------------
  pip install xgboost lightgbm scikit-learn rdkit pandas numpy
"""

import os, json, argparse, warnings, math
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
PHYSCHEM_COLS = [
    "mw","logp","hbd","hba","tpsa",
    "rot_bonds","rings","arom_rings",
    "heavy_atoms","frac_csp3","stereo",
]

# XGB — tighter regularisation
XGB_PARAMS = dict(
    n_estimators          = 3000,
    learning_rate         = 0.02,        # slower lr → better generalisation
    max_depth             = 5,           # shallower trees
    subsample             = 0.7,
    colsample_bytree      = 0.6,         # see fewer features per tree
    colsample_bylevel     = 0.7,
    min_child_weight      = 5,           # require more samples per leaf
    gamma                 = 0.2,
    reg_alpha             = 0.3,         # stronger L1
    reg_lambda            = 2.0,         # stronger L2
    objective             = "reg:squarederror",
    eval_metric           = "rmse",
    early_stopping_rounds = 100,
    tree_method           = "hist",
    device                = "cpu",
    random_state          = 42,
    n_jobs                = -1,
)


# ══════════════════════════════════════════════
# RDKIT 2D DESCRIPTOR BLOCK
# ══════════════════════════════════════════════
# Curated list — remove descriptors known to be unstable / always-zero
_EXCLUDE = {
    "Ipc",           # can blow up numerically
    "BCUT2D_MWHI","BCUT2D_MWLOW",
    "BCUT2D_CHGHI","BCUT2D_CHGLO",
    "BCUT2D_LOGPHI","BCUT2D_LOGPLOW",
    "BCUT2D_MRHI","BCUT2D_MRLOW",
}

RDKIT_DESC_FUNCS = [
    (name, func)
    for name, func in Descriptors.descList
    if name not in _EXCLUDE
]

def rdkit_descriptors(mol):
    vals = []
    for _, func in RDKIT_DESC_FUNCS:
        try:
            v = func(mol)
            vals.append(float(v) if (v is not None and np.isfinite(float(v))) else 0.0)
        except Exception:
            vals.append(0.0)
    return np.array(vals, dtype=np.float32)


# ══════════════════════════════════════════════
# FEATURE BUILDER
# ══════════════════════════════════════════════
def build_features(df: pd.DataFrame, desc_scaler=None, fit_scaler=False, use_rdkit=False):
    """
    Returns X (N, D) and optionally the fitted desc_scaler.
    desc_scaler is applied to the RDKit 2D descriptors only
    (MACCS/ECFP bits are already binary; physchem already pre-scaled).
    """
    print(f"  Building features for {len(df):,} molecules …")

    physchem = df[PHYSCHEM_COLS].fillna(0.0).values.astype(np.float32)

    maccs_list, ecfp4_list, ecfp6_list, rdkit_list = [], [], [], []

    for smi in df["canon_smiles"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            maccs_list.append(np.zeros(167,  dtype=np.float32))
            ecfp4_list.append(np.zeros(2048, dtype=np.float32))
            ecfp6_list.append(np.zeros(2048, dtype=np.float32))
            if use_rdkit:
                rdkit_list.append(np.zeros(len(RDKIT_DESC_FUNCS), dtype=np.float32))
            continue

        maccs_list.append(np.array(MACCSkeys.GenMACCSKeys(mol),   dtype=np.float32))
        ecfp4_list.append(np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048), dtype=np.float32))
        ecfp6_list.append(np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048), dtype=np.float32))
        if use_rdkit:
            rdkit_list.append(rdkit_descriptors(mol))

    maccs_arr  = np.stack(maccs_list)
    ecfp4_arr  = np.stack(ecfp4_list)
    ecfp6_arr  = np.stack(ecfp6_list)

    if use_rdkit:
        rdkit_arr  = np.stack(rdkit_list)
        # Scale RDKit descriptors (fit on train only)
        if fit_scaler:
            desc_scaler = StandardScaler()
            rdkit_arr = desc_scaler.fit_transform(rdkit_arr).astype(np.float32)
        elif desc_scaler is not None:
            rdkit_arr = desc_scaler.transform(rdkit_arr).astype(np.float32)
        X = np.concatenate([physchem, maccs_arr, ecfp4_arr, ecfp6_arr, rdkit_arr], axis=1)
        print(f"  Feature matrix : {X.shape}  "
              f"(physchem={physchem.shape[1]}, MACCS={maccs_arr.shape[1]}, "
              f"ECFP4={ecfp4_arr.shape[1]}, ECFP6={ecfp6_arr.shape[1]}, "
              f"RDKit2D={rdkit_arr.shape[1]})")
    else:
        X = np.concatenate([physchem, maccs_arr, ecfp4_arr, ecfp6_arr], axis=1)
        print(f"  Feature matrix : {X.shape}  "
              f"(physchem={physchem.shape[1]}, MACCS={maccs_arr.shape[1]}, "
              f"ECFP4={ecfp4_arr.shape[1]}, ECFP6={ecfp6_arr.shape[1]})")
    return X, desc_scaler


def metrics(y_true, y_pred, split=""):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  [{split:<5s}] RMSE {rmse:.4f}  MAE {mae:.4f}  R² {r2:.4f}")
    return rmse, mae, r2


# ══════════════════════════════════════════════
# OPTIONAL STACKED ENSEMBLE
# ══════════════════════════════════════════════
def train_stacked(X_train, y_train, X_val, y_val, X_test, y_test, out_dir):
    try:
        import lightgbm as lgb
    except ImportError:
        print("  lightgbm not installed — skipping stack (pip install lightgbm)")
        return None

    print("\n── Stacked ensemble (XGB + LGB) ──")

    lgb_params = dict(
        n_estimators=3000, learning_rate=0.02, max_depth=5,
        num_leaves=31, subsample=0.7, colsample_bytree=0.6,
        min_child_samples=20, reg_alpha=0.3, reg_lambda=2.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )

    # XGB (same params as main)
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=500)

    # Simple average blend
    for split, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        pred = (xgb_model.predict(X) + lgb_model.predict(X)) / 2
        metrics(y, pred, split)

    return xgb_model, lgb_model


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading data …")
    train_df = pd.read_csv(args.train)
    val_df   = pd.read_csv(args.val)
    test_df  = pd.read_csv(args.test)
    print(f"  Train {len(train_df):,}  Val {len(val_df):,}  Test {len(test_df):,}")

    print("\nBuilding features …")
    X_train, desc_scaler = build_features(train_df, fit_scaler=True, use_rdkit=args.use_rdkit)
    X_val,   _           = build_features(val_df,   desc_scaler=desc_scaler, use_rdkit=args.use_rdkit)
    X_test,  _           = build_features(test_df,  desc_scaler=desc_scaler, use_rdkit=args.use_rdkit)

    y_train = train_df["pic50"].values
    y_val   = val_df["pic50"].values
    y_test  = test_df["pic50"].values

    # ── Train XGB ──────────────────────────────────────────────
    print(f"\nTraining XGBoost …")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=200,
    )
    print(f"\n  Best iteration : {model.best_iteration}")

    # ── Metrics ────────────────────────────────────────────────
    print("\nMetrics (XGB solo):")
    train_rmse, train_mae, train_r2 = metrics(y_train, model.predict(X_train), "Train")
    val_rmse,   val_mae,   val_r2   = metrics(y_val,   model.predict(X_val),   "Val  ")
    test_rmse,  test_mae,  test_r2  = metrics(y_test,  model.predict(X_test),  "Test ")

    # ── Optional stack ─────────────────────────────────────────
    if args.stack:
        print("\nMetrics (XGB + LGB ensemble):")
        train_stacked(X_train, y_train, X_val, y_val, X_test, y_test, args.output_dir)

    # ── Save predictions ───────────────────────────────────────
    test_pred_df = test_df[["canon_smiles","pic50","zone","scaffold"]].copy()
    test_pred_df["pred_pic50"] = model.predict(X_test)
    test_pred_df["residual"]   = test_pred_df["pic50"] - test_pred_df["pred_pic50"]
    test_pred_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

    # ── Feature importance ─────────────────────────────────────
    feat_names = (
        PHYSCHEM_COLS
        + [f"MACCS_{i}"  for i in range(167)]
        + [f"ECFP4_{i}"  for i in range(2048)]
        + [f"ECFP6_{i}"  for i in range(2048)]
    )
    if args.use_rdkit:
        feat_names += [name for name, _ in RDKIT_DESC_FUNCS]
    imp     = model.feature_importances_
    top20   = sorted(zip(feat_names, imp.tolist()), key=lambda x: -x[1])[:20]
    print("\nTop-20 features:")
    for name, score in top20:
        print(f"  {name:<30s}  {score:.5f}")

    # ── Save artefacts ─────────────────────────────────────────
    model.save_model(os.path.join(args.output_dir, "xgb_model.ubj"))

    import pickle
    with open(os.path.join(args.output_dir, "desc_scaler.pkl"), "wb") as f:
        pickle.dump(desc_scaler, f)

    summary = {
        "best_iteration": model.best_iteration,
        "feature_dim": int(X_train.shape[1]),
        "train": {"rmse": train_rmse, "mae": train_mae, "r2": train_r2},
        "val":   {"rmse": val_rmse,   "mae": val_mae,   "r2": val_r2},
        "test":  {"rmse": test_rmse,  "mae": test_mae,  "r2": test_r2},
        "top20_features": top20,
    }
    with open(os.path.join(args.output_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",      default="data_splits_random/train.csv")
    parser.add_argument("--val",        default="data_splits_random/val.csv")
    parser.add_argument("--test",       default="data_splits_random/test.csv")
    parser.add_argument("--output_dir", default="run_xgb_v2")
    parser.add_argument("--stack",      action="store_true",
                        help="Also train LightGBM and blend predictions")
    parser.add_argument("--use_rdkit",  action="store_true",
                        help="Build and use ~200 RDKit 2D descriptors (disabled by default)")
    args = parser.parse_args()
    main(args)
