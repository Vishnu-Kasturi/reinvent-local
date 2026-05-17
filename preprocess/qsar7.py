"""
QSAR 7: Advanced Stacked Ensemble with Zone-Aware Weighting
============================================================
Architecture:
  1. Base Models:
     - XGBoost Regressor (Gradient Boosting)
     - LightGBM Regressor (Leaf-wise Boosting)
     - RandomForest Regressor (Bagging)
     - ExtraTrees Regressor (Extremely Randomized Trees)
  2. Meta-Learner:
     - Ridge Regression (Stacked on OOF predictions)
  3. Weighting:
     - Zone-aware sample weights: low=4.0, bulk=1.0, high=3.0

Features:
  - 11 PhysChem descriptors
  - 167 MACCS keys
  - 2048-bit ECFP4 + 2048-bit ECFP6
  - ~200 RDKit 2D descriptors (Standardized)

Usage:
  python qsar7.py
"""

import os, json, argparse, warnings, math, pickle
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PHYSCHEM_COLS = [
    "mw","logp","hbd","hba","tpsa",
    "rot_bonds","rings","arom_rings",
    "heavy_atoms","frac_csp3","stereo",
]

ZONE_WEIGHTS = {"low": 4.0, "bulk": 1.0, "high": 3.0}

# ══════════════════════════════════════════════
# FEATURE BUILDER
# ══════════════════════════════════════════════
_EXCLUDE = {
    "Ipc", "BCUT2D_MWHI","BCUT2D_MWLOW","BCUT2D_CHGHI",
    "BCUT2D_CHGLO","BCUT2D_LOGPHI","BCUT2D_LOGPLOW","BCUT2D_MRHI","BCUT2D_MRLOW",
}
RDKIT_DESC_FUNCS = [(n, f) for n, f in Descriptors.descList if n not in _EXCLUDE]

def get_descriptors(mol):
    vals = []
    for _, func in RDKIT_DESC_FUNCS:
        try:
            v = func(mol)
            vals.append(float(v) if (v is not None and np.isfinite(float(v))) else 0.0)
        except:
            vals.append(0.0)
    return np.array(vals, dtype=np.float32)

def build_features(df, desc_scaler=None, fit_scaler=False):
    print(f"  Building features for {len(df):,} molecules …")
    physchem = df[PHYSCHEM_COLS].fillna(0.0).values.astype(np.float32)
    maccs, ecfp4, ecfp6, rdkit = [], [], [], []

    for smi in df["canon_smiles"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            maccs.append(np.zeros(167, 32)); ecfp4.append(np.zeros(2048, 32))
            ecfp6.append(np.zeros(2048, 32)); rdkit.append(np.zeros(len(RDKIT_DESC_FUNCS), 32))
            continue
        maccs.append(np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32))
        ecfp4.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048), dtype=np.float32))
        ecfp6.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048), dtype=np.float32))
        rdkit.append(get_descriptors(mol))

    m_arr, e4_arr, e6_arr, r_arr = np.stack(maccs), np.stack(ecfp4), np.stack(ecfp6), np.stack(rdkit)
    
    if fit_scaler:
        desc_scaler = StandardScaler()
        r_arr = desc_scaler.fit_transform(r_arr)
    elif desc_scaler is not None:
        r_arr = desc_scaler.transform(r_arr)

    X = np.concatenate([physchem, m_arr, e4_arr, e6_arr, r_arr], axis=1)
    return X, desc_scaler

def get_weights(df):
    if "zone" not in df.columns:
        return np.ones(len(df))
    return np.array([ZONE_WEIGHTS.get(z, 1.0) for z in df["zone"]])

# ══════════════════════════════════════════════
# STACKING ENGINE
# ══════════════════════════════════════════════
class QsarStack:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            "xgb": xgb.XGBRegressor(n_estimators=2000, learning_rate=0.02, max_depth=6, 
                                    subsample=0.8, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=2.0,
                                    random_state=random_state, n_jobs=-1, tree_method="hist"),
            "lgb": lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, max_depth=6, 
                                     subsample=0.7, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=2.0,
                                     random_state=random_state, n_jobs=-1, verbose=-1),
            "rf":  RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=2,
                                         max_features="sqrt", random_state=random_state, n_jobs=-1),
            "et":  ExtraTreesRegressor(n_estimators=500, max_depth=15, min_samples_leaf=2,
                                       max_features="sqrt", random_state=random_state, n_jobs=-1)
        }
        self.meta = RidgeCV(alphas=[0.1, 1.0, 10.0])

    def fit(self, X, y, weights):
        print(f"\n  Training Stacked Ensemble (5-fold OOF) …")
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros((len(X), len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            print(f"    Fitting {name} …")
            for train_idx, val_idx in kf.split(X):
                X_t, y_t, w_t = X[train_idx], y[train_idx], weights[train_idx]
                X_v = X[val_idx]
                
                if name in ["xgb", "lgb"]:
                    model.fit(X_t, y_t, sample_weight=w_t)
                else:
                    model.fit(X_t, y_t, sample_weight=w_t)
                oof_preds[val_idx, i] = model.predict(X_v)
            
            # Final fit on all data
            model.fit(X, y, sample_weight=weights)
        
        self.meta.fit(oof_preds, y)
        print(f"    Meta-learner weights: {self.meta.coef_.round(4)}")

    def predict(self, X):
        base_preds = np.column_stack([m.predict(X) for m in self.models.values()])
        return self.meta.predict(base_preds)

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data_splits_random/train.csv")
    parser.add_argument("--val",   default="data_splits_random/val.csv")
    parser.add_argument("--test",  default="data_splits_random/test.csv")
    parser.add_argument("--output_dir", default="run_qsar7")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading data …")
    train_df = pd.read_csv(args.train)
    val_df   = pd.read_csv(args.val)
    test_df  = pd.read_csv(args.test)
    
    y_train, w_train = train_df["pic50"].values, get_weights(train_df)
    y_val,   w_val   = val_df["pic50"].values,   get_weights(val_df)
    y_test,  w_test  = test_df["pic50"].values,  get_weights(test_df)

    X_train, scaler = build_features(train_df, fit_scaler=True)
    X_val,   _      = build_features(val_df,   desc_scaler=scaler)
    X_test,  _      = build_features(test_df,  desc_scaler=scaler)

    stack = QsarStack()
    stack.fit(X_train, y_train, w_train)

    # Metrics
    def report(y_true, y_pred, split):
        r2 = r2_score(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        print(f"  [{split:<5s}] R² {r2:.4f} | RMSE {rmse:.4f} | MAE {mae:.4f}")
        return {"r2": r2, "rmse": rmse, "mae": mae}

    print("\nResults:")
    res_train = report(y_train, stack.predict(X_train), "Train")
    res_val   = report(y_val,   stack.predict(X_val),   "Val")
    res_test  = report(y_test,  stack.predict(X_test),  "Test")

    # Zone-wise breakdown
    print("\nPer-Zone Test R²:")
    zones = test_df["zone"].values if "zone" in test_df.columns else ["bulk"]*len(test_df)
    preds = stack.predict(X_test)
    for z in ["low", "bulk", "high"]:
        mask = (zones == z)
        if mask.any():
            r2_z = r2_score(y_test[mask], preds[mask])
            print(f"  {z:<5s}: {r2_z:.4f}")

    # Save
    with open(os.path.join(args.output_dir, "stack_model.pkl"), "wb") as f:
        pickle.dump(stack, f)
    with open(os.path.join(args.output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(args.output_dir, "run_summary.json"), "w") as f:
        json.dump({"train": res_train, "val": res_val, "test": res_test, 
                   "meta_weights": stack.meta.coef_.tolist() if hasattr(stack.meta, 'coef_') else []}, f, indent=2)
    
    print(f"\nOutputs saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
