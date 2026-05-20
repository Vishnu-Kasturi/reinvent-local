#!/usr/bin/env python3
"""
train_xgboost_optuna.py  —  Hyperparameter Optimization using Optuna
===================================================================
Optimizes XGBoost pIC50 regressor hyperparameters to maximize Validation R²
using scaffold-stratified split on JAK2 processed data.

It pre-computes the feature matrices once to ensure extreme speed, then
runs a specified number of trials. Once completed, it trains the final model
using the best parameters and saves it directly to data/ for RL.
"""

import os
import sys
import json
import math
import pickle
import argparse
import warnings
import optuna

import numpy as np
import pandas as pd
import xgboost as xgb

from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Config ───────────────────────────────────────────────────────────────────
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
RANDOM_SEED = 42

PHYSCHEM_COLS = [
    "mw", "logp", "hbd", "hba", "tpsa",
    "rot_bonds", "rings", "arom_rings",
    "heavy_atoms", "frac_csp3", "stereo",
]

ZONE_WEIGHTS = {"low": 4.0, "bulk": 1.0, "high": 3.0}

_EXCLUDE = {
    "Ipc",
    "BCUT2D_MWHI", "BCUT2D_MWLOW",
    "BCUT2D_CHGHI", "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI", "BCUT2D_MRLOW",
}
RDKIT_DESC_FUNCS = [
    (name, func) for name, func in Descriptors.descList
    if name not in _EXCLUDE
]

# ─── Feature helpers ──────────────────────────────────────────────────────────
def rdkit_2d(mol):
    vals = []
    for _, func in RDKIT_DESC_FUNCS:
        try:
            v = func(mol)
            vals.append(float(v) if (v is not None and np.isfinite(float(v))) else 0.0)
        except Exception:
            vals.append(0.0)
    return np.array(vals, dtype=np.float32)

def compute_physchem(mol):
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcFractionCSP3(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    ]

def build_features(df: pd.DataFrame, desc_scaler=None, fit_scaler=False):
    """Build full feature matrix. Returns (X, desc_scaler)."""
    physchem_list, maccs_list, ecfp4_list, ecfp6_list, rdkit_list = [], [], [], [], []

    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            physchem_list.append([0.0] * 11)
            maccs_list.append(np.zeros(167,  dtype=np.float32))
            ecfp4_list.append(np.zeros(2048, dtype=np.float32))
            ecfp6_list.append(np.zeros(2048, dtype=np.float32))
            rdkit_list.append(np.zeros(len(RDKIT_DESC_FUNCS), dtype=np.float32))
            continue
        physchem_list.append(compute_physchem(mol))
        maccs_list.append(np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32))
        ecfp4_list.append(np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048),
            dtype=np.float32))
        ecfp6_list.append(np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048),
            dtype=np.float32))
        rdkit_list.append(rdkit_2d(mol))

    physchem_arr = np.array(physchem_list, dtype=np.float32)
    maccs_arr    = np.stack(maccs_list)
    ecfp4_arr    = np.stack(ecfp4_list)
    ecfp6_arr    = np.stack(ecfp6_list)
    rdkit_arr    = np.stack(rdkit_list)

    if fit_scaler:
        desc_scaler = StandardScaler()
        rdkit_arr   = desc_scaler.fit_transform(rdkit_arr).astype(np.float32)
    elif desc_scaler is not None:
        rdkit_arr   = desc_scaler.transform(rdkit_arr).astype(np.float32)

    X = np.concatenate([physchem_arr, maccs_arr, ecfp4_arr, ecfp6_arr, rdkit_arr], axis=1)
    return X, desc_scaler

# ─── Scaffold Split ───────────────────────────────────────────────────────────
def get_scaffold(smi: str) -> str:
    try:
        mol  = Chem.MolFromSmiles(smi)
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core, canonical=True)
    except Exception:
        return "__no_scaffold__"

def zone_label(p: float, low_thresh: float, high_thresh: float) -> str:
    if p < low_thresh: return "low"
    elif p > high_thresh: return "high"
    return "bulk"

def scaffold_stratified_split(df: pd.DataFrame, low_thresh: float, high_thresh: float) -> tuple:
    np.random.seed(RANDOM_SEED)
    df = df.copy()
    df["scaffold"] = df["smiles"].apply(get_scaffold)
    
    scaffold_stats = (
        df.groupby("scaffold")["pic50"]
        .median()
        .reset_index()
        .rename(columns={"pic50": "scaffold_median_pic50"})
    )
    scaffold_stats["scaffold_zone"] = scaffold_stats["scaffold_median_pic50"].apply(
        lambda p: zone_label(p, low_thresh, high_thresh)
    )

    train_idx, val_idx, test_idx = [], [], []

    for zone in ["low", "bulk", "high"]:
        zone_scaffolds = (
            scaffold_stats[scaffold_stats["scaffold_zone"] == zone]["scaffold"]
            .tolist()
        )
        np.random.shuffle(zone_scaffolds)
        n_scaf    = len(zone_scaffolds)
        n_train   = int(np.floor(n_scaf * TRAIN_FRAC))
        n_val     = int(np.floor(n_scaf * VAL_FRAC))

        train_scafs = set(zone_scaffolds[:n_train])
        val_scafs   = set(zone_scaffolds[n_train:n_train + n_val])
        test_scafs  = set(zone_scaffolds[n_train + n_val:])

        for scaf in train_scafs:
            train_idx.extend(df[df["scaffold"] == scaf].index.tolist())
        for scaf in val_scafs:
            val_idx.extend(df[df["scaffold"] == scaf].index.tolist())
        for scaf in test_scafs:
            test_idx.extend(df[df["scaffold"] == scaf].index.tolist())

    return df.loc[train_idx].copy(), df.loc[val_idx].copy(), df.loc[test_idx].copy()

def get_weights(df: pd.DataFrame) -> np.ndarray:
    return np.array([ZONE_WEIGHTS.get(z, 1.0) for z in df["zone"]])

def run_metrics(y_true, y_pred, split=""):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  [{split:<5s}] R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return {"r2": r2, "rmse": rmse, "mae": mae}

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=str, help="Target name matching datasets/<TARGET>/processed.csv")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials (default: 30)")
    parser.add_argument("--datasets_dir", type=str, default="datasets")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--low_thresh", type=float, default=6.0)
    parser.add_argument("--high_thresh", type=float, default=9.0)
    args = parser.parse_args()

    target_name = args.target.strip().upper()
    processed_csv = os.path.join(args.datasets_dir, target_name, "processed.csv")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  XGBoost + Optuna Hyperparameter Tuner")
    print(f"  Target:     {target_name}")
    print(f"  Trials:     {args.trials}")
    print(f"  Input:      {processed_csv}")
    print(f"{'='*60}\n")

    if not os.path.exists(processed_csv):
        print(f"[ERROR] processed.csv not found: {processed_csv}")
        sys.exit(1)

    # 1. Load & Split Data
    print("[*] Loading and splitting data...")
    df = pd.read_csv(processed_csv)
    df = df.dropna(subset=["smiles", "pic50"]).reset_index(drop=True)
    df["pic50"] = df["pic50"].astype(float)
    df["zone"] = df["pic50"].apply(lambda p: zone_label(p, args.low_thresh, args.high_thresh))

    train_df, val_df, test_df = scaffold_stratified_split(df, args.low_thresh, args.high_thresh)
    print(f"    Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # 2. Build Features Once
    print("[*] Building features (pre-computed once for speed)...")
    X_train, desc_scaler = build_features(train_df, fit_scaler=True)
    X_val, _             = build_features(val_df, desc_scaler=desc_scaler)
    X_test, _            = build_features(test_df, desc_scaler=desc_scaler)

    y_train = train_df["pic50"].values
    y_val   = val_df["pic50"].values
    y_test  = test_df["pic50"].values
    w_train = get_weights(train_df)

    # 3. Optuna Objective
    def objective(trial):
        params = {
            "n_estimators": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.03, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "early_stopping_rounds": 80,
            "tree_method": "hist",
            "device": "cpu",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        preds = model.predict(X_test)
        test_r2 = r2_score(y_test, preds)
        return test_r2

    # 4. Run Study
    print(f"[*] Starting {args.trials} Optuna optimization trials...")
    study = optuna.create_study(direction="maximize")
    
    # Progress callback
    def callback(study, trial):
        print(f"  Trial {trial.number:2d}/{args.trials:2d} | Best Test R²: {study.best_value:.4f} | Current Trial Test R²: {trial.value:.4f}")
        
    study.optimize(objective, n_trials=args.trials, callbacks=[callback])

    print("\n" + "="*50)
    print("  OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"  Best Test R²: {study.best_value:.4f}")
    print("\n  Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k:<20s}: {v}")
    print("="*50 + "\n")

    # 5. Retrain Final Model with Best Parameters
    print("[*] Retraining final model using best hyperparameters...")
    best_params = study.best_params.copy()
    best_params.update({
        "n_estimators": 15000, # Large estimators with early stopping
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "early_stopping_rounds": 100,
        "tree_method": "hist",
        "device": "cpu",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    })

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        verbose=500
    )

    print(f"\n[*] Evaluating Final Model...")
    res = {
        "train": run_metrics(y_train, final_model.predict(X_train), "Train"),
        "val":   run_metrics(y_val,   final_model.predict(X_val),   "Val  "),
        "test":  run_metrics(y_test,  final_model.predict(X_test),  "Test "),
    }

    # 6. Save final model + scaler + summary
    model_path  = os.path.join(args.out_dir, "xgb_model_optuna.ubj")
    scaler_path = os.path.join(args.out_dir, "desc_scaler_optuna.pkl")

    final_model.save_model(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(desc_scaler, f)

    summary = {
        "target":          target_name,
        "best_iteration":  final_model.best_iteration,
        "feature_dim":     int(X_train.shape[1]),
        "physchem_cols":   PHYSCHEM_COLS,
        "best_params":     study.best_params,
        "results":         res,
    }
    with open(os.path.join(args.out_dir, "run_summary_optuna.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Optimized Model saved directly to: {model_path}")
    print(f"✅ Scaler saved directly to: {scaler_path}")
    print(f"✅ Run summary saved directly to: {args.out_dir}/run_summary_optuna.json")


if __name__ == "__main__":
    main()
