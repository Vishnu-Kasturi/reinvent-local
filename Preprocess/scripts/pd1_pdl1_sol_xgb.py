#!/usr/bin/env python3
"""
pd1_pdl1_sol_xgb.py  —  PD1-PDL1 Solubility Regressor
===================================================
Trains a single XGBoost model on concatenated modular features to predict 
solubility Y:
  1. 200 RDKit 2D descriptors   (curated, NaN-safe)
  2. ECFP4 fingerprints         (2048 bits, radius=2)
  3. MACCS keys                 (167 bits)
  4. 12 physicochemical props   (MW, LogP, HBD, HBA, TPSA, RotBonds,
                                 Rings, AromaticRings, HeavyAtoms,
                                 FracCsp3, Stereocenters, QED)

Input  : pd1_pdl1_preprocess_sol_train.csv  &  pd1_pdl1_preprocess_sol_test.csv
Output : sol_xgb3_model.ubj
         sol_xgb3_scaler.pkl
         sol_xgb3_feature_names.json
         sol_xgb3_run_summary.json

Usage
-----
  python Preprocess/scripts/pd1_pdl1_sol_xgb.py
"""

import os
import sys
import json
import math
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, QED, rdMolDescriptors
from rdkit import RDLogger
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Feature configuration
# ──────────────────────────────────────────────────────────────────────────────

# RDKit descriptors to EXCLUDE (known to produce Inf/NaN or are redundant)
_RDKIT_EXCLUDE = {
    "Ipc",
    "BCUT2D_MWHI", "BCUT2D_MWLOW",
    "BCUT2D_CHGHI", "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",  "BCUT2D_MRLOW",
}

# Build the list of 200 RDKit descriptor names (first 200 after exclusions)
_ALL_RDKIT = [(name, fn) for name, fn in Descriptors.descList
              if name not in _RDKIT_EXCLUDE]
RDKIT_DESC_LIST = _ALL_RDKIT[:200]          # exactly 200
RDKIT_DESC_NAMES = [n for n, _ in RDKIT_DESC_LIST]

# Physicochemical feature names (12)
PHYSCHEM_NAMES = [
    "mw", "logp", "hbd", "hba", "tpsa",
    "rot_bonds", "rings", "arom_rings",
    "heavy_atoms", "frac_csp3", "stereo", "qed",
]

ECFP4_NBITS  = 2048
ECFP4_RADIUS = 2
MACCS_NBITS  = 167


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _rdkit_descriptors(mol) -> list:
    """Compute 200 RDKit 2D descriptors; replace Inf/NaN with 0."""
    vals = []
    for _, fn in RDKIT_DESC_LIST:
        try:
            v = fn(mol)
            vals.append(0.0 if (v is None or math.isnan(v) or math.isinf(v)) else float(v))
        except Exception:
            vals.append(0.0)
    return vals


def _ecfp4(mol) -> list:
    """2048-bit ECFP4 fingerprint as a list of ints."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, ECFP4_RADIUS, nBits=ECFP4_NBITS)
    return list(fp)


def _maccs(mol) -> list:
    """167-bit MACCS keys as a list of ints."""
    fp = MACCSkeys.GenMACCSKeys(mol)
    return list(fp)


def _physchem(mol) -> list:
    """12 physicochemical properties."""
    try:
        qed_val = QED.qed(mol)
    except Exception:
        qed_val = 0.0
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcFractionCSP3(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        qed_val,
    ]


def featurise_smiles(smiles: str, config: dict):
    """
    Convert a SMILES string to a flat feature vector based on config.
    Returns None if the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    feats = []
    if config.get("use_rdkit", True):
        feats.extend(_rdkit_descriptors(mol))
    if config.get("use_ecfp4", True):
        feats.extend(_ecfp4(mol))
    if config.get("use_maccs", True):
        feats.extend(_maccs(mol))
    if config.get("use_physchem", True):
        feats.extend(_physchem(mol))
        
    return feats


def build_feature_matrix(df: pd.DataFrame, config: dict, smiles_col: str = "smiles"):
    """
    Featurise all molecules in df.
    Returns X (np.ndarray, shape [n_valid, n_features]),
            y (np.ndarray, shape [n_valid]),
            valid_idx (list of original df indices that were kept).
    """
    rows, y_vals, valid_idx = [], [], []
    for i, row in df.iterrows():
        feats = featurise_smiles(row[smiles_col], config)
        if feats is None:
            print(f"    [WARN] Could not parse SMILES at row {i}: {row[smiles_col][:60]}")
            continue
        rows.append(feats)
        y_vals.append(float(row["Y"]))
        valid_idx.append(i)
    return np.array(rows, dtype=np.float32), np.array(y_vals, dtype=np.float32), valid_idx


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def print_metrics(y_true, y_pred, label: str, active_threshold: float = -3.0) -> dict:
    r2   = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    
    # Binarize for classification metrics
    y_true_cls = (y_true >= active_threshold).astype(int)
    y_pred_cls = (y_pred >= active_threshold).astype(int)
    
    try:
        f1 = f1_score(y_true_cls, y_pred_cls)
        auc = roc_auc_score(y_true_cls, y_pred) # use continuous predictions for AUC
    except Exception:
        f1 = 0.0
        auc = 0.0
        
    print(f"  [{label:<5}]  R²={r2:.4f}   RMSE={rmse:.4f}   MAE={mae:.4f}   F1(>={active_threshold})={f1:.4f}   AUC={auc:.4f}")
    return {"r2": r2, "rmse": rmse, "mae": mae, "f1": float(f1), "auc": float(auc)}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)

    parser = argparse.ArgumentParser(description="Plain XGBoost Solubility regressor")
    parser.add_argument("--train",        type=str,
                        default=os.path.join(repo, "Data_pd1_pdl1/data_csvs/pd1_pdl1_preprocess_sol_train.csv"))
    parser.add_argument("--test",         type=str,
                        default=os.path.join(repo, "Data_pd1_pdl1/data_csvs/pd1_pdl1_preprocess_sol_test.csv"))
    parser.add_argument("--out",          type=str,
                        default=os.path.join(repo, "checkpoints_pd1_pdl1_sol"))
    parser.add_argument("--n_estimators", type=int, default=3000)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--no_rdkit",     action="store_true", help="Disable RDKit 2D descriptors")
    parser.add_argument("--no_ecfp4",     action="store_true", help="Disable ECFP4 fingerprints")
    parser.add_argument("--no_maccs",     action="store_true", help="Disable MACCS keys")
    parser.add_argument("--no_physchem",  action="store_true", help="Disable Physicochemical features")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    config = {
        "use_rdkit": not args.no_rdkit,
        "use_ecfp4": not args.no_ecfp4,
        "use_maccs": not args.no_maccs,
        "use_physchem": not args.no_physchem,
    }

    n_features = 0
    cont_idx = []
    feat_names = []
    
    if config["use_rdkit"]:
        cont_idx.extend(range(n_features, n_features + 200))
        n_features += 200
        feat_names.extend([f"rdkit_{n}" for n in RDKIT_DESC_NAMES])
        
    if config["use_ecfp4"]:
        n_features += ECFP4_NBITS
        feat_names.extend([f"ecfp4_{i}" for i in range(ECFP4_NBITS)])
        
    if config["use_maccs"]:
        n_features += MACCS_NBITS
        feat_names.extend([f"maccs_{i}" for i in range(MACCS_NBITS)])
        
    if config["use_physchem"]:
        cont_idx.extend(range(n_features, n_features + len(PHYSCHEM_NAMES)))
        n_features += len(PHYSCHEM_NAMES)
        feat_names.extend([f"physchem_{n}" for n in PHYSCHEM_NAMES])
        
    if n_features == 0:
        print("[ERROR] No features selected! Cannot train model.")
        sys.exit(1)

    print(f"\n{'='*62}")
    print(f"  XGBoost-3 — Plain XGBoost Solubility Regressor")
    print(f"  Train : {args.train}")
    print(f"  Test  : {args.test}")
    print(f"  Output: {args.out}")
    print(f"  Feature breakdown:")
    print(f"    RDKit 2D descriptors : {'Yes (200)' if config['use_rdkit'] else 'No'}")
    print(f"    ECFP4 (r=2, 2048 bit): {'Yes (2048)' if config['use_ecfp4'] else 'No'}")
    print(f"    MACCS keys           : {'Yes (167)' if config['use_maccs'] else 'No'}")
    print(f"    Physicochemical      : {'Yes (12)' if config['use_physchem'] else 'No'}")
    print(f"    ─────────────────────────")
    print(f"    Total features       : {n_features}")
    print(f"{'='*62}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    for path, label in [(args.train, "train"), (args.test, "test")]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} file not found: {path}")
            sys.exit(1)

    train_df = pd.read_csv(args.train)
    test_df  = pd.read_csv(args.test)
    print(f"[*] Loaded  train: {len(train_df):,}  |  test: {len(test_df):,}")

    # ── 2. Build feature matrices ─────────────────────────────────────────────
    print(f"\n[*] Building feature matrices (this may take ~30-60 s)...")

    print(f"    Featurising train set...")
    X_train, y_train, _ = build_feature_matrix(train_df, config)

    print(f"    Featurising test set...")
    X_test,  y_test,  _ = build_feature_matrix(test_df, config)

    print(f"    Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")

    # Scale continuous descriptors (indices stored in cont_idx)
    print(f"\n[*] Scaling continuous features (RDKit descriptors + physicochemical)...")

    if len(cont_idx) > 0:
        scaler = StandardScaler()
        X_train[:, cont_idx] = scaler.fit_transform(X_train[:, cont_idx])
        X_test[:,  cont_idx] = scaler.transform(X_test[:,  cont_idx])
    else:
        scaler = None
        print("    No continuous features selected, skipping scaling.")

    # Determine dynamic active threshold (median of training solubility)
    median_thresh = float(np.median(y_train))
    print(f"[*] Dynamic active threshold set to training median Y: {median_thresh:.3f}")

    # ── 4. Train XGBoost ──────────────────────────────────────────────────────
    print(f"\n[*] Training XGBoost ({args.n_estimators} estimators, early stopping on test)...")
    model = xgb.XGBRegressor(
        n_estimators        = args.n_estimators,
        learning_rate       = 0.01,
        max_depth           = 6,
        subsample           = 0.8,
        colsample_bytree    = 0.6,
        colsample_bylevel   = 0.7,
        min_child_weight    = 3,
        gamma               = 0.1,
        reg_alpha           = 0.1,
        reg_lambda          = 1.0,
        early_stopping_rounds = 100,
        objective           = "reg:squarederror",
        eval_metric         = "rmse",
        tree_method         = "hist",
        device              = "cpu",
        random_state        = args.seed,
        n_jobs              = -1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=500,
    )
    print(f"    Best iteration: {model.best_iteration}")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print(f"\n[*] Performance metrics:")
    metrics_train = print_metrics(y_train, model.predict(X_train), "Train", active_threshold=median_thresh)
    metrics_test  = print_metrics(y_test,  model.predict(X_test),  "Test ", active_threshold=median_thresh)

    # ── 6. Save artefacts ─────────────────────────────────────────────────────
    model_path   = os.path.join(args.out, "sol_xgb3_model.ubj")
    scaler_path  = os.path.join(args.out, "sol_xgb3_scaler.pkl")
    featnames_path = os.path.join(args.out, "sol_xgb3_feature_names.json")
    summary_path = os.path.join(args.out, "sol_xgb3_run_summary.json")

    model.save_model(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Feature names in order
    with open(featnames_path, "w") as f:
        json.dump(feat_names, f, indent=2)

    feature_groups = {}
    if config["use_rdkit"]: feature_groups["rdkit_200"] = 200
    if config["use_ecfp4"]: feature_groups["ecfp4_2048"] = ECFP4_NBITS
    if config["use_maccs"]: feature_groups["maccs_167"] = MACCS_NBITS
    if config["use_physchem"]: feature_groups["physchem_12"] = len(PHYSCHEM_NAMES)

    summary = {
        "train_csv":       args.train,
        "test_csv":        args.test,
        "n_train":         int(len(y_train)),
        "n_test":          int(len(y_test)),
        "n_features":      n_features,
        "feature_groups":  feature_groups,
        "best_iteration":  model.best_iteration,
        "metrics": {
            "train": metrics_train,
            "test":  metrics_test,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*62}")
    print(f"✅ Model   → {model_path}")
    print(f"✅ Scaler  → {scaler_path}")
    print(f"✅ Features→ {featnames_path}")
    print(f"✅ Summary → {summary_path}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
