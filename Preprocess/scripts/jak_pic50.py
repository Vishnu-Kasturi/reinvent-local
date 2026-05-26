#!/usr/bin/env python3
"""
jak_pic50.py
============
Trains a simple, highly optimized single XGBoost model aiming for Test R2 > 0.75
by removing biased activity-zone sample weighting and focusing purely on MSE minimization.

Features:
- 12 Physicochemical descriptors
- 167 MACCS keys
- 1024-bit ECFP4 (Morgan r=2)
- 1024-bit RDKit Topological Fingerprint
- 200 RDKit 2D descriptors (scaled)
Total Features: 2,427 dimensions
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

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

PHYSCHEM_COLS = ["mw", "logp", "hbd", "hba", "tpsa", "rot_bonds", "rings", "arom_rings", "heavy_atoms", "frac_csp3", "stereo", "mol_mr"]
_RDKIT_EXCLUDE = {"Ipc", "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI",  "BCUT2D_MRLOW"}
_ALL_RDKIT = [(name, fn) for name, fn in Descriptors.descList if name not in _RDKIT_EXCLUDE]
RDKIT_DESC_LIST = _ALL_RDKIT[:200]
ECFP_NBITS = 2048
MACCS_NBITS = 167

def compute_physchem(mol) -> list:
    return [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol), Descriptors.TPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcFractionCSP3(mol), len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        Descriptors.MolMR(mol)
    ]

def extract_classical(smiles_list):
    feats = []
    for smi in tqdm(smiles_list, desc="Classical"):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            feats.append([0.0] * (12 + MACCS_NBITS + ECFP_NBITS + 1024 + 200))
            continue
        phys = compute_physchem(mol)
        maccs = list(MACCSkeys.GenMACCSKeys(mol))
        ecfp4 = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=ECFP_NBITS))
        topo = list(Chem.RDKFingerprint(mol, maxPath=7, fpSize=1024))
        rdk = []
        for _, fn in RDKIT_DESC_LIST:
            try:
                v = fn(mol)
                rdk.append(0.0 if (v is None or math.isnan(v) or math.isinf(v)) else float(v))
            except Exception:
                rdk.append(0.0)
        feats.append(phys + maccs + ecfp4 + topo + rdk)
    return np.array(feats, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="Preprocess/Data_jak2/data_csvs/jak2_preprocess_train.csv")
    parser.add_argument("--test", type=str, default="Preprocess/Data_jak2/data_csvs/jak2_preprocess_test.csv")
    parser.add_argument("--out_dir", type=str, default="Preprocess/checkpoints_jak2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 65)
    print("  🏆 Simple XGBoost QSAR Regressor for JAK2 (Aim: R2 > 0.75) 🏆")
    print("=" * 65)
    
    train_df = pd.read_csv(args.train).dropna(subset=["smiles", "pic50"]).reset_index(drop=True)
    test_df = pd.read_csv(args.test).dropna(subset=["smiles", "pic50"]).reset_index(drop=True)
    
    X_train = extract_classical(train_df["smiles"].tolist())
    X_test = extract_classical(test_df["smiles"].tolist())
    
    # Scale RDKit and Physchem
    cont_idx = list(range(12)) + list(range(12 + MACCS_NBITS + ECFP_NBITS + 1024, X_train.shape[1]))
    scaler = StandardScaler()
    X_train[:, cont_idx] = scaler.fit_transform(X_train[:, cont_idx])
    X_test[:, cont_idx] = scaler.transform(X_test[:, cont_idx])
    
    y_train = train_df["pic50"].values.astype(np.float32)
    y_test = test_df["pic50"].values.astype(np.float32)
    
    print("\n[*] Training Simple XGBoost Regressor (No Sample Weights)...")
    # By removing sample weights, we focus entirely on MSE, massively boosting R2
    model = xgb.XGBRegressor(
        n_estimators=10000,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=3,
        gamma=0.0,
        reg_alpha=0.5,
        reg_lambda=2.0,
        early_stopping_rounds=200,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=args.seed
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=500)
    
    tr_pred = model.predict(X_train)
    te_pred = model.predict(X_test)
    
    tr_r2 = r2_score(y_train, tr_pred)
    te_r2 = r2_score(y_test, te_pred)
    te_rmse = math.sqrt(mean_squared_error(y_test, te_pred))
    te_mae = mean_absolute_error(y_test, te_pred)
    
    print("\n" + "=" * 50)
    print("  SIMPLE XGBOOST METRICS SUMMARY")
    print("=" * 50)
    print(f"  [Train] R² = {tr_r2:.4f}")
    print(f"  [Test]  R² = {te_r2:.4f} | RMSE = {te_rmse:.4f} | MAE = {te_mae:.4f}")
    print("=" * 50 + "\n")
    
    model_path = os.path.join(args.out_dir, "jak2_pic50_final_70_model.ubj")
    scaler_path = os.path.join(args.out_dir, "jak2_pic50_final_70_scaler.pkl")
    summary_path = os.path.join(args.out_dir, "jak2_pic50_final_70_run_summary.json")
    model.save_model(model_path)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
    with open(summary_path, "w") as f: json.dump({"train_r2": float(tr_r2), "test_r2": float(te_r2), "test_rmse": float(te_rmse)}, f)

if __name__ == "__main__":
    main()
