#!/usr/bin/env python3
"""
cross_eval_pic50.py
===================
Cross-evaluates the JAK2 final_70 XGBoost model on the PD1-PDL1 pIC50 test set,
and the PD1-PDL1 pIC50 XGBoost model on the JAK2 test set.

No data preprocessing changes are made. Both test sets are featurized using
their respective model's feature extractor (same feature pipeline used at
training time) and evaluated directly.

Usage
-----
  python Preprocess/scripts/cross_eval_pic50.py
"""

import os
import math
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, QED, rdMolDescriptors
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ──────────────────────────────────────────────────────────────────────
JAK2_TEST_CSV   = os.path.join(ROOT, "Data_jak2/data_csvs/jak2_preprocess_test.csv")
PD1_TEST_CSV    = os.path.join(ROOT, "Data_pd1_pdl1/data_csvs/pd1_pdl1_preprocess_test.csv")

JAK2_MODEL      = os.path.join(ROOT, "checkpoints_jak2/jak2_pic50_final_70_model.ubj")
JAK2_SCALER     = os.path.join(ROOT, "checkpoints_jak2/jak2_pic50_final_70_scaler.pkl")

PD1_MODEL       = os.path.join(ROOT, "checkpoints_pd1_pdl1_pic50/pic50_xgb2_model.ubj")
PD1_SCALER      = os.path.join(ROOT, "checkpoints_pd1_pdl1_pic50/pic50_xgb2_scaler.pkl")
PD1_FEATNAMES   = os.path.join(ROOT, "checkpoints_pd1_pdl1_pic50/pic50_xgb2_feature_names.json")

# ── JAK2 Feature Extractor ────────────────────────────────────────────────────
# Feature order: 12 physchem | 167 MACCS | 1024 ECFP4 | 1024 Topo | 200 RDKit
# Matches jak_pic50.py extract_classical() exactly.

_RDKIT_EXCLUDE = {"Ipc", "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI",
                  "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW",
                  "BCUT2D_MRHI", "BCUT2D_MRLOW"}
_ALL_RDKIT = [(n, fn) for n, fn in Descriptors.descList if n not in _RDKIT_EXCLUDE]
RDKIT_DESC_LIST = _ALL_RDKIT[:200]


def jak2_physchem(mol):
    return [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        Descriptors.TPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcFractionCSP3(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        Descriptors.MolMR(mol),
    ]


def featurize_jak2(smiles_list):
    """Exactly replicates jak_pic50.py extract_classical().
    Feature layout: 12 physchem | 167 MACCS | 2048 ECFP4 | 1024 Topo | 200 RDKit = 3451
    """
    feats = []
    DIM = 12 + 167 + 2048 + 1024 + 200  # = 3451, matches jak2_pic50_final_70_model.ubj
    for smi in tqdm(smiles_list, desc="  JAK2 feats"):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            feats.append([0.0] * DIM)
            continue
        phys  = jak2_physchem(mol)
        maccs = list(MACCSkeys.GenMACCSKeys(mol))
        ecfp4 = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        topo  = list(Chem.RDKFingerprint(mol, maxPath=7, fpSize=1024))
        rdk   = []
        for _, fn in RDKIT_DESC_LIST:
            try:
                v = fn(mol)
                rdk.append(0.0 if (v is None or math.isnan(v) or math.isinf(v)) else float(v))
            except Exception:
                rdk.append(0.0)
        feats.append(phys + maccs + ecfp4 + topo + rdk)
    return np.array(feats, dtype=np.float32)


# ── PD1 Feature Extractor ─────────────────────────────────────────────────────
# Feature order: 200 RDKit | 2048 ECFP4 | 167 MACCS | 12 physchem
# Matches pd1_pdl1_pic50_xgb.py (xgboost_2.py) featurise_smiles() exactly.

PHYSCHEM_NAMES = ["mw","logp","hbd","hba","tpsa","rot_bonds","rings",
                  "arom_rings","heavy_atoms","frac_csp3","stereo","qed"]

def pd1_physchem(mol):
    try:
        qed_val = QED.qed(mol)
    except Exception:
        qed_val = 0.0
    return [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcFractionCSP3(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        qed_val,
    ]


def featurize_pd1(smiles_list):
    """Exactly replicates pd1_pdl1_pic50_xgb.py featurise_smiles()."""
    feats = []
    DIM = 200 + 2048 + 167 + 12
    for smi in tqdm(smiles_list, desc="  PD1 feats"):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            feats.append([0.0] * DIM)
            continue
        rdk = []
        for _, fn in RDKIT_DESC_LIST:
            try:
                v = fn(mol)
                rdk.append(0.0 if (v is None or math.isnan(v) or math.isinf(v)) else float(v))
            except Exception:
                rdk.append(0.0)
        ecfp4 = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        maccs = list(MACCSkeys.GenMACCSKeys(mol))
        phys  = pd1_physchem(mol)
        feats.append(rdk + ecfp4 + maccs + phys)
    return np.array(feats, dtype=np.float32)


# ── Metric helpers ────────────────────────────────────────────────────────────

def report(y_true, y_pred, label):
    r2   = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"  {label}")
    print(f"    R²   = {r2:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    print(f"    MAE  = {mae:.4f}")
    return {"r2": r2, "rmse": rmse, "mae": mae}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  🔀 Cross-Evaluation: JAK2 ↔ PD1-PDL1 pIC50 Models")
    print("=" * 62)

    # -- Load CSVs -------------------------------------------------------
    for path, name in [(JAK2_TEST_CSV, "JAK2 test"), (PD1_TEST_CSV, "PD1-PDL1 test"),
                       (JAK2_MODEL, "JAK2 model"), (JAK2_SCALER, "JAK2 scaler"),
                       (PD1_MODEL, "PD1 model"), (PD1_SCALER, "PD1 scaler")]:
        if not os.path.exists(path):
            print(f"[ERROR] Missing: {path}")
            raise FileNotFoundError(path)

    jak2_test = pd.read_csv(JAK2_TEST_CSV).dropna(subset=["smiles","pic50"]).reset_index(drop=True)
    pd1_test  = pd.read_csv(PD1_TEST_CSV).dropna(subset=["smiles","pic50"]).reset_index(drop=True)

    print(f"\n[*] Loaded JAK2 test  : {len(jak2_test):,} molecules")
    print(f"[*] Loaded PD1 test   : {len(pd1_test):,} molecules\n")

    # -- Load models -----------------------------------------------------
    jak2_model = xgb.XGBRegressor()
    jak2_model.load_model(JAK2_MODEL)
    with open(JAK2_SCALER, "rb") as f:
        jak2_scaler = pickle.load(f)

    pd1_model = xgb.XGBRegressor()
    pd1_model.load_model(PD1_MODEL)
    with open(PD1_SCALER, "rb") as f:
        pd1_scaler = pickle.load(f)

    # ── CROSS 1: JAK2 model → PD1-PDL1 test set ─────────────────────────
    print("─" * 62)
    print(" CROSS 1: JAK2 model predicting PD1-PDL1 pIC50 test set")
    print("─" * 62)
    print("[*] Featurizing PD1-PDL1 test set using JAK2 feature pipeline...")
    X_pd1_jak2style = featurize_jak2(pd1_test["smiles"].tolist())
    # Apply JAK2 scaler: cont_idx = 0:12 (physchem) and 12+167+2048+1024: end (rdk 200 cols)
    jak2_cont_idx = list(range(12)) + list(range(12 + 167 + 2048 + 1024, X_pd1_jak2style.shape[1]))
    X_pd1_jak2style[:, jak2_cont_idx] = jak2_scaler.transform(X_pd1_jak2style[:, jak2_cont_idx])
    y_pd1_true = pd1_test["pic50"].values.astype(np.float32)
    y_pd1_jak2_pred = jak2_model.predict(X_pd1_jak2style)
    metrics_c1 = report(y_pd1_true, y_pd1_jak2_pred, "JAK2 model → PD1-PDL1 Test")

    # ── CROSS 2: PD1 model → JAK2 test set ───────────────────────────────
    print()
    print("─" * 62)
    print(" CROSS 2: PD1-PDL1 model predicting JAK2 pIC50 test set")
    print("─" * 62)
    print("[*] Featurizing JAK2 test set using PD1-PDL1 feature pipeline...")
    X_jak2_pd1style = featurize_pd1(jak2_test["smiles"].tolist())
    # Apply PD1 scaler: cont_idx = 0:200 (rdkit) and 200+2048+167: end (physchem)
    pd1_cont_idx = list(range(200)) + list(range(200 + 2048 + 167, X_jak2_pd1style.shape[1]))
    X_jak2_pd1style[:, pd1_cont_idx] = pd1_scaler.transform(X_jak2_pd1style[:, pd1_cont_idx])
    y_jak2_true = jak2_test["pic50"].values.astype(np.float32)
    y_jak2_pd1_pred = pd1_model.predict(X_jak2_pd1style)
    metrics_c2 = report(y_jak2_true, y_jak2_pd1_pred, "PD1-PDL1 model → JAK2 Test")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  CROSS-EVALUATION SUMMARY")
    print("=" * 62)
    print(f"  JAK2 model   → PD1-PDL1 Test  |  R² = {metrics_c1['r2']:.4f}  RMSE = {metrics_c1['rmse']:.4f}")
    print(f"  PD1-PDL1 model → JAK2 Test    |  R² = {metrics_c2['r2']:.4f}  RMSE = {metrics_c2['rmse']:.4f}")
    print("=" * 62)

    # Save summary
    out_path = os.path.join(ROOT, "checkpoints_jak2", "cross_eval_summary.json")
    with open(out_path, "w") as f:
        json.dump({
            "jak2_model_on_pd1_test": metrics_c1,
            "pd1_model_on_jak2_test": metrics_c2,
        }, f, indent=2)
    print(f"\n✅ Cross-eval summary saved → {out_path}\n")


if __name__ == "__main__":
    main()
