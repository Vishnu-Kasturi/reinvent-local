"""
2_train_xgboost.py  —  Step 2 of the JAK2 (or any target) pipeline
====================================================================
Reads processed.csv from datasets/<TARGET>/ (output of 1_prepare_data.py),
trains an XGBoost pIC50 regressor, and saves the model + scaler to data/.

Feature set:
  - 11 physicochemical descriptors (MW, LogP, HBD, HBA, TPSA, RotBonds,
    Rings, AromaticRings, HeavyAtoms, FractionCSP3, StereoCount)
  - 167 MACCS keys
  - 2048-bit ECFP4 (Morgan r=2)
  - 2048-bit ECFP6 (Morgan r=3)
  - ~200 RDKit 2D descriptors (scaled)

Split strategy:
  Zone-stratified scaffold split (70/15/15 train/val/test).
  Scaffold groups are assigned to a zone by their median pIC50, then
  entire scaffold groups are moved together — no molecule from the same
  scaffold appears in more than one split (zero leakage).

  Zones:
    low  (pIC50 < 6.0)  — weight 4.0 (rare, important boundary)
    bulk (6.0–9.0)      — weight 1.0
    high (pIC50 > 9.0)  — weight 3.0 (rare, high-value hits)

Outputs (saved to data/):
  xgb_model.ubj       — XGBoost model (binary format for fast loading)
  desc_scaler.pkl      — StandardScaler for RDKit 2D descriptors
  run_summary.json     — metrics + feature dim + best iteration

Usage:
  python preprocess/2_train_xgboost.py JAK2
  python preprocess/2_train_xgboost.py EGFR --low_thresh 5.5 --high_thresh 9.5
"""

import os
import sys
import json
import math
import pickle
import argparse
import warnings
from collections import defaultdict

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

# ─── Config ───────────────────────────────────────────────────────────────────
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
# test = 1 - TRAIN_FRAC - VAL_FRAC = 0.15
RANDOM_SEED = 42

PHYSCHEM_COLS = [
    "mw", "logp", "hbd", "hba", "tpsa",
    "rot_bonds", "rings", "arom_rings",
    "heavy_atoms", "frac_csp3", "stereo",
]

ZONE_WEIGHTS = {"low": 4.0, "bulk": 1.0, "high": 3.0}

XGB_PARAMS = dict(
    n_estimators          = 15000,
    learning_rate         = 0.005,
    max_depth             = 5,
    subsample             = 0.7,
    colsample_bytree      = 0.6,
    colsample_bylevel     = 0.7,
    min_child_weight      = 5,
    gamma                 = 0.2,
    reg_alpha             = 0.3,
    reg_lambda            = 2.0,
    objective             = "reg:squarederror",
    eval_metric           = "rmse",
    early_stopping_rounds = 100,
    tree_method           = "hist",
    device                = "cpu",
    random_state          = RANDOM_SEED,
    n_jobs                = -1,
)

# RDKit 2D descriptors — exclude known unstable ones
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
    print(f"  Building features for {len(df):,} molecules...")

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
    print(f"  Feature matrix: {X.shape}  "
          f"(physchem=11, MACCS=167, ECFP4=2048, ECFP6=2048, RDKit2D={len(RDKIT_DESC_FUNCS)})")
    return X, desc_scaler


# ─── Scaffold utilities ───────────────────────────────────────────────────────
def get_scaffold(smi: str) -> str:
    try:
        mol  = Chem.MolFromSmiles(smi)
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core, canonical=True)
    except Exception:
        return "__no_scaffold__"


def zone_label(p: float, low_thresh: float, high_thresh: float) -> str:
    if p < low_thresh:
        return "low"
    elif p > high_thresh:
        return "high"
    return "bulk"


def scaffold_stratified_split(df: pd.DataFrame, low_thresh: float,
                               high_thresh: float) -> tuple:
    """
    Zone-stratified scaffold split with zero leakage.
    Scaffold groups are zoned by their MEDIAN pIC50, then entire
    scaffold groups are assigned to one split.
    """
    np.random.seed(RANDOM_SEED)
    print("\n[*] Computing Murcko scaffolds...")
    df = df.copy()
    df["scaffold"] = df["smiles"].apply(get_scaffold)
    n_scaffolds    = df["scaffold"].nunique()
    print(f"    Unique scaffolds: {n_scaffolds:,}")

    # Zone each scaffold by its median pIC50
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

        print(f"    [{zone:4s}] scaffolds — train:{len(train_scafs):,}  "
              f"val:{len(val_scafs):,}  test:{len(test_scafs):,}")

    train_df = df.loc[train_idx].copy()
    val_df   = df.loc[val_idx].copy()
    test_df  = df.loc[test_idx].copy()

    print(f"\n  Molecule counts:")
    print(f"    Train : {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"    Val   : {len(val_df):,}   ({len(val_df)/len(df)*100:.1f}%)")
    print(f"    Test  : {len(test_df):,}  ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def get_weights(df: pd.DataFrame) -> np.ndarray:
    if "zone" not in df.columns:
        return np.ones(len(df))
    return np.array([ZONE_WEIGHTS.get(z, 1.0) for z in df["zone"]])


def metrics(y_true, y_pred, split=""):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  [{split:<5s}] R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return {"r2": r2, "rmse": rmse, "mae": mae}


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=str,
                        help="Target name matching datasets/<TARGET>/processed.csv")
    parser.add_argument("--datasets_dir", type=str, default="datasets")
    parser.add_argument("--out_dir",      type=str, default="data",
                        help="Output directory for model + scaler (default: data/)")
    parser.add_argument("--low_thresh",   type=float, default=6.0,
                        help="pIC50 below this = 'low' zone (default: 6.0)")
    parser.add_argument("--high_thresh",  type=float, default=9.0,
                        help="pIC50 above this = 'high' zone (default: 9.0)")
    args = parser.parse_args()

    target_name  = args.target.strip().upper()
    processed_csv = os.path.join(args.datasets_dir, target_name, "processed.csv")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  XGBoost pIC50 Trainer")
    print(f"  Target:     {target_name}")
    print(f"  Input:      {processed_csv}")
    print(f"  Output:     {args.out_dir}/")
    print(f"{'='*60}\n")

    if not os.path.exists(processed_csv):
        print(f"[ERROR] processed.csv not found: {processed_csv}")
        print(f"  Run: python preprocess/1_prepare_data.py {target_name} first")
        sys.exit(1)

    # ── Load ─────────────────────────────────────────────────────────────────
    print("[1/5] Loading data...")
    df = pd.read_csv(processed_csv)
    print(f"  Loaded {len(df):,} molecules")

    # Ensure required columns exist — accept both 'pic50' and 'pIC50'
    if "smiles" not in df.columns:
        print("[ERROR] processed.csv must have a 'smiles' column")
        sys.exit(1)
    if "pic50" not in df.columns and "pIC50" in df.columns:
        df = df.rename(columns={"pIC50": "pic50"})
    if "pic50" not in df.columns:
        print("[ERROR] processed.csv must have a 'pic50' or 'pIC50' column")
        sys.exit(1)

    # Drop any remaining NaNs
    df = df.dropna(subset=["smiles", "pic50"]).reset_index(drop=True)
    df["pic50"] = df["pic50"].astype(float)

    # Add zone labels
    df["zone"] = df["pic50"].apply(
        lambda p: zone_label(p, args.low_thresh, args.high_thresh))

    zone_counts = df["zone"].value_counts()
    for z in ["low", "bulk", "high"]:
        print(f"  Zone [{z:4s}]: {zone_counts.get(z, 0):,}")

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\n[2/5] Zone-stratified scaffold split...")
    train_df, val_df, test_df = scaffold_stratified_split(
        df, args.low_thresh, args.high_thresh)

    # ── Features ──────────────────────────────────────────────────────────────
    print("\n[3/5] Building features...")
    X_train, desc_scaler = build_features(train_df, fit_scaler=True)
    X_val,   _           = build_features(val_df,   desc_scaler=desc_scaler)
    X_test,  _           = build_features(test_df,  desc_scaler=desc_scaler)

    y_train = train_df["pic50"].values
    y_val   = val_df["pic50"].values
    y_test  = test_df["pic50"].values

    w_train = get_weights(train_df)

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    print("\n[4/5] Training XGBoost...")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        verbose=500,
    )
    print(f"\n  Best iteration: {model.best_iteration}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating...")
    res = {
        "train": metrics(y_train, model.predict(X_train), "Train"),
        "val":   metrics(y_val,   model.predict(X_val),   "Val  "),
        "test":  metrics(y_test,  model.predict(X_test),  "Test "),
    }

    # Per-zone test R²
    print("\n  Per-zone Test R²:")
    test_preds = model.predict(X_test)
    for z in ["low", "bulk", "high"]:
        mask = (test_df["zone"].values == z)
        if mask.any():
            r2_z = r2_score(y_test[mask], test_preds[mask])
            print(f"    [{z:4s}]: {r2_z:.4f}  (n={mask.sum()})")

    # ── Save predictions ──────────────────────────────────────────────────────
    pred_df = test_df[["smiles", "pic50", "zone"]].copy()
    pred_df["pred_pic50"] = test_preds
    pred_df["residual"]   = pred_df["pic50"] - pred_df["pred_pic50"]
    pred_df.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    # ── Save model + scaler ───────────────────────────────────────────────────
    model_path  = os.path.join(args.out_dir, "xgb_model.ubj")
    scaler_path = os.path.join(args.out_dir, "desc_scaler.pkl")

    model.save_model(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(desc_scaler, f)

    summary = {
        "target":          target_name,
        "best_iteration":  model.best_iteration,
        "feature_dim":     int(X_train.shape[1]),
        "physchem_cols":   PHYSCHEM_COLS,
        "results":         res,
    }
    with open(os.path.join(args.out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
    print(f"\n  R² (Val):  {res['val']['r2']:.4f}")
    print(f"  R² (Test): {res['test']['r2']:.4f}")
    print(f"\n  Ready for RL! The TOML configs point to:")
    print(f"    params.model_path  = \"../../data/xgb_model.ubj\"")
    print(f"    params.scaler_path = \"../../data/desc_scaler.pkl\"")


if __name__ == "__main__":
    main()
