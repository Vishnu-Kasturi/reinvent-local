#!/usr/bin/env python3
"""
pd1_pdl1_preprocess_pic50.py
============================
Preprocesses raw PD1-PDL1 bioactivity data:
1. Loads raw data (SMILES, pic50).
2. Drops NaNs.
3. Cleans SMILES using RDKit (MANDATORY NO DESALTING!).
4. Deduplicates compounds by taking the median pIC50 per unique canonical SMILES.
5. Computes Bemis-Murcko scaffolds.
6. Performs scaffold-stratified split (80/20) with activity stratification to prevent leakage.
7. Saves outputs in Preprocess/Data_pd1_pdl1/data_csvs/ as:
   - pd1_pdl1_preprocess_train.csv
   - pd1_pdl1_preprocess_test.csv
   - pd1_pdl1_preprocess_all.csv
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

# Disable verbose RDKit warnings
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Chemistry helpers
# ──────────────────────────────────────────────────────────────────────────────

def clean_smiles_no_desalt(smiles: str) -> str | None:
    """
    Parses SMILES and returns the canonical version.
    Explicitly bypasses any desalting per the mandatory instructions.
    """
    if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
        return None
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def get_scaffold(smiles: str) -> str:
    """
    Extracts the Bemis-Murcko scaffold.
    Falls back to the molecule's own canonical SMILES if acyclic.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        core = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(core, canonical=True)
        return smi if smi else smiles
    except Exception:
        return smiles

# ──────────────────────────────────────────────────────────────────────────────
# Stratification & Split helpers
# ──────────────────────────────────────────────────────────────────────────────

def zone_label(pic50: float, low_thresh: float, high_thresh: float) -> str:
    if pic50 < low_thresh:
        return "low"
    elif pic50 >= high_thresh:
        return "high"
    return "medium"

def scaffold_stratified_split(df: pd.DataFrame, train_frac: float = 0.80, seed: int = 42):
    """
    Scaffold-stratified split with activity zone balancing to prevent leakage:
    1. Group scaffolds and find their median pIC50.
    2. Bin scaffolds into low/medium/high activity zones.
    3. Split scaffolds 80/20 within each zone to ensure balanced distributions.
    """
    rng = np.random.default_rng(seed)
    
    # Calculate median pIC50 per scaffold
    scaf_stats = (
        df.groupby("scaffold")["pic50"]
        .median()
        .reset_index()
        .rename(columns={"pic50": "scaf_median_pic50"})
    )
    
    # Determine activity zone thresholds dynamically (33rd and 66th percentiles)
    low_thresh = df["pic50"].quantile(0.33)
    high_thresh = df["pic50"].quantile(0.66)
    
    scaf_stats["zone"] = scaf_stats["scaf_median_pic50"].apply(
        lambda p: zone_label(p, low_thresh, high_thresh)
    )
    
    train_scaffolds, test_scaffolds = set(), set()
    
    for zone in ["low", "medium", "high"]:
        zone_scafs = scaf_stats[scaf_stats["zone"] == zone]["scaffold"].tolist()
        if not zone_scafs:
            continue
        rng.shuffle(zone_scafs)
        n_train = int(np.ceil(len(zone_scafs) * train_frac))
        train_scaffolds.update(zone_scafs[:n_train])
        test_scaffolds.update(zone_scafs[n_train:])
        
    train_df = df[df["scaffold"].isin(train_scaffolds)].copy().reset_index(drop=True)
    test_df  = df[df["scaffold"].isin(test_scaffolds)].copy().reset_index(drop=True)
    
    return train_df, test_df, low_thresh, high_thresh

# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline execution
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess PD1-PDL1 Dataset")
    parser.add_argument("--raw", type=str, default="Preprocess/Data_pd1_pdl1/pd1_pdl1_pic50_raw.csv",
                        help="Path to raw CSV file")
    parser.add_argument("--out_dir", type=str, default="Preprocess/Data_pd1_pdl1/data_csvs",
                        help="Directory to save preprocessed outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()
    
    print("=" * 65)
    print("  🚀 PD1-PDL1 Bioactivity Preprocessing (NO DESALTING) 🚀")
    print("=" * 65)
    
    if not os.path.exists(args.raw):
        print(f"[ERROR] Raw CSV file not found at: {args.raw}")
        sys.exit(1)
        
    os.makedirs(args.out_dir, exist_ok=True)
    
    # ── 1. Load Data ──────────────────────────────────────────────────────────
    print("[*] Loading raw data...")
    try:
        df_raw = pd.read_csv(args.raw, sep="\t")
        if "SMILES" not in df_raw.columns:
            # Fallback to comma separation
            df_raw = pd.read_csv(args.raw)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        sys.exit(1)
        
    print(f"    Loaded {len(df_raw):,} records.")
    
    # Normalize column names
    df_raw.columns = [col.strip().lower() for col in df_raw.columns]
    
    if "smiles" not in df_raw.columns or "pic50" not in df_raw.columns:
        print(f"[ERROR] Missing required 'smiles' or 'pic50' columns. Found: {list(df_raw.columns)}")
        sys.exit(1)
            
    # ── 2. Clean and Filter ───────────────────────────────────────────────────
    print("[*] Filtering NaNs and standardizing SMILES (NO DESALTING)...")
    n_start = len(df_raw)
    
    df = df_raw.dropna(subset=["smiles", "pic50"]).copy()
    df["pic50"] = pd.to_numeric(df["pic50"], errors="coerce")
    df = df.dropna(subset=["pic50"])
    
    cleaned_smiles = []
    for smi in df["smiles"]:
        cleaned_smiles.append(clean_smiles_no_desalt(smi))
            
    df["canonical_smiles"] = cleaned_smiles
    df = df.dropna(subset=["canonical_smiles"])
    
    print(f"    Raw records: {n_start:,}")
    print(f"    Cleaned records (Valid canonical SMILES): {len(df):,} (dropped {n_start - len(df):,})")
    
    # ── 3. Deduplicate by Canonical SMILES ────────────────────────────────────
    print("[*] Deduplicating duplicate compounds (using median pIC50)...")
    n_before = len(df)
    
    df_dedup = (
        df.groupby("canonical_smiles")["pic50"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"canonical_smiles": "smiles", "median": "pic50", "count": "n_measurements"})
    )
    
    print(f"    Deduplication complete: {n_before:,} -> {len(df_dedup):,} unique compounds")
    print(f"    Molecules with multiple measurements: {(df_dedup['n_measurements'] > 1).sum():,}")
    
    # ── 4. Compute Bemis-Murcko Scaffolds ─────────────────────────────────────
    print("[*] Extracting Bemis-Murcko scaffolds...")
    df_dedup["scaffold"] = df_dedup["smiles"].apply(get_scaffold)
    n_scaf = df_dedup["scaffold"].nunique()
    print(f"    Total unique scaffolds: {n_scaf:,} (Singleton scaffolds: {(df_dedup.groupby('scaffold').size() == 1).sum():,})")
    
    # ── 5. Scaffold-Stratified 80/20 Split ────────────────────────────────────
    print(f"[*] Splitting dataset 80/20 (Scaffold-Stratified with activity zone balancing)...")
    train_df, test_df, low, high = scaffold_stratified_split(df_dedup, train_frac=0.80, seed=args.seed)
    
    print("\n" + "=" * 50)
    print("  SPLIT STATS SUMMARY")
    print("=" * 50)
    print(f"  Total Unique Compounds: {len(df_dedup):,}")
    print(f"  Train Set Size        : {len(train_df):,} ({len(train_df)/len(df_dedup)*100:.1f}%)")
    print(f"  Test Set Size         : {len(test_df):,} ({len(test_df)/len(df_dedup)*100:.1f}%)")
    print(f"\n  pIC50 Distribution:")
    print(f"    Train -> Mean: {train_df['pic50'].mean():.2f} | Range: [{train_df['pic50'].min():.2f}, {train_df['pic50'].max():.2f}]")
    print(f"    Test  -> Mean: {test_df['pic50'].mean():.2f} | Range: [{test_df['pic50'].min():.2f}, {test_df['pic50'].max():.2f}]")
    
    overlap = set(train_df["scaffold"]) & set(test_df["scaffold"])
    print(f"  Scaffold Overlap      : {len(overlap):,} (should be 0 for zero leakage!)")
    print("=" * 50 + "\n")
    
    # ── 6. Save Outputs ───────────────────────────────────────────────────────
    out_all = os.path.join(args.out_dir, "pd1_pdl1_preprocess_all.csv")
    out_train = os.path.join(args.out_dir, "pd1_pdl1_preprocess_train.csv")
    out_test = os.path.join(args.out_dir, "pd1_pdl1_preprocess_test.csv")
    
    df_dedup[["smiles", "pic50", "scaffold", "n_measurements"]].to_csv(out_all, index=False)
    train_df[["smiles", "pic50", "scaffold", "n_measurements"]].to_csv(out_train, index=False)
    test_df[["smiles", "pic50", "scaffold", "n_measurements"]].to_csv(out_test, index=False)
    
    print(f"✅ Preprocessing pipeline complete!")
    print(f"✅ Saved clean combined data  → {out_all}")
    print(f"✅ Saved stratified Train set → {out_train}")
    print(f"✅ Saved stratified Test set  → {out_test}\n")

if __name__ == "__main__":
    main()
