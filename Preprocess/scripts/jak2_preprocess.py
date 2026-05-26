#!/usr/bin/env python3
"""
jak2_preprocess.py
===================
Preprocesses raw JAK2 bioactivity data from Preprocess/Data_jak2/jak2raw.csv:
1. Loads semicolon-separated ChEMBL raw data.
2. Filters for IC50 values and standardizes units to nM.
3. Computes pIC50 = 9 - log10(Standard Value).
4. Desalts and cleans SMILES using RDKit (strips salts and extracts largest organic fragment).
5. Deduplicates compounds by taking the median pIC50 per unique canonical SMILES.
6. Computes Bemis-Murcko scaffolds.
7. Performs scaffold-stratified split (80/20) with activity stratification to prevent leakage.
8. Saves outputs in Preprocess/Data_jak2/ as:
   - processed_train.csv
   - processed_test.csv
   - processed_all.csv
"""

import os
import sys
import math
import warnings
import argparse
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Scaffolds import MurckoScaffold

# Disable verbose RDKit warnings
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Chemistry & Desalting helpers
# ──────────────────────────────────────────────────────────────────────────────

def desalt_and_standardise(smiles: str, remover: SaltRemover) -> str | None:
    """
    Desalts a SMILES string by:
    1. Parsing it to an RDKit Mol object.
    2. Stripping standard salts using SaltRemover.
    3. Extracting the largest organic fragment if the molecule is still a mixture.
    4. Returning the clean canonical SMILES.
    """
    if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
        return None
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # 1. Strip standard salts
        mol = remover.StripMol(mol)
        if mol is None:
            return None
            
        # Get canonical SMILES after stripping
        smi = Chem.MolToSmiles(mol, canonical=True)
        
        # 2. Extract largest fragment if still disconnected (mixture/salt residues)
        if "." in smi:
            frags = smi.split(".")
            frag_mols = []
            for f in frags:
                m = Chem.MolFromSmiles(f)
                if m is not None:
                    frag_mols.append((f, m.GetNumHeavyAtoms()))
            if not frag_mols:
                return None
            # Sort by heavy atom count (largest first)
            frag_mols.sort(key=lambda x: x[1], reverse=True)
            smi = frag_mols[0][0]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
                
        # Return final canonical clean SMILES
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
    parser = argparse.ArgumentParser(description="Preprocess raw JAK2 ChEMBL dataset")
    parser.add_argument("--raw", type=str, default="Preprocess/Data_jak2/jak2raw.csv",
                        help="Path to raw ChEMBL CSV file")
    parser.add_argument("--out_dir", type=str, default="Preprocess/Data_jak2",
                        help="Directory to save preprocessed outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()
    
    print("=" * 65)
    print("  🚀 JAK2 Bioactivity Preprocessing & Desalting Pipeline 🚀")
    print("=" * 65)
    
    if not os.path.exists(args.raw):
        print(f"[ERROR] Raw CSV file not found at: {args.raw}")
        sys.exit(1)
        
    os.makedirs(args.out_dir, exist_ok=True)
    
    # ── 1. Load ChEMBL Data ───────────────────────────────────────────────────
    print("[*] Loading raw data...")
    try:
        # ChEMBL exports are typically semicolon-delimited
        df_raw = pd.read_csv(args.raw, sep=";")
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        sys.exit(1)
        
    print(f"    Loaded {len(df_raw):,} records from ChEMBL.")
    
    # Normalize column names for robust parsing
    df_raw.columns = [col.strip().lower() for col in df_raw.columns]
    
    # Check required columns
    required_cols = ["smiles", "standard type", "standard value", "standard units"]
    for rc in required_cols:
        if rc not in df_raw.columns:
            print(f"[ERROR] Missing required column: '{rc}'. Columns found: {list(df_raw.columns)}")
            sys.exit(1)
            
    # ── 2. Filter and Clean bioactivity values ────────────────────────────────
    print("[*] Performing QSAR-grade bioactivity data cleaning...")
    n_start = len(df_raw)
    
    # 1. Require valid smiles
    df = df_raw.dropna(subset=["smiles"]).copy()
    
    # 2. Extract direct pChEMBL value if available (standardized negative log activity)
    if "pchembl value" in df.columns:
        pchem = pd.to_numeric(df["pchembl value"], errors="coerce")
    else:
        pchem = pd.Series([np.nan] * len(df), index=df.index)
        
    # 3. Calculate from standard value if pChEMBL is not available
    # We require standard relation to be '=' (implicit or explicit) and units to be nM
    std_val = pd.to_numeric(df["standard value"], errors="coerce")
    
    # Clean standard relation to remove quotes or whitespace
    rel = df["standard relation"].fillna("=").astype(str).str.strip().str.replace("'", "").str.replace('"', "")
    units = df["standard units"].fillna("").astype(str).str.strip().str.lower()
    
    calc = np.where(
        (rel == "=") & (units == "nm") & (std_val > 0.0),
        9.0 - np.log10(std_val),
        np.nan
    )
    
    # Merge direct pChEMBL and calculated pIC50
    df["pic50"] = pchem.fillna(pd.Series(calc, index=df.index))
    
    # 4. Filter for IC50 standard type (which is standard for QSAR modeling)
    df = df[df["standard type"].str.upper() == "IC50"]
    
    # Drop rows without a valid pIC50
    df = df.dropna(subset=["pic50"])
    
    # Sensible pIC50 range kept (e.g. 2.0 to 14.0)
    df = df[(df["pic50"] >= 2.0) & (df["pic50"] <= 14.0)]
    
    print(f"    Raw records: {n_start:,}")
    print(f"    Cleaned records with exact, high-confidence pIC50: {len(df):,} (dropped {n_start - len(df):,})")
    
    # ── 4. Desalt & Strip Salts using RDKit ───────────────────────────────────
    print("[*] Desalting and standardizing SMILES using RDKit...")
    remover = SaltRemover()  # default RDKit salt dictionary
    
    cleaned_smiles = []
    n_desalted = 0
    for smi in df["smiles"]:
        clean = desalt_and_standardise(smi, remover)
        cleaned_smiles.append(clean)
        if clean is not None and clean != smi:
            n_desalted += 1
            
    df["canonical_smiles"] = cleaned_smiles
    df = df.dropna(subset=["canonical_smiles"])
    print(f"    Successfully desalted and standardized: {len(df):,} molecules.")
    print(f"    Number of molecules actively stripped of salts: {n_desalted:,}")
    
    # ── 5. Deduplicate by Canonical SMILES ────────────────────────────────────
    print("[*] Deduplicating duplicate compounds (using median pIC50)...")
    n_before = len(df)
    
    # Group by canonical SMILES and average standard values/median pIC50
    df_dedup = (
        df.groupby("canonical_smiles")["pic50"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"canonical_smiles": "smiles", "median": "pic50", "count": "n_measurements"})
    )
    
    print(f"    Deduplication complete: {n_before:,} -> {len(df_dedup):,} unique compounds")
    print(f"    Molecules with multiple measurements: {(df_dedup['n_measurements'] > 1).sum():,}")
    
    # ── 6. Compute Bemis-Murcko Scaffolds ─────────────────────────────────────
    print("[*] Extracting Bemis-Murcko scaffolds...")
    df_dedup["scaffold"] = df_dedup["smiles"].apply(get_scaffold)
    n_scaf = df_dedup["scaffold"].nunique()
    print(f"    Total unique scaffolds: {n_scaf:,} (Singleton scaffolds: {(df_dedup.groupby('scaffold').size() == 1).sum():,})")
    
    # ── 7. Scaffold-Stratified 80/20 Split ────────────────────────────────────
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
    
    # Scaffold overlap check
    overlap = set(train_df["scaffold"]) & set(test_df["scaffold"])
    print(f"  Scaffold Overlap      : {len(overlap):,} (should be 0 for zero leakage!)")
    print("=" * 50 + "\n")
    
    # ── 8. Save Outputs ───────────────────────────────────────────────────────
    out_all = os.path.join(args.out_dir, "data_csvs/jak2_preprocess_all.csv")
    out_train = os.path.join(args.out_dir, "data_csvs/jak2_preprocess_train.csv")
    out_test = os.path.join(args.out_dir, "data_csvs/jak2_preprocess_test.csv")
    
    # Ensure parent output directory exists
    os.makedirs(os.path.dirname(out_all), exist_ok=True)
    
    df_dedup[["smiles", "pic50", "scaffold", "n_measurements"]].to_csv(out_all, index=False)
    train_df[["smiles", "pic50", "scaffold", "n_measurements"]].to_csv(out_train, index=False)
    test_df[["smiles", "pic50", "scaffold", "n_measurements"]].to_csv(out_test, index=False)
    
    print(f"✅ Preprocessing pipeline complete!")
    print(f"✅ Saved clean combined data  → {out_all}")
    print(f"✅ Saved stratified Train set → {out_train}")
    print(f"✅ Saved stratified Test set  → {out_test}\n")


if __name__ == "__main__":
    main()
