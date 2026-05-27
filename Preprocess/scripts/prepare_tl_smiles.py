#!/usr/bin/env python3
"""
prepare_tl_smiles.py
====================
Filters a raw dataset (pIC50 > 6), canonicalizes SMILES, drops duplicates,
and splits into 80/20 train/val .smi files using a Scaffold-Stratified Split
for REINVENT4 Transfer Learning.

Usage:
  python Preprocess/scripts/prepare_tl_smiles.py \
      --input_csv Preprocess/Data_jak2/data_csvs/jak2_preprocess_all.csv \
      --train_out data/jak2_TL_train.smi \
      --val_out data/jak2_TL_val.smi
"""
import argparse
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def get_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        core = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(core, canonical=True)
        return smi if smi else smiles
    except Exception:
        return smiles

def scaffold_split(df: pd.DataFrame, train_frac: float = 0.80, seed: int = 42):
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
    
    def zone_label(pic50):
        if pic50 < low_thresh: return "low"
        elif pic50 >= high_thresh: return "high"
        return "medium"
        
    scaf_stats["zone"] = scaf_stats["scaf_median_pic50"].apply(zone_label)
    
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
    
    return train_df, test_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--val_out", required=True)
    args = parser.parse_args()
    
    print(f"[*] Loading {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv, sep="\t")
        if df.shape[1] <= 1:
            df = pd.read_csv(args.input_csv, sep=",")
    except Exception:
        df = pd.read_csv(args.input_csv)
    
    # Normalize all column names to lowercase and strip whitespace
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Handle possible pic50 field names
    if "pic50" not in df.columns:
        if "pic50_raw" in df.columns:
            df.rename(columns={"pic50_raw": "pic50"}, inplace=True)
        elif "y" in df.columns:
            df.rename(columns={"y": "pic50"}, inplace=True)
        
    df = df.dropna(subset=["smiles", "pic50"])
    
    # Filter pIC50 > 6
    initial_len = len(df)
    df = df[df["pic50"] > 6.0].copy()
    print(f"[*] Filtered pIC50 > 6.0: {initial_len} -> {len(df)} molecules")
    
    # Canonicalize and deduplicate (filter unsupported atoms)
    # REINVENT default prior only supports: C, N, O, F, Cl, Br, S
    ALLOWED_ATOMS = {6, 7, 8, 9, 16, 17, 35}
    
    clean_smiles = []
    for smi in tqdm(df["smiles"], desc="Canonicalizing & Filtering Tokens"):
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            # Check if all atoms are supported
            valid_mol = True
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() not in ALLOWED_ATOMS:
                    valid_mol = False
                    break
            
            if valid_mol:
                clean_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=False))
            else:
                clean_smiles.append(None)
        else:
            clean_smiles.append(None)
            
    df["canonical_smiles"] = clean_smiles
    df = df.dropna(subset=["canonical_smiles"])
    
    df_dedup = (
        df.groupby("canonical_smiles")["pic50"]
        .median()
        .reset_index()
        .rename(columns={"canonical_smiles": "smiles", "pic50": "pic50"})
    )
    
    print(f"[*] Unique valid canonical SMILES: {len(df_dedup)}")
    if len(df_dedup) < 10:
        raise ValueError("Not enough data after filtering!")
        
    print("[*] Extracting Bemis-Murcko scaffolds for split...")
    df_dedup["scaffold"] = df_dedup["smiles"].apply(get_scaffold)
    
    print("[*] Performing scaffold-stratified split...")
    train_df, val_df = scaffold_split(df_dedup, train_frac=0.80, seed=42)
    
    train_smi = train_df["smiles"].tolist()
    val_smi = val_df["smiles"].tolist()
    
    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    
    with open(args.train_out, "w") as f:
        for s in train_smi: f.write(s + "\n")
    with open(args.val_out, "w") as f:
        for s in val_smi: f.write(s + "\n")
        
    print(f"[*] Saved train ({len(train_smi)}) to {args.train_out}")
    print(f"[*] Saved val ({len(val_smi)}) to {args.val_out}")

if __name__ == "__main__":
    main()
