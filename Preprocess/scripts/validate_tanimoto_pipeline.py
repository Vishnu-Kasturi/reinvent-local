#!/usr/bin/env python3
"""
validate_tanimoto_pipeline.py
=============================
Validates RL output SMILES against original training SMILES using Tanimoto.
"""
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

def compute_fps(smiles_list):
    fps = []
    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            valid.append(smi)
    return fps, valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--raw_csv", required=True)
    args = parser.parse_args()
    
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(repo_root, "results")
    
    # 1. Load train
    df_raw = pd.read_csv(args.raw_csv, sep="\t")
    if "SMILES" not in df_raw.columns and "smiles" not in df_raw.columns:
        df_raw = pd.read_csv(args.raw_csv)
        
    df_raw.columns = [col.strip().lower() for col in df_raw.columns]
    
    if "smiles" not in df_raw.columns:
        print(f"[!] 'smiles' column not found in {args.raw_csv}")
        return
        
    train_smi = df_raw["smiles"].dropna()
    train_fps, _ = compute_fps(train_smi)
    
    # 2. Load RL output
    rl_files = glob.glob(os.path.join(results_dir, f"{args.target}_rl_toml_*.csv"))
    if not rl_files:
        print(f"[!] No RL outputs found for {args.target}")
        return
        
    latest_rl = max(rl_files, key=os.path.getmtime)
    print(f"[*] Validating latest RL output: {latest_rl}")
    df_rl = pd.read_csv(latest_rl)
    rl_fps, _ = compute_fps(df_rl["SMILES"].dropna())
    
    if not rl_fps:
        return
        
    # 3. Compute Tanimoto
    max_tanimotos = []
    for fp in tqdm(rl_fps, desc="Computing Tanimoto"):
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        max_tanimotos.append(max(sims))
        
    mean_sim = np.mean(max_tanimotos)
    med_sim = np.median(max_tanimotos)
    gt_40 = sum(1 for x in max_tanimotos if x >= 0.4) / len(max_tanimotos)
    exact = sum(1 for x in max_tanimotos if x == 1.0)
    
    print("\n" + "="*50)
    print(f" {args.target.upper()} RL TANIMOTO VALIDATION")
    print("="*50)
    print(f"  Mean max Tanimoto : {mean_sim:.3f}")
    print(f"  Median max Tanimoto: {med_sim:.3f}")
    print(f"  % >= 0.4 similarity: {gt_40:.1%}")
    print(f"  Exact copies       : {exact} ({exact/len(max_tanimotos):.1%})")
    
    with open(os.path.join(results_dir, f"{args.target}_rl_tanimoto.json"), "w") as f:
        json.dump({
            "mean": float(mean_sim),
            "median": float(med_sim),
            "gt_0.4": float(gt_40),
            "exact_copies": int(exact)
        }, f, indent=2)
        
    plt.figure(figsize=(8,5))
    plt.hist(max_tanimotos, bins=30, alpha=0.7, color='green')
    plt.axvline(mean_sim, color='red', linestyle='dashed', linewidth=1)
    plt.title(f"{args.target.upper()} RL Generated vs Original Dataset")
    plt.xlabel("Max Tanimoto Similarity")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(results_dir, f"{args.target}_rl_tanimoto.png"))
    plt.close()
    print(f"[*] Saved histogram to {args.target}_rl_tanimoto.png")

if __name__ == "__main__":
    main()
