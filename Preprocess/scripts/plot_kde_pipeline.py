#!/usr/bin/env python3
"""
plot_kde_pipeline.py
====================
Parametric KDE plotter for JAK2 and PD1-PDL1 RL pipelines.
"""
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import RDConfig
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def compute_properties(smiles_list):
    mw, qed, sa = [], [], []
    for smi in tqdm(smiles_list, desc="Computing Props"):
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol:
            mw.append(Descriptors.MolWt(mol))
            qed.append(QED.qed(mol))
            try:
                sa.append(sascorer.calculateScore(mol))
            except:
                sa.append(np.nan)
        else:
            mw.append(np.nan)
            qed.append(np.nan)
            sa.append(np.nan)
    return mw, qed, sa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--raw_pic50", required=True)
    parser.add_argument("--raw_sol", required=False, default=None)
    args = parser.parse_args()
    
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(repo_root, "results")
    
    # 1. Load baseline
    print(f"[*] Loading baseline {args.raw_pic50}")
    df_base = pd.read_csv(args.raw_pic50)
    
    # standardize col names
    if "pic50" not in df_base.columns and "pIC50" in df_base.columns:
        df_base.rename(columns={"pIC50": "pic50"}, inplace=True)
        
    df_base = df_base.dropna(subset=["smiles", "pic50"])
    
    mw, qed, sa = compute_properties(df_base["smiles"])
    df_base["MW"] = mw
    df_base["QED"] = qed
    df_base["SA"] = sa
    df_base["Source"] = "Original Data"
    df_base["Solubility"] = np.nan
    
    # Load solubility pool separately to avoid canonicalization matching issues
    df_sol_clean = pd.DataFrame()
    if args.raw_sol and os.path.exists(args.raw_sol):
        df_sol = pd.read_csv(args.raw_sol)
        df_sol.rename(columns={"Y": "Solubility"}, inplace=True, errors="ignore")
        df_sol_clean = pd.DataFrame({
            "Solubility": df_sol["Solubility"].dropna(),
            "Source": "Original Data"
        })
    
    # 2. Load RL Output
    rl_files = glob.glob(os.path.join(results_dir, f"{args.target}_rl_toml_*.csv"))
    if not rl_files:
        print(f"[!] No RL outputs found for {args.target}")
        return
        
    latest_rl = max(rl_files, key=os.path.getmtime)
    print(f"[*] Loading RL {latest_rl}")
    df_rl = pd.read_csv(latest_rl)
    
    # Filter to optimized steps (last 20% of training steps) to represent the optimized model
    if "step" in df_rl.columns:
        max_step = df_rl["step"].max()
        cutoff = int(max_step * 0.8)
        print(f"[*] Filtering RL output to optimized steps (> {cutoff}) | {len(df_rl)} -> ", end="")
        df_rl = df_rl[df_rl["step"] > cutoff]
        print(f"{len(df_rl)} molecules")
        
    df_rl.rename(columns={"SMILES": "smiles"}, inplace=True)
    
    # Map RL columns
    pic50_col_raw = f"{args.target.upper().replace('_','')}pIC50_raw (raw)"
    pic50_col = f"{args.target.upper().replace('_','')}pIC50_raw"
    
    if pic50_col_raw in df_rl.columns:
        df_rl["pic50"] = df_rl[pic50_col_raw]
    elif pic50_col in df_rl.columns:
        df_rl["pic50"] = df_rl[pic50_col]
    elif "JAK2pIC50_raw (raw)" in df_rl.columns:
        df_rl["pic50"] = df_rl["JAK2pIC50_raw (raw)"]
         
    sol_col_raw = f"{args.target.upper().replace('_','')}Sol_raw (raw)"
    sol_col = f"{args.target.upper().replace('_','')}Sol_raw"
    
    if sol_col_raw in df_rl.columns:
        df_rl["Solubility"] = df_rl[sol_col_raw]
    elif sol_col in df_rl.columns:
        df_rl["Solubility"] = df_rl[sol_col]
    else:
        df_rl["Solubility"] = np.nan
        
    mw, qed, sa = compute_properties(df_rl["smiles"])
    df_rl["MW"] = mw
    df_rl["QED"] = qed
    df_rl["SA"] = sa
    df_rl["Source"] = "RL Generated"
    
    # Combine
    df_all = pd.concat([df_base, df_rl, df_sol_clean], ignore_index=True)
    
    # Plot
    metrics = ["pic50", "MW", "QED", "SA"]
    if df_all["Solubility"].notna().any():
        metrics.append("Solubility")
        
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    
    for i, m in enumerate(metrics):
        sns.kdeplot(data=df_all, x=m, hue="Source", common_norm=False, 
                    fill=True, alpha=0.3, ax=axes[i])
        axes[i].set_title(f"{m} Distribution")
        
        if m == "MW":
            axes[i].axvline(500, color='r', linestyle='--')
        elif m == "QED":
            axes[i].axvline(0.6, color='r', linestyle='--')
            
    plt.tight_layout()
    out_png = os.path.join(results_dir, f"{args.target}_kde_comparison.png")
    plt.savefig(out_png)
    print(f"[*] Saved KDE plots to {out_png}")

if __name__ == "__main__":
    main()
