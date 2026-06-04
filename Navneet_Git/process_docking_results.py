import sys
import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Add paths to sys.path so we can import feature computation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../REINVENT4")))

from rdkit import Chem
from reinvent_plugins.components.pd1_pdl1_features import compute_features

# Paths to models and scalers
PIC50_MODEL_PATH = "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_model.ubj"
PIC50_SCALER_PATH = "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_scaler.pkl"
SOL_MODEL_PATH = "Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj"
SOL_SCALER_PATH = "Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl"

# Correct relative paths from workspace root
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return None

def process_file(filepath):
    print(f"[*] Processing {filepath}...")
    df = pd.read_csv(filepath, sep='\t', header=None)
    df.columns = ['epoch', 'mol_id', 'docking_score', 'smiles']
    
    # 1. Canonicalize SMILES
    print("   Canonicalizing SMILES...")
    df['canonical_smiles'] = df['smiles'].apply(canonicalize)
    df = df.dropna(subset=['canonical_smiles'])
    
    # 2. Sort by docking score (ascending: most negative/best affinity first)
    df['docking_score'] = pd.to_numeric(df['docking_score'], errors='coerce')
    df = df.dropna(subset=['docking_score'])
    df = df.sort_values(by='docking_score', ascending=True)
    
    # 3. Drop duplicates based on canonical smiles, retaining highest dock score (first)
    initial_len = len(df)
    df = df.drop_duplicates(subset=['canonical_smiles'], keep='first')
    print(f"   Removed duplicates: {initial_len} -> {len(df)} molecules")
    
    smiles_list = df['canonical_smiles'].tolist()
    
    # 4. Predict pIC50
    print("   Predicting pIC50...")
    X_pic50, valid_mask = compute_features(smiles_list, PIC50_SCALER_PATH)
    pic50_model = xgb.Booster()
    pic50_model.load_model(PIC50_MODEL_PATH)
    d_pic50 = xgb.DMatrix(X_pic50)
    df['pic50'] = pic50_model.predict(d_pic50)
    
    # 5. Predict Solubility
    print("   Predicting solubility...")
    X_sol, _ = compute_features(smiles_list, SOL_SCALER_PATH)
    sol_model = xgb.Booster()
    sol_model.load_model(SOL_MODEL_PATH)
    d_sol = xgb.DMatrix(X_sol)
    df['solubility'] = sol_model.predict(d_sol)
    
    # Reset index and return
    df = df.reset_index(drop=True)
    return df

def main():
    # Process both files
    df_rl = process_file("Navneet_Git/RL_filtered.csv")
    df_no_tl = process_file("Navneet_Git/RL_without_TL_filtered.csv")
    
    # Save the processed results
    rl_out_path = "Navneet_Git/RL_filtered_processed.csv"
    no_tl_out_path = "Navneet_Git/RL_without_TL_filtered_processed.csv"
    df_rl.to_csv(rl_out_path, index=False)
    df_no_tl.to_csv(no_tl_out_path, index=False)
    print(f"[+] Saved processed RL results to {rl_out_path}")
    print(f"[+] Saved processed RL without TL results to {no_tl_out_path}")
    
    # Plot distributions
    print("[*] Generating distribution plots...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # pIC50 Distribution
    sns.kdeplot(df_rl['pic50'], fill=True, color="#3b82f6", label="RL (with TL)", alpha=0.5, linewidth=2.5, ax=axes[0])
    sns.kdeplot(df_no_tl['pic50'], fill=True, color="#ef4444", label="RL (without TL)", alpha=0.5, linewidth=2.5, ax=axes[0])
    axes[0].set_title("Predicted pIC50 Distribution", fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel("pIC50", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].legend(fontsize=11)
    
    # Solubility Distribution
    sns.kdeplot(df_rl['solubility'], fill=True, color="#10b981", label="RL (with TL)", alpha=0.5, linewidth=2.5, ax=axes[1])
    sns.kdeplot(df_no_tl['solubility'], fill=True, color="#f59e0b", label="RL (without TL)", alpha=0.5, linewidth=2.5, ax=axes[1])
    axes[1].set_title("Predicted Solubility Distribution", fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel("Solubility (logS)", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    plot_path = "Navneet_Git/pic50_sol_distributions.png"
    plt.savefig(plot_path, dpi=300)
    print(f"[+] Saved distribution plots to {plot_path}")

if __name__ == "__main__":
    main()
