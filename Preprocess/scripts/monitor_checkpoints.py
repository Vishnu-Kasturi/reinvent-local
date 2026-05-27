#!/usr/bin/env python3
"""
monitor_checkpoints.py
======================
Actively monitors the models/ directory for new Transfer Learning checkpoints,
automatically samples molecules, calculates similarity/drug-likeness properties,
and plots Tanimoto & KDE comparisons in real-time.

Usage:
  python Preprocess/scripts/monitor_checkpoints.py \
      --target pd1_pdl1 \
      --raw_csv Preprocess/Data_pd1_pdl1/pd1_pdl1_pic50_raw.csv
"""
import os
import sys
import time
import glob
import subprocess
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem, RDConfig
from rdkit.Chem import Descriptors, QED, AllChem, DataStructs
from rdkit import RDLogger

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
try:
    import sascorer
except ImportError:
    sascorer = None

RDLogger.DisableLog('rdApp.*')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_fps(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    return fps

def compute_properties(smiles_list):
    mw, qed, sa = [], [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol:
            mw.append(Descriptors.MolWt(mol))
            qed.append(QED.qed(mol))
            if sascorer:
                try:
                    sa.append(sascorer.calculateScore(mol))
                except Exception:
                    sa.append(np.nan)
            else:
                sa.append(np.nan)
        else:
            mw.append(np.nan)
            qed.append(np.nan)
            sa.append(np.nan)
    return mw, qed, sa

def generate_audit_plots(epoch, target, raw_csv, sampled_csv, results_dir):
    print(f"[*] Generating audit plots for Epoch {epoch}...")
    
    # 1. Load Raw Data
    try:
        df_raw = pd.read_csv(raw_csv, sep="\t")
        if df_raw.shape[1] <= 1:
            df_raw = pd.read_csv(raw_csv, sep=",")
    except Exception:
        df_raw = pd.read_csv(raw_csv)
        
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]
    raw_smiles = df_raw["smiles"].dropna().tolist()
    
    # 2. Load Sampled Data
    df_samp = pd.read_csv(sampled_csv)
    df_samp.columns = [c.strip().lower() for c in df_samp.columns]
    sampled_smiles = df_samp["smiles"].dropna().tolist()
    
    if not sampled_smiles:
        print("[!] No sampled smiles found in output CSV!")
        return
        
    # 3. Calculate Tanimoto similarities
    print("    Computing Morgan Fingerprints...")
    raw_fps = compute_fps(raw_smiles)
    samp_fps = compute_fps(sampled_smiles)
    
    max_similarities = []
    exact_copies = 0
    for fp in samp_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp, raw_fps)
        max_sim = max(sims) if sims else 0.0
        max_similarities.append(max_sim)
        if max_sim >= 0.999:
            exact_copies += 1
            
    mean_sim = np.mean(max_similarities)
    copy_rate = exact_copies / len(samp_fps) if samp_fps else 0.0
    
    # 4. Calculate Properties (MW, QED, SA)
    print("    Calculating physical properties...")
    raw_mw, raw_qed, raw_sa = compute_properties(raw_smiles)
    samp_mw, samp_qed, samp_sa = compute_properties(sampled_smiles)
    
    df_plot_raw = pd.DataFrame({
        "MW": raw_mw,
        "QED": raw_qed,
        "SA": raw_sa,
        "Source": "Original Data"
    })
    df_plot_samp = pd.DataFrame({
        "MW": samp_mw,
        "QED": samp_qed,
        "SA": samp_sa,
        "Source": f"Epoch {epoch} Sampled"
    })
    df_combined = pd.concat([df_plot_raw, df_plot_samp], ignore_index=True)
    
    # 5. Build beautiful 4-panel subplot figure
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    # Panel 0: Tanimoto Histogram
    sns.histplot(max_similarities, bins=30, kde=True, ax=axes[0], color="#0d6efd", alpha=0.7)
    axes[0].axvline(mean_sim, color="red", linestyle="--", label=f"Mean: {mean_sim:.3f}")
    axes[0].set_title(f"Tanimoto Similarity to Train\n(Copy Rate: {copy_rate:.1%})")
    axes[0].set_xlabel("Max Tanimoto Similarity")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    
    # Panel 1: Molecular Weight KDE
    sns.kdeplot(data=df_combined, x="MW", hue="Source", common_norm=False, fill=True, alpha=0.3, ax=axes[1])
    axes[1].axvline(500, color='gray', linestyle=':', label="Lipinski Limit")
    axes[1].set_title("Molecular Weight")
    axes[1].set_xlabel("MW (Da)")
    
    # Panel 2: QED KDE
    sns.kdeplot(data=df_combined, x="QED", hue="Source", common_norm=False, fill=True, alpha=0.3, ax=axes[2])
    axes[2].axvline(0.6, color='gray', linestyle=':', label="QED Target")
    axes[2].set_title("Quantitative Estimate of Drug-likeness")
    axes[2].set_xlabel("QED Score")
    
    # Panel 3: SA Score KDE
    sns.kdeplot(data=df_combined, x="SA", hue="Source", common_norm=False, fill=True, alpha=0.3, ax=axes[3])
    axes[3].axvline(4.0, color='gray', linestyle=':', label="Synthesizable Limit")
    axes[3].set_title("Synthetic Accessibility")
    axes[3].set_xlabel("SA Score (Lower is better)")
    
    plt.suptitle(f"{target.upper()} Transfer Learning Epoch {epoch} Evaluation Plot", y=1.05, weight='bold', fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, f"{target}_TL_e{epoch}_audit.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[+] Saved checkpoint audit plot to: {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="jak2 or pd1_pdl1")
    parser.add_argument("--raw_csv", required=True, help="Original raw csv data file")
    args = parser.parse_args()
    
    models_dir = os.path.join(REPO_ROOT, "models")
    results_dir = os.path.join(REPO_ROOT, "results")
    configs_dir = os.path.join(REPO_ROOT, "REINVENT4", "configs")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)
    
    print("==================================================")
    print(f" LIVE CHECKPOINT MONITOR STARTED FOR TARGET: {args.target}")
    print(f" Watching models/ for {args.target}_TL.model.*.chkpt")
    print("==================================================")
    
    processed_checkpoints = {}  # maps epoch -> last modified mtime
    
    # Run loop
    while True:
        # Check standard epochs 10 to 100
        for epoch in range(10, 110, 10):
            chkpt_pattern = os.path.join(models_dir, f"{args.target}_TL.model.{epoch}.chkpt")
            if os.path.exists(chkpt_pattern):
                current_mtime = os.path.getmtime(chkpt_pattern)
                if epoch not in processed_checkpoints or current_mtime > processed_checkpoints[epoch]:
                    # Ensure the checkpoint is fully written and stable (wait 3 seconds)
                    time.sleep(3)
                    # refresh mtime after wait
                    current_mtime = os.path.getmtime(chkpt_pattern)
                    
                    print(f"\n[!] DETECTED NEW OR UPDATED CHECKPOINT: Epoch {epoch} | Size: {os.path.getsize(chkpt_pattern)/1024/1024:.1f} MB")
                    
                    # Setup custom REINVENT sampling configuration
                    out_csv = os.path.join(results_dir, f"{args.target}_tl_sample_e{epoch}.csv")
                    toml_content = f"""
run_type = "sampling"
device = "cpu"
tb_logdir = "tb_{args.target}_eval_e{epoch}"
json_out_config = "json_{args.target}_eval_e{epoch}.json"
[parameters]
model_file = "{chkpt_pattern}"
output_file = "{out_csv}"
num_smiles = 500
unique_molecules = true
randomize_smiles = true
"""
                    toml_path = os.path.join(configs_dir, f"eval_{args.target}_e{epoch}.toml")
                    with open(toml_path, "w") as f:
                        f.write(toml_content)
                    
                    print(f"[*] Generating 500 sample molecules via REINVENT4 sampling...")
                    try:
                        subprocess.run(
                            ["reinvent", toml_path],
                            cwd=os.path.join(REPO_ROOT, "REINVENT4"),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=True
                        )
                    except Exception as e:
                        print(f"[!] REINVENT4 sampling failed for Epoch {epoch}: {e}")
                        processed_checkpoints[epoch] = current_mtime
                        continue
                        
                    if os.path.exists(out_csv):
                        generate_audit_plots(epoch, args.target, args.raw_csv, out_csv, results_dir)
                    else:
                        print(f"[!] Sampling output CSV not found for Epoch {epoch}")
                        
                    processed_checkpoints[epoch] = current_mtime
                
        # We can stop only if we have processed all checkpoints up to 100 AND they are stable (older than 10 seconds)
        # However, running indefinitely is safer to ensure it keeps watching. Let's poll every 5 seconds.
        time.sleep(5)

if __name__ == "__main__":
    main()
