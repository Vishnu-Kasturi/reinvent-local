#!/usr/bin/env python3
"""
Evaluate Transfer Learning Checkpoints
======================================
Automates sampling and Tanimoto similarity audits for all 10 intermediate
Transfer Learning checkpoints of JAK2.

For each checkpoint:
1. Generates a sampling TOML config.
2. Runs REINVENT4 sampling (500 molecules).
3. Evaluates SMILES validity, uniqueness, exact copies (similarity = 1.0),
   and mean max Tanimoto similarity against the training set.
4. Prints a beautiful summary table, saves metrics to CSV, and plots trends.
"""

import os
import glob
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

REPO_ROOT = "/Users/vishnukasturi/Intern/reinvent-local"
TRAIN_SMILES_PATH = os.path.join(REPO_ROOT, "datasets", "JAK2", "clean_smiles.smi")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# 1. Load training set fingerprints
print("[*] Loading training set from ChEMBL...")
if not os.path.exists(TRAIN_SMILES_PATH):
    raise FileNotFoundError(f"Training set SMILES file not found: {TRAIN_SMILES_PATH}")

train_smiles = []
with open(TRAIN_SMILES_PATH, "r") as f:
    for line in f:
        smi = line.strip()
        if smi:
            train_smiles.append(smi)

train_fps = []
for smi in train_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        train_fps.append(fp)

print(f"    Loaded {len(train_fps)} valid training fingerprints.")

# 2. Identify all 10 checkpoints
checkpoints = []
for epoch in range(10, 110, 10):
    chkpt_path = os.path.join(MODELS_DIR, f"jak2_focused.model.{epoch}.chkpt")
    if os.path.exists(chkpt_path):
        checkpoints.append((epoch, chkpt_path))
    else:
        # Fallback to focused.model for epoch 100 if final chkpt has standard name
        if epoch == 100:
            final_path = os.path.join(MODELS_DIR, "jak2_focused.model")
            if os.path.exists(final_path):
                checkpoints.append((epoch, final_path))

if not checkpoints:
    print("[ERROR] No intermediate checkpoint files found in models/ folder!")
    print("Please make sure you have: models/jak2_focused.model.10.chkpt, etc.")
    exit(1)

# Sort checkpoints by epoch
checkpoints = sorted(checkpoints, key=lambda x: x[0])
print(f"[*] Found {len(checkpoints)} checkpoints to evaluate.")

# 3. Process each checkpoint
results = []

for epoch, chkpt_path in checkpoints:
    print(f"\n========================================\nEvaluating Epoch {epoch} Checkpoint...")
    
    # Define temporary files
    temp_toml_path = os.path.join(REPO_ROOT, "REINVENT4", "configs", f"temp_eval_sampling_{epoch}.toml")
    temp_csv_path = os.path.join(RESULTS_DIR, f"temp_sampling_{epoch}.csv")
    
    # Write sampling TOML
    toml_content = f"""run_type = "sampling"
device = "cpu"
json_out_config = "temp_eval_json_{epoch}.json"

[parameters]
model_file = "{chkpt_path}"
num_smiles = 500
output_file = "{temp_csv_path}"

unique_molecules = true
sample_strategy = "multinomial"
temperature = 1.0
"""
    with open(temp_toml_path, "w") as f:
        f.write(toml_content)
        
    # Execute REINVENT4 sampling
    print(f"  [Reinvent] Sampling 500 SMILES from epoch {epoch} model...")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}/REINVENT4:{env.get('PYTHONPATH', '')}"
    
    try:
        subprocess.run(
            ["reinvent", "-l", f"../logs/temp_eval_{epoch}.log", f"configs/temp_eval_sampling_{epoch}.toml"],
            cwd=os.path.join(REPO_ROOT, "REINVENT4"),
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"  [ERROR] Reinvent sampling failed for epoch {epoch}: {e}")
        # Clean up config
        if os.path.exists(temp_toml_path): os.remove(temp_toml_path)
        continue

    # Load sampled candidates
    if not os.path.exists(temp_csv_path):
        print(f"  [ERROR] Output file not generated for epoch {epoch}!")
        if os.path.exists(temp_toml_path): os.remove(temp_toml_path)
        continue
        
    df_samp = pd.read_csv(temp_csv_path)
    sampled_smiles = df_samp["SMILES"].tolist()
    
    # Compute metrics
    total_sampled = len(sampled_smiles)
    valid_mols = []
    valid_smiles = []
    
    for s in sampled_smiles:
        mol = Chem.MolFromSmiles(str(s))
        if mol is not None:
            valid_mols.append(mol)
            valid_smiles.append(Chem.MolToSmiles(mol))
            
    n_valid = len(valid_mols)
    validity = (n_valid / total_sampled) * 100 if total_sampled > 0 else 0.0
    
    # Uniqueness
    unique_smi = set(valid_smiles)
    n_unique = len(unique_smi)
    uniqueness = (n_unique / n_valid) * 100 if n_valid > 0 else 0.0
    
    # Tanimoto similarities
    max_similarities = []
    exact_copies = 0
    novel_compounds = 0 # Similarity < 0.4
    
    for mol in valid_mols:
        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, train_fps)
        max_sim = max(sims)
        max_similarities.append(max_sim)
        
        if max_sim >= 0.999: # Direct copy
            exact_copies += 1
        if max_sim < 0.40:
            novel_compounds += 1
            
    mean_max_tani = np.mean(max_similarities) if max_similarities else 0.0
    exact_pct = (exact_copies / n_valid) * 100 if n_valid > 0 else 0.0
    novel_pct = (novel_compounds / n_valid) * 100 if n_valid > 0 else 0.0
    
    # Save checkpoint stats
    metrics = {
        "Epoch": epoch,
        "Validity (%)": validity,
        "Uniqueness (%)": uniqueness,
        "Mean Max Tanimoto": mean_max_tani,
        "Exact Copies (%)": exact_pct,
        "Novel (<0.4) (%)": novel_pct
    }
    results.append(metrics)
    
    print(f"  [Stats] Validity: {validity:.1f}% | Uniqueness: {uniqueness:.1f}%")
    print(f"  [Stats] Mean Max Similarity: {mean_max_tani:.3f} | Exact Copies: {exact_pct:.1f}% | Novel: {novel_pct:.1f}%")
    
    # Clean up temporary run files
    if os.path.exists(temp_toml_path): os.remove(temp_toml_path)
    if os.path.exists(temp_csv_path): os.remove(temp_csv_path)
    # Clean up intermediate json configs
    for json_file in glob.glob(os.path.join(REPO_ROOT, "REINVENT4", f"*_{epoch}.json")):
        os.remove(json_file)

# 4. Print beautiful summary
df_res = pd.DataFrame(results)
summary_csv_path = os.path.join(RESULTS_DIR, "tl_checkpoints_evaluation.csv")
df_res.to_csv(summary_csv_path, index=False)

print("\n\n" + "="*80)
print("             TRANSFER LEARNING EVALUATION SUMMARY")
print("="*80)
print("Epoch | Validity (%) | Uniqueness (%) | Mean Max Tanimoto | Exact Copies (%) | Novel (<0.4) (%)")
print("------|--------------|----------------|-------------------|------------------|-----------------")
for idx, row in df_res.iterrows():
    print(f"{int(row['Epoch']):5d} | {row['Validity (%)']:12.1f} | {row['Uniqueness (%)']:14.1f} | {row['Mean Max Tanimoto']:17.3f} | {row['Exact Copies (%)']:16.1f} | {row['Novel (<0.4) (%)']:15.1f}")
print("="*80)
print(f"Detailed CSV saved → results/tl_checkpoints_evaluation.csv")

# 5. Plot trends and save image
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Prior Model Evolution Over Transfer Learning Epochs", fontsize=16, fontweight='bold')

# Plot 1: Validity & Uniqueness
axes[0, 0].plot(df_res["Epoch"], df_res["Validity (%)"], marker='o', color='forestgreen', label="Validity (%)", linewidth=2)
axes[0, 0].plot(df_res["Epoch"], df_res["Uniqueness (%)"], marker='s', color='darkorange', label="Uniqueness (%)", linewidth=2)
axes[0, 0].set_title("SMILES Grammar & Diversity")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Percentage (%)")
axes[0, 0].grid(True, linestyle='--', alpha=0.6)
axes[0, 0].legend()

# Plot 2: Mean Max Tanimoto Similarity
axes[0, 1].plot(df_res["Epoch"], df_res["Mean Max Tanimoto"], marker='^', color='dodgerblue', linewidth=2)
axes[0, 1].set_title("Average Peak Tanimoto vs Training Set")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Mean Max Similarity")
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# Plot 3: Exact Copies (%)
axes[1, 0].plot(df_res["Epoch"], df_res["Exact Copies (%)"], marker='d', color='crimson', linewidth=2)
axes[1, 0].set_title("Exact Copies (Similarity = 1.0)")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Percentage (%)")
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

# Plot 4: Novel Molecules (<0.4 Similarity) (%)
axes[1, 1].plot(df_res["Epoch"], df_res["Novel (<0.4) (%)"], marker='v', color='purple', linewidth=2)
axes[1, 1].set_title("Novel Lead Compounds (< 0.4 Tanimoto)")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Percentage (%)")
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plot_png_path = os.path.join(RESULTS_DIR, "tl_checkpoints_evaluation.png")
plt.savefig(plot_png_path, dpi=300)
plt.close()
print(f"Trends plot saved   → results/tl_checkpoints_evaluation.png")
