#!/usr/bin/env python3
"""
evaluate_tl_checkpoints.py
==========================
Automates sampling and Tanimoto similarity audits for all intermediate
Transfer Learning checkpoints of a given target.

Usage:
  python Preprocess/scripts/evaluate_tl_checkpoints.py --target jak2 --raw_csv data/jak2raw.csv
"""
import os
import glob
import subprocess
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_fps(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    return fps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Target name (jak2 or pd1_pdl1)")
    parser.add_argument("--raw_csv", required=True, help="Original unmodified dataset")
    args = parser.parse_args()
    
    models_dir = os.path.join(REPO_ROOT, "models")
    results_dir = os.path.join(REPO_ROOT, "results")
    
    df = pd.read_csv(args.raw_csv, sep="\t")
    if "SMILES" not in df.columns and "smiles" not in df.columns:
        df = pd.read_csv(args.raw_csv)
        
    df.columns = [col.strip().lower() for col in df.columns]
    
    if "smiles" not in df.columns:
        print(f"[!] 'smiles' column not found in {args.raw_csv}")
        return
        
    train_smiles = df["smiles"].dropna().tolist()
        
    train_fps = compute_fps(train_smiles)
    print(f"    Loaded {len(train_fps)} valid training fingerprints.")
    
    checkpoints = []
    base_name = f"{args.target}_TL.model"
    for epoch in range(10, 110, 10):
        chkpt_path = os.path.join(models_dir, f"{base_name}.{epoch}.chkpt")
        if os.path.exists(chkpt_path):
            checkpoints.append((epoch, chkpt_path))
            
    if not checkpoints:
        # Check standard checkpoints
        for f in glob.glob(os.path.join(models_dir, f"{args.target}_focused.model.*.chkpt")):
            try:
                epoch = int(f.split('.')[-2])
                checkpoints.append((epoch, f))
            except: pass
            
    if not checkpoints:
        print(f"[!] No checkpoints found for {args.target}")
        return
        
    checkpoints = sorted(checkpoints, key=lambda x: x[0])
    print(f"[*] Found {len(checkpoints)} checkpoints to evaluate.")
    
    results = []
    
    for epoch, path in checkpoints:
        print(f"\n=======================================================")
        print(f" Evaluating Epoch {epoch}")
        print(f"=======================================================")
        
        out_csv = os.path.join(results_dir, f"{args.target}_tl_sample_e{epoch}.csv")
        toml_content = f"""
run_type = "sampling"
device = "cpu"
tb_logdir = "tb_{args.target}_eval_e{epoch}"
json_out_config = "json_{args.target}_eval_e{epoch}.json"
[parameters]
model_file = "{path}"
output_file = "{out_csv}"
num_smiles = 500
unique_molecules = true
randomize_smiles = true
"""
        toml_path = os.path.join(REPO_ROOT, "REINVENT4", "configs", f"eval_{args.target}_e{epoch}.toml")
        with open(toml_path, "w") as f:
            f.write(toml_content)
            
        print("[*] Generating 500 molecules via REINVENT4...")
        subprocess.run(
            ["conda", "run", "-n", "reinvent-qsar", "reinvent", toml_path],
            cwd=os.path.join(REPO_ROOT, "REINVENT4"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        if not os.path.exists(out_csv):
            print(f" [!] Output CSV not found for Epoch {epoch}")
            continue
            
        samp = pd.read_csv(out_csv)
        valid = len(samp)
        unique = samp["SMILES"].nunique()
        print(f"    Valid SMILES: {valid}/500")
        print(f"    Unique SMILES: {unique}/{valid}")
        
        if valid == 0:
            continue
            
        samp_fps = compute_fps(samp["SMILES"])
        max_tanimotos = []
        exact_copies = 0
        
        for fp in tqdm(samp_fps, desc="    Tanimoto vs Train"):
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            max_sim = max(sims)
            max_tanimotos.append(max_sim)
            if max_sim == 1.0:
                exact_copies += 1
                
        mean_sim = np.mean(max_tanimotos)
        med_sim = np.median(max_tanimotos)
        copy_rate = exact_copies / len(samp_fps)
        
        print(f"    Mean max Tanimoto: {mean_sim:.3f}")
        print(f"    Median max Tanimoto: {med_sim:.3f}")
        print(f"    Exact copies (sim=1.0): {exact_copies} ({copy_rate:.1%})")
        
        results.append({
            "Epoch": epoch,
            "Valid": valid,
            "Unique": unique,
            "Mean_Tanimoto": mean_sim,
            "Median_Tanimoto": med_sim,
            "Exact_Copies": exact_copies,
            "Copy_Rate": copy_rate,
            "Path": path
        })
        
    if not results:
        return
        
    df_res = pd.DataFrame(results)
    print("\n\n" + "="*80)
    print(f" {args.target.upper()} TL CHECKPOINT EVALUATION SUMMARY")
    print("="*80)
    print(df_res.to_string(index=False))
    
    # Auto-select best
    # Criteria: lowest copy rate, highest mean tanimoto between 0.3 and 0.6
    filtered = df_res[(df_res["Mean_Tanimoto"] >= 0.3) & (df_res["Mean_Tanimoto"] <= 0.6)]
    if filtered.empty:
        filtered = df_res
        
    best_idx = filtered.sort_values(by=["Copy_Rate", "Mean_Tanimoto"], ascending=[True, False]).index[0]
    best_epoch = df_res.loc[best_idx, "Epoch"]
    best_path = df_res.loc[best_idx, "Path"]
    
    print("\n[*] BEST CHECKPOINT:")
    print(f"    Epoch: {best_epoch}")
    print(f"    Copy Rate: {df_res.loc[best_idx, 'Copy_Rate']:.1%}")
    print(f"    Mean Tanimoto: {df_res.loc[best_idx, 'Mean_Tanimoto']:.3f}")
    
    with open(os.path.join(results_dir, f"{args.target}_best_epoch.txt"), "w") as f:
        f.write(best_path)
        
    out_csv = os.path.join(results_dir, f"{args.target}_tl_eval.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\n[*] Metrics saved to {out_csv}")

if __name__ == "__main__":
    main()
