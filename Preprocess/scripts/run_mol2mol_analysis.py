#!/usr/bin/env python3
"""
run_mol2mol_analysis.py
========================
Analyzes REINVENT4 mol2mol output:
1. Calculates Morgan Tanimoto similarity between generated molecules and parent leads.
2. Standardizes SMILES to canonical form.
3. Finds the best optimized analogue for each starting lead using customizable heuristics.
4. Generates three distributions in a single high-resolution plot (pIC50, Solubility logS KDEs, Tanimoto histogram).
5. Generates 1-to-1 property shifts (Dumbbell plots) for pIC50 and solubility.
6. Renders 2D structures side-by-side (Original vs. Optimized Pairs) as high-res PNG grids.
7. Saves the processed pairs to a CSV.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from matplotlib.lines import Line2D

def canonical(smi):
    if not smi or pd.isna(smi):
        return None
    try:
        m = Chem.MolFromSmiles(str(smi))
        if not m:
            return None
        # Neutralize formal charges
        for atom in m.GetAtoms():
            if atom.GetFormalCharge() != 0:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(0)
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except:
        try:
            # Fallback to standard canonicalization if neutralization fails
            m = Chem.MolFromSmiles(str(smi))
            if m:
                return Chem.MolToSmiles(m)
        except:
            pass
        return None

def compute_tanimoto(smi1, smi2):
    try:
        m1 = Chem.MolFromSmiles(str(smi1))
        m2 = Chem.MolFromSmiles(str(smi2))
        if m1 and m2:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, 2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, 2048)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        pass
    return np.nan

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze REINVENT4 Mol2Mol results.")
    parser.add_argument("--results_csv", required=True, help="Path to mol2mol results CSV from REINVENT")
    parser.add_argument("--leads_csv", required=True, help="Path to input starting leads CSV (e.g., top15_balanced_with_TL.csv)")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated CSV and plots")
    parser.add_argument("--run_name", default="mol2mol_run", help="Run prefix for naming files")
    parser.add_argument("--artifacts_dir", default=None, help="Optional App Data artifacts directory to copy final outputs")
    
    # Column configuration
    parser.add_argument("--pic50_col", default="PD1PDL1pIC50 (raw)", help="pIC50 column name in results CSV")
    parser.add_argument("--sol_col", default="PD1PDL1Sol (raw)", help="Solubility column name in results CSV")
    parser.add_argument("--sa_col", default="SAScore (raw)", help="SAScore column name in results CSV")
    parser.add_argument("--score_col", default="Score", help="Total score column name in results CSV")
    
    # Thresholds
    parser.add_argument("--target_pic50", type=float, default=8.5, help="Target pIC50 value for vlines")
    parser.add_argument("--target_sol", type=float, default=-3.0, help="Target solubility value for vlines")
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    print(f"[*] Reading starting leads CSV: {args.leads_csv}")
    if not os.path.exists(args.leads_csv):
        print(f"[ERROR] Starting leads CSV not found: {args.leads_csv}")
        sys.exit(1)
    df_leads = pd.read_csv(args.leads_csv)
    
    print(f"[*] Reading Mol2Mol results CSV: {args.results_csv}")
    if not os.path.exists(args.results_csv):
        print(f"[ERROR] Mol2Mol results CSV not found: {args.results_csv}")
        sys.exit(1)
    df_res = pd.read_csv(args.results_csv)
    
    # ---------------------------------------------------------
    # 2. Preprocess & Standardize SMILES
    # ---------------------------------------------------------
    print("[*] Standardizing SMILES to canonical format...")
    # Resolve columns dynamically for leads
    smi_col_leads = 'canonical_smiles' if 'canonical_smiles' in df_leads.columns else ('smiles' if 'smiles' in df_leads.columns else None)
    pic50_col_leads = 'pic50' if 'pic50' in df_leads.columns else ('predicted_pic50' if 'predicted_pic50' in df_leads.columns else None)
    sol_col_leads = 'solubility' if 'solubility' in df_leads.columns else ('predicted_solubility' if 'predicted_solubility' in df_leads.columns else None)
    
    if not smi_col_leads:
        print("[ERROR] Could not identify SMILES column in starting leads CSV.")
        sys.exit(1)
        
    df_leads['canonical_smiles_std'] = df_leads[smi_col_leads].apply(canonical)
    
    # Standardize results SMILES
    if 'Input_SMILES' not in df_res.columns or 'SMILES' not in df_res.columns:
        print("[ERROR] Mol2Mol CSV must contain 'Input_SMILES' and 'SMILES' columns.")
        sys.exit(1)
        
    df_res['canonical_input'] = df_res['Input_SMILES'].apply(canonical)
    df_res['canonical_gen'] = df_res['SMILES'].apply(canonical)
    
    # Drop rows that failed canonicalization
    df_res = df_res.dropna(subset=['canonical_input', 'canonical_gen'])
    
    # Check and dynamically guess results columns if specified ones are missing
    # 1. pIC50 / Activity column
    if args.pic50_col not in df_res.columns:
        matched = [c for c in df_res.columns if c.strip().lower() == args.pic50_col.strip().lower()]
        if matched:
            args.pic50_col = matched[0]
        else:
            guesses = [c for c in df_res.columns if any(x in c.lower() for x in ['pic50', 'activity', 'active', 'pred_p'])]
            if guesses:
                args.pic50_col = guesses[0]
                print(f"[*] Guessed pIC50 column: '{args.pic50_col}'")
            else:
                print(f"[WARNING] Could not find or guess pIC50 column. Creating NaNs.")
                df_res[args.pic50_col] = np.nan

    # 2. Solubility column
    if args.sol_col not in df_res.columns:
        matched = [c for c in df_res.columns if c.strip().lower() == args.sol_col.strip().lower()]
        if matched:
            args.sol_col = matched[0]
        else:
            guesses = [c for c in df_res.columns if any(x in c.lower() for x in ['sol', 'logs', 'solubility', 'log_s'])]
            if guesses:
                args.sol_col = guesses[0]
                print(f"[*] Guessed solubility column: '{args.sol_col}'")
            else:
                print(f"[WARNING] Could not find or guess solubility column. Creating NaNs.")
                df_res[args.sol_col] = np.nan

    # 3. SA Score column
    if args.sa_col not in df_res.columns:
        matched = [c for c in df_res.columns if c.strip().lower() == args.sa_col.strip().lower()]
        if matched:
            args.sa_col = matched[0]
        else:
            guesses = [c for c in df_res.columns if any(x in c.lower() for x in ['sa', 'synthetic', 'sascore'])]
            if guesses:
                args.sa_col = guesses[0]
                print(f"[*] Guessed SAScore column: '{args.sa_col}'")
            else:
                print(f"[WARNING] Could not find or guess SAScore column. Creating NaNs.")
                df_res[args.sa_col] = np.nan
    
    # ---------------------------------------------------------
    # 3. Calculate Tanimoto Similarity to Parents
    # ---------------------------------------------------------
    print("[*] Calculating Morgan fingerprint Tanimoto similarities...")
    tanimoto_vals = []
    for idx, row in df_res.iterrows():
        tanimoto_vals.append(compute_tanimoto(row['canonical_input'], row['canonical_gen']))
    df_res['Tanimoto'] = tanimoto_vals
    
    # ---------------------------------------------------------
    # 4. Filter and Find Best Analogue for Each Lead
    # ---------------------------------------------------------
    print("[*] Matching starting leads with best optimized analogues...")
    pairs_data = []
    pair_mols = []
    pair_legends = []
    
    for idx, row in df_leads.iterrows():
        lead_smi = row['canonical_smiles_std']
        lead_pic50 = row[pic50_col_leads] if pic50_col_leads else np.nan
        lead_sol = row[sol_col_leads] if sol_col_leads else np.nan
        mol_id = row['mol_id'] if 'mol_id' in row else f"lead_{idx+1}"
        epoch = int(row['epoch']) if 'epoch' in row else 0
        
        lead_m = Chem.MolFromSmiles(lead_smi)
        if not lead_m:
            print(f"[!] Invalid lead SMILES for {mol_id}: {lead_smi}")
            continue
            
        # Filter generated candidates for this specific lead
        df_cand = df_res[df_res['canonical_input'] == lead_smi].copy()
        if df_cand.empty:
            # Fallback to string matching on input column
            orig_smi = row[smi_col_leads]
            df_cand = df_res[df_res['Input_SMILES'] == orig_smi].copy()
            
        if df_cand.empty:
            print(f"[!] No generated candidates found for lead {mol_id}")
            continue
            
        # Clean candidates
        df_cand = df_cand.dropna(subset=['Tanimoto'])
        
        # Stepwise relaxed filters to find the best candidate
        # We look for Tanimoto >= 0.40, SAScore <= 4.5, and pic50 >= threshold
        # Then we select the candidate with the highest solubility
        best_cand = None
        for pIC50_threshold in [7.5, 7.0, 6.5, 6.0, 5.0, 0.0]:
            df_filt = df_cand[
                (df_cand['Tanimoto'] >= 0.40) & 
                (df_cand[args.sa_col] <= 4.5) & 
                (df_cand[args.pic50_col] >= pIC50_threshold)
            ]
            if not df_filt.empty:
                best_cand = df_filt.sort_values(by=args.sol_col, ascending=False).iloc[0]
                break
                
        if best_cand is None:
            # Fallback 1: Relax SA score and Tanimoto filter
            df_filt = df_cand[df_cand['Tanimoto'] >= 0.30]
            if not df_filt.empty:
                best_cand = df_filt.sort_values(by=args.sol_col, ascending=False).iloc[0]
                
        if best_cand is None:
            # Fallback 2: Pick candidate with highest solubility
            best_cand = df_cand.sort_values(by=args.sol_col, ascending=False).iloc[0]
            
        opt_smi = best_cand['SMILES']
        opt_pic50 = best_cand[args.pic50_col]
        opt_sol = best_cand[args.sol_col]
        opt_sa = best_cand[args.sa_col]
        tan = best_cand['Tanimoto']
        sol_imp = opt_sol - lead_sol
        
        pair_row = {
            "Rank": idx + 1,
            "mol_id": mol_id,
            "epoch": epoch,
            "lead_smiles": row[smi_col_leads],
            "lead_pic50": lead_pic50,
            "lead_solubility": lead_sol,
            "optimized_smiles": opt_smi,
            "optimized_pic50": opt_pic50,
            "optimized_solubility": opt_sol,
            "sascore": opt_sa,
            "tanimoto": tan,
            "solubility_improvement": sol_imp
        }
        pairs_data.append(pair_row)
        
        # Prepare 2D coordinates for RDKit drawing
        opt_m = Chem.MolFromSmiles(opt_smi)
        if opt_m:
            AllChem.Compute2DCoords(lead_m)
            AllChem.Compute2DCoords(opt_m)
            pair_mols.extend([lead_m, opt_m])
            
            lead_legend = (
                f"Rank {idx+1} | {mol_id}\n"
                f"Parent (Ep {epoch})\n"
                f"pIC50: {lead_pic50:.2f}\n"
                f"logS: {lead_sol:.2f}"
            )
            opt_legend = (
                f"Rank {idx+1} | Optimized\n"
                f"pIC50: {opt_pic50:.2f}\n"
                f"logS: {opt_sol:.2f} ({'+' if sol_imp >= 0 else ''}{sol_imp:.2f})\n"
                f"Tan: {tan:.2f} | SA: {opt_sa:.2f}"
            )
            pair_legends.extend([lead_legend, opt_legend])
            
    df_pairs = pd.DataFrame(pairs_data)
    out_csv_path = os.path.join(args.output_dir, f"{args.run_name}_top15_pairs.csv")
    df_pairs.to_csv(out_csv_path, index=False)
    print(f"[+] Saved matched pairs CSV: {out_csv_path}")
    
    # ---------------------------------------------------------
    # 5. Plot Property KDE Distributions and Tanimoto Histogram
    # ---------------------------------------------------------
    print("[*] Generating KDE distribution plots...")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f"Mol2Mol Optimization — {args.run_name.upper()} Property Distributions", fontsize=15, weight='bold', y=0.98)
    
    # Colors
    color_lead = "#3498db" # Sleek blue
    color_gen = "#2ecc71"  # Harmonious green
    color_opt = "#e74c3c"  # Premium red
    
    # 5a. pIC50 KDE
    if pic50_col_leads and pic50_col_leads in df_leads.columns:
        sns.kdeplot(df_leads[pic50_col_leads].dropna(), label="Original Leads", color=color_lead, fill=True, alpha=0.15, lw=2.5, ax=axes[0])
    sns.kdeplot(df_res[args.pic50_col].dropna(), label="Generated Pool", color=color_gen, fill=True, alpha=0.1, lw=2.0, ax=axes[0])
    sns.kdeplot(df_pairs["optimized_pic50"].dropna(), label="Selected Optimized", color=color_opt, fill=True, alpha=0.15, lw=2.5, ax=axes[0])
    axes[0].axvline(args.target_pic50, color="gold", ls="--", lw=1.5, label=f"Target pIC50: {args.target_pic50}")
    axes[0].set_title("pIC50 Distribution", weight="bold")
    axes[0].set_xlabel("pIC50")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=9)
    
    # 5b. Solubility logS KDE
    if sol_col_leads and sol_col_leads in df_leads.columns:
        sns.kdeplot(df_leads[sol_col_leads].dropna(), label="Original Leads", color=color_lead, fill=True, alpha=0.15, lw=2.5, ax=axes[1])
    sns.kdeplot(df_res[args.sol_col].dropna(), label="Generated Pool", color=color_gen, fill=True, alpha=0.1, lw=2.0, ax=axes[1])
    sns.kdeplot(df_pairs["optimized_solubility"].dropna(), label="Selected Optimized", color=color_opt, fill=True, alpha=0.15, lw=2.5, ax=axes[1])
    axes[1].axvline(args.target_sol, color="gold", ls="--", lw=1.5, label=f"Threshold logS: {args.target_sol}")
    axes[1].set_title("Solubility (logS) Distribution", weight="bold")
    axes[1].set_xlabel("logS")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=9)
    
    # 5c. Tanimoto Similarity Histogram
    tan_vals = df_res['Tanimoto'].dropna()
    sns.histplot(tan_vals, bins=30, ax=axes[2], color="#9b59b6", alpha=0.75, edgecolor="white", kde=True)
    mean_tan = tan_vals.mean()
    med_tan = tan_vals.median()
    axes[2].axvline(mean_tan, color="black", ls="--", lw=1.5, label=f"Mean: {mean_tan:.3f}")
    axes[2].axvline(med_tan, color="#555555", ls=":", lw=1.5, label=f"Median: {med_tan:.3f}")
    axes[2].set_title("Tanimoto Similarity to Parent Lead", weight="bold")
    axes[2].set_xlabel("Tanimoto Similarity")
    axes[2].set_ylabel("Frequency")
    axes[2].legend(fontsize=9)
    
    plt.tight_layout()
    dist_plot_path = os.path.join(args.output_dir, f"{args.run_name}_distributions.png")
    plt.savefig(dist_plot_path, dpi=200)
    plt.close()
    print(f"[+] Saved distributions plot: {dist_plot_path}")
    
    # ---------------------------------------------------------
    # 6. Plot 1-to-1 Shifts (Dumbbell Plot)
    # ---------------------------------------------------------
    print("[*] Generating 1-to-1 property shift dumbbell plots...")
    df_shifts = df_pairs.iloc[::-1].reset_index(drop=True) # Reverse for top-down ranking visual
    
    fig_shift, axes_shift = plt.subplots(1, 2, figsize=(15, 8.5))
    fig_shift.suptitle(f"Lead Optimization Shifts — {args.run_name.upper()} 1-to-1 Pair Shifts", fontsize=15, weight='bold', y=0.98)
    
    y_positions = np.arange(len(df_shifts))
    
    # 6a. pIC50 Shifts
    axes_shift[0].hlines(y=y_positions, xmin=df_shifts[['lead_pic50', 'optimized_pic50']].min(axis=1), xmax=df_shifts[['lead_pic50', 'optimized_pic50']].max(axis=1), color='grey', alpha=0.4, lw=1.5)
    axes_shift[0].scatter(df_shifts['lead_pic50'], y_positions, color=color_lead, s=90, label='Parent Lead', zorder=3)
    axes_shift[0].scatter(df_shifts['optimized_pic50'], y_positions, color=color_opt, s=90, label='Optimized Analogue', zorder=3)
    axes_shift[0].axvline(args.target_pic50, color="gold", ls="--", lw=1.5, zorder=1)
    axes_shift[0].set_yticks(y_positions)
    axes_shift[0].set_yticklabels(df_shifts['mol_id'], weight='bold')
    axes_shift[0].set_title('pIC50 Shift per Compound', weight='bold')
    axes_shift[0].set_xlabel('pIC50')
    axes_shift[0].legend(loc='lower left')
    
    # 6b. Solubility Shifts
    axes_shift[1].hlines(y=y_positions, xmin=df_shifts[['lead_solubility', 'optimized_solubility']].min(axis=1), xmax=df_shifts[['lead_solubility', 'optimized_solubility']].max(axis=1), color='grey', alpha=0.4, lw=1.5)
    axes_shift[1].scatter(df_shifts['lead_solubility'], y_positions, color=color_lead, s=90, label='Parent Lead', zorder=3)
    axes_shift[1].scatter(df_shifts['optimized_solubility'], y_positions, color=color_opt, s=90, label='Optimized Analogue', zorder=3)
    axes_shift[1].axvline(args.target_sol, color="gold", ls="--", lw=1.5, zorder=1)
    axes_shift[1].set_yticks(y_positions)
    axes_shift[1].set_yticklabels(df_shifts['mol_id'], weight='bold')
    axes_shift[1].set_title('Solubility (logS) Shift per Compound', weight='bold')
    axes_shift[1].set_xlabel('Solubility (logS)')
    axes_shift[1].legend(loc='lower right')
    
    plt.tight_layout()
    shift_plot_path = os.path.join(args.output_dir, f"{args.run_name}_pair_shifts.png")
    plt.savefig(shift_plot_path, dpi=200)
    plt.close()
    print(f"[+] Saved shift dumbbell plot: {shift_plot_path}")
    
    # ---------------------------------------------------------
    # 7. Render RDKit Side-by-Side Molecular Structure Grid
    # ---------------------------------------------------------
    print("[*] Rendering RDKit pair grids...")
    dopts = Draw.rdMolDraw2D.MolDrawOptions()
    dopts.legendFontSize = 32
    dopts.bondLineWidth = 2.5
    
    # Draw pairs grid (3 pairs per row -> 6 columns)
    # We render pair_mols which has [lead_1, opt_1, lead_2, opt_2, ...]
    if pair_mols:
        img = Draw.MolsToGridImage(
            pair_mols,
            molsPerRow=6,
            subImgSize=(500, 520),
            legends=pair_legends,
            useSVG=False,
            drawOptions=dopts
        )
        grid_plot_path = os.path.join(args.output_dir, f"{args.run_name}_pairs_grid.png")
        img.save(grid_plot_path)
        print(f"[+] Saved RDKit structural pair grid: {grid_plot_path}")
    else:
        grid_plot_path = None
        print("[!] No molecular structures to render.")
        
    # ---------------------------------------------------------
    # 8. Copy to Artifacts Directory
    # ---------------------------------------------------------
    if args.artifacts_dir:
        os.makedirs(args.artifacts_dir, exist_ok=True)
        print(f"[*] Copying outputs to artifacts directory: {args.artifacts_dir}")
        os.system(f"cp {out_csv_path} {args.artifacts_dir}/")
        os.system(f"cp {dist_plot_path} {args.artifacts_dir}/")
        os.system(f"cp {shift_plot_path} {args.artifacts_dir}/")
        if grid_plot_path:
            os.system(f"cp {grid_plot_path} {args.artifacts_dir}/")
        print("[+] Artifact copies completed successfully.")
        
    print("\n" + "="*50)
    print(" MOL2MOL POST-RUN ANALYSIS COMPLETE ")
    print("="*50)
    print(f"Matched CSV:  {out_csv_path}")
    print(f"Dist KDE:     {dist_plot_path}")
    print(f"Dumbbells:    {shift_plot_path}")
    if grid_plot_path:
        print(f"RDKit Grid:   {grid_plot_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
