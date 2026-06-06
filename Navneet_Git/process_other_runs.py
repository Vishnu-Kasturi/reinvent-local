import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from matplotlib.lines import Line2D

def canonical(smi):
    if not smi or pd.isna(smi): return None
    try:
        m = Chem.MolFromSmiles(str(smi))
        if not m: return None
        return Chem.MolToSmiles(m)
    except:
        return None

def process_run(run_name, csv_filename):
    repo = "/Users/vishnukasturi/Intern/reinvent-local"
    input_csv_path = f"{repo}/Navneet_Git/MolID_Epoch/top15_balanced_with_TL.csv"
    results_csv_path = f"{repo}/results/{csv_filename}"
    
    out_dir = f"{repo}/Navneet_Git/mol2mol_results/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    out_csv = f"{out_dir}/top15_pairs.csv"
    out_png_grid = f"{out_dir}/top15_pairs_grid.png"
    out_kde = f"{out_dir}/distributions.png"
    out_shifts = f"{out_dir}/pair_shifts_1to1.png"
    
    print(f"\n======================================")
    print(f"[*] Processing run: {run_name}")
    print(f"======================================")
    
    df_input = pd.read_csv(input_csv_path)
    df_res = pd.read_csv(results_csv_path)
    
    df_input['canonical_smiles'] = df_input['smiles'].apply(canonical)
    df_res['canonical_input'] = df_res['Input_SMILES'].apply(canonical)
    df_res['canonical_gen'] = df_res['SMILES'].apply(canonical)
    
    # 1. Distributions
    print(f"[*] Computing Tanimoto similarities for {run_name}...")
    all_tanimoto = []
    for _, row in df_res.iterrows():
        try:
            m1 = Chem.MolFromSmiles(row['canonical_input'])
            m2 = Chem.MolFromSmiles(row['canonical_gen'])
            if m1 and m2:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, 2048)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, 2048)
                all_tanimoto.append(DataStructs.TanimotoSimilarity(fp1, fp2))
            else:
                all_tanimoto.append(np.nan)
        except:
            all_tanimoto.append(np.nan)
            
    df_res['Tanimoto'] = all_tanimoto
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.kdeplot(df_input['pic50'], ax=axes[0], label='Original Leads', fill=True, color='blue')
    sns.kdeplot(df_res['PD1PDL1pIC50 (raw)'], ax=axes[0], label=f'Generated ({run_name})', fill=True, color='green')
    axes[0].set_title('pIC50 Distribution')
    axes[0].set_xlabel('pIC50')
    axes[0].legend()
    
    sns.kdeplot(df_input['solubility'], ax=axes[1], label='Original Leads', fill=True, color='blue')
    sns.kdeplot(df_res['PD1PDL1Sol (raw)'], ax=axes[1], label=f'Generated ({run_name})', fill=True, color='green')
    axes[1].set_title('Solubility (logS) Distribution')
    axes[1].set_xlabel('logS')
    axes[1].legend()
    
    sns.histplot(df_res['Tanimoto'].dropna(), bins=30, ax=axes[2], color='purple', kde=True)
    axes[2].set_title('Tanimoto Similarity to Parent')
    axes[2].set_xlabel('Tanimoto Similarity')
    
    plt.tight_layout()
    plt.savefig(out_kde)
    plt.close()
    print(f"[+] Saved distributions: {out_kde}")
    
    # 2. Extract best pair
    print(f"[*] Finding best generated analogues for {run_name}...")
    pairs_data = []
    pair_mols = []
    pair_legends = []
    
    for idx, row in df_input.iterrows():
        lead_smi = row['canonical_smiles']
        lead_pic50 = row['pic50']
        lead_sol = row['solubility']
        mol_id = row['mol_id']
        
        lead_m = Chem.MolFromSmiles(lead_smi)
        if not lead_m: continue
        
        df_cand = df_res[df_res['canonical_input'] == lead_smi].copy()
        if df_cand.empty:
            df_cand = df_res[df_res['Input_SMILES'] == row['smiles']].copy()
            
        if df_cand.empty:
            continue
            
        df_cand = df_cand.dropna(subset=['Tanimoto'])
        df_cand = df_cand[df_cand['SAScore (raw)'] <= 4.5]
        
        if not df_cand.empty:
            df_cand = df_cand.sort_values(by=['PD1PDL1pIC50 (raw)', 'PD1PDL1Sol (raw)'], ascending=[False, False])
            best_cand = df_cand.iloc[0]
            
            opt_smi = best_cand['canonical_gen']
            opt_m = Chem.MolFromSmiles(opt_smi)
            if not opt_m: continue
            
            opt_pic50 = best_cand['PD1PDL1pIC50 (raw)']
            opt_sol = best_cand['PD1PDL1Sol (raw)']
            opt_sa = best_cand['SAScore (raw)']
            tan = best_cand['Tanimoto']
            
            pairs_data.append({
                "Rank": idx + 1,
                "mol_id": mol_id,
                "lead_smiles": row['smiles'],
                "lead_pic50": lead_pic50,
                "lead_solubility": lead_sol,
                "optimized_smiles": best_cand['SMILES'],
                "optimized_pic50": opt_pic50,
                "optimized_solubility": opt_sol,
                "sascore": opt_sa,
                "tanimoto": tan
            })
            
            AllChem.Compute2DCoords(lead_m)
            AllChem.Compute2DCoords(opt_m)
            pair_mols.extend([lead_m, opt_m])
            
            sol_diff = opt_sol - lead_sol
            pair_legends.append(f"Rank {idx+1} | {mol_id}\npIC50: {lead_pic50:.2f}\nlogS: {lead_sol:.2f}")
            pair_legends.append(f"Optimized Analogue\npIC50: {opt_pic50:.2f}\nlogS: {opt_sol:.2f} ({'+' if sol_diff>=0 else ''}{sol_diff:.2f})\nTan: {tan:.2f} | SA: {opt_sa:.2f}")
            
    df_pairs = pd.DataFrame(pairs_data)
    df_pairs.to_csv(out_csv, index=False)
    print(f"[+] Saved pairs CSV: {out_csv}")
    
    dopts = Draw.rdMolDraw2D.MolDrawOptions()
    dopts.legendFontSize = 36
    img = Draw.MolsToGridImage(pair_mols, molsPerRow=6, subImgSize=(500, 500), legends=pair_legends, useSVG=False, drawOptions=dopts)
    img.save(out_png_grid)
    print(f"[+] Saved pairs grid: {out_png_grid}")
    
    # 3. Plot Shifts
    print(f"[*] Plotting 1-to-1 shifts for {run_name}...")
    df_shifts = df_pairs.iloc[::-1].reset_index(drop=True)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'PD1-PDL1 Leads (Top 15) — Original vs. Optimized Pairs ({run_name})', fontsize=16, fontweight='bold', y=0.98)
    
    color_orig, color_opt, color_target = '#2ecc71', '#e74c3c', '#f1c40f'
    
    # KDE
    sns.kdeplot(data=df_shifts, x='lead_pic50', color=color_orig, fill=True, alpha=0.2, linewidth=2, ax=axes[0,0])
    sns.kdeplot(data=df_shifts, x='optimized_pic50', color=color_opt, fill=True, alpha=0.2, linewidth=2, ax=axes[0,0])
    axes[0,0].legend([Line2D([0], [0], color=color_orig, lw=3), Line2D([0], [0], color=color_opt, lw=3), Line2D([0], [0], color=color_target, lw=2, linestyle='--')], ['Original Leads', 'Optimized Analogues', 'Target pIC50: 8.5'], loc='upper left')
    axes[0,0].axvline(8.5, color=color_target, linestyle='--', zorder=0)
    axes[0,0].set_title('pIC50 Overall Distribution', fontweight='bold')
    axes[0,0].set_xlabel('pIC50')
    axes[0,0].set_ylabel('Density')
    
    sns.kdeplot(data=df_shifts, x='lead_solubility', color=color_orig, fill=True, alpha=0.2, linewidth=2, ax=axes[0,1])
    sns.kdeplot(data=df_shifts, x='optimized_solubility', color=color_opt, fill=True, alpha=0.2, linewidth=2, ax=axes[0,1])
    axes[0,1].legend([Line2D([0], [0], color=color_orig, lw=3), Line2D([0], [0], color=color_opt, lw=3), Line2D([0], [0], color=color_target, lw=2, linestyle='--')], ['Original Leads', 'Optimized Analogues', 'TOML Threshold: -3.0'], loc='upper right')
    axes[0,1].axvline(-3.0, color=color_target, linestyle='--', zorder=0)
    axes[0,1].set_title('Solubility (logS) Overall Distribution', fontweight='bold')
    axes[0,1].set_xlabel('logS')
    axes[0,1].set_ylabel('Density')
    
    # Shifts
    y_positions = np.arange(len(df_shifts))
    axes[1,0].hlines(y=y_positions, xmin=df_shifts[['lead_pic50', 'optimized_pic50']].min(axis=1), xmax=df_shifts[['lead_pic50', 'optimized_pic50']].max(axis=1), color='grey', alpha=0.5)
    axes[1,0].scatter(df_shifts['lead_pic50'], y_positions, color=color_orig, s=80, label='Original', zorder=3)
    axes[1,0].scatter(df_shifts['optimized_pic50'], y_positions, color=color_opt, s=80, label='Optimized', zorder=3)
    axes[1,0].axvline(8.5, color=color_target, linestyle='--', zorder=0)
    axes[1,0].set_yticks(y_positions)
    axes[1,0].set_yticklabels(df_shifts['mol_id'])
    axes[1,0].set_title('1-to-1 pIC50 Shifts', fontweight='bold')
    axes[1,0].set_xlabel('pIC50')
    axes[1,0].legend(loc='lower left')
    
    axes[1,1].hlines(y=y_positions, xmin=df_shifts[['lead_solubility', 'optimized_solubility']].min(axis=1), xmax=df_shifts[['lead_solubility', 'optimized_solubility']].max(axis=1), color='grey', alpha=0.5)
    axes[1,1].scatter(df_shifts['lead_solubility'], y_positions, color=color_orig, s=80, label='Original', zorder=3)
    axes[1,1].scatter(df_shifts['optimized_solubility'], y_positions, color=color_opt, s=80, label='Optimized', zorder=3)
    axes[1,1].axvline(-3.0, color=color_target, linestyle='--', zorder=0)
    axes[1,1].set_yticks(y_positions)
    axes[1,1].set_yticklabels(df_shifts['mol_id'])
    axes[1,1].set_title('1-to-1 Solubility (logS) Shifts', fontweight='bold')
    axes[1,1].set_xlabel('logS')
    axes[1,1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(out_shifts, dpi=300)
    plt.close()
    print(f"[+] Saved pair shifts: {out_shifts}")

def main():
    runs = [
        ("mmp", "pd1_pdl1_mol2mol_mmp_1.csv"),
        ("scaffold_generic", "pd1_pdl1_mol2mol_scaffold_generic_1.csv")
    ]
    for r_name, csv_file in runs:
        process_run(r_name, csv_file)

if __name__ == "__main__":
    main()
