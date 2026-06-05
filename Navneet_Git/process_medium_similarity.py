import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw

def canonical(smi):
    if not smi or pd.isna(smi):
        return None
    try:
        m = Chem.MolFromSmiles(str(smi))
        if not m:
            return None
        return Chem.MolToSmiles(m)
    except:
        return None

def main():
    repo = "/Users/vishnukasturi/Intern/reinvent-local"
    input_csv_path = f"{repo}/Navneet_Git/MolID_Epoch/top15_balanced_with_TL.csv"
    results_csv_path = f"{repo}/results/pd1_pdl1_mol2mol_medium_similarity_1.csv"
    
    out_dir = f"{repo}/Navneet_Git/mol2mol_results/medium_similarity"
    os.makedirs(out_dir, exist_ok=True)
    
    out_csv = f"{out_dir}/top15_pairs.csv"
    out_png = f"{out_dir}/top15_pairs.png"
    out_kde = f"{out_dir}/distributions.png"
    
    # Also save to artifacts
    brain_dir = "/Users/vishnukasturi/.gemini/antigravity/brain/b1740afb-b51b-4bea-901b-35388d53206f/artifacts"
    os.makedirs(brain_dir, exist_ok=True)
    
    print("[*] Loading input top 15 compounds...")
    df_input = pd.read_csv(input_csv_path)
    
    print("[*] Loading generated results...")
    df_res = pd.read_csv(results_csv_path)
    
    # Standardize SMILES
    df_input['canonical_smiles'] = df_input['smiles'].apply(canonical)
    df_res['canonical_input'] = df_res['Input_SMILES'].apply(canonical)
    df_res['canonical_gen'] = df_res['SMILES'].apply(canonical)
    
    # ---------------------------------------------------------
    # 1. Distribution Plots (pIC50, Solubility, Tanimoto)
    # ---------------------------------------------------------
    print("[*] Generating distribution plots...")
    
    # Compute Tanimoto for all generated vs their input
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
    
    # pIC50
    sns.kdeplot(df_input['pic50'], ax=axes[0], label='Original Leads', fill=True, color='blue')
    sns.kdeplot(df_res['PD1PDL1pIC50 (raw)'], ax=axes[0], label='Generated (Medium Sim)', fill=True, color='green')
    axes[0].set_title('pIC50 Distribution')
    axes[0].set_xlabel('pIC50')
    axes[0].legend()
    
    # Solubility
    sns.kdeplot(df_input['solubility'], ax=axes[1], label='Original Leads', fill=True, color='blue')
    sns.kdeplot(df_res['PD1PDL1Sol (raw)'], ax=axes[1], label='Generated (Medium Sim)', fill=True, color='green')
    axes[1].set_title('Solubility (logS) Distribution')
    axes[1].set_xlabel('logS')
    axes[1].legend()
    
    # Tanimoto
    sns.histplot(df_res['Tanimoto'].dropna(), bins=30, ax=axes[2], color='purple', kde=True)
    axes[2].set_title('Tanimoto Similarity to Parent')
    axes[2].set_xlabel('Tanimoto Similarity')
    
    plt.tight_layout()
    plt.savefig(out_kde)
    plt.savefig(f"{brain_dir}/medium_sim_distributions.png")
    plt.close()
    print("[+] Saved distribution plots.")
    
    # ---------------------------------------------------------
    # 2. Extract best pair per lead
    # ---------------------------------------------------------
    print("[*] Finding best generated analogues...")
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
        lead_fp = AllChem.GetMorganFingerprintAsBitVect(lead_m, 2, 2048)
        
        df_cand = df_res[df_res['canonical_input'] == lead_smi].copy()
        if df_cand.empty:
            df_cand = df_res[df_res['Input_SMILES'] == row['smiles']].copy()
            
        if df_cand.empty:
            continue
            
        # Filter for good ones
        # For medium similarity, maybe we want tanimoto 0.4 - 0.7, good pIC50, good sol
        df_cand = df_cand.dropna(subset=['Tanimoto'])
        df_cand = df_cand[df_cand['SAScore (raw)'] <= 4.5]
        
        # Prefer higher pIC50 and higher solubility
        # We can rank by sum of normalized scores or simply pick best pIC50 among those that don't drop solubility much
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
            pair_legends.append(
                f"Rank {idx+1} | {mol_id}\n"
                f"pIC50: {lead_pic50:.2f}\n"
                f"logS: {lead_sol:.2f}"
            )
            pair_legends.append(
                f"Optimized Analogue\n"
                f"pIC50: {opt_pic50:.2f}\n"
                f"logS: {opt_sol:.2f} ({'+' if sol_diff>=0 else ''}{sol_diff:.2f})\n"
                f"Tan: {tan:.2f} | SA: {opt_sa:.2f}"
            )
            
    df_pairs = pd.DataFrame(pairs_data)
    df_pairs.to_csv(out_csv, index=False)
    df_pairs.to_csv(f"{brain_dir}/medium_sim_pairs.csv", index=False)
    
    dopts = Draw.rdMolDraw2D.MolDrawOptions()
    dopts.legendFontSize = 36
    img = Draw.MolsToGridImage(
        pair_mols,
        molsPerRow=6,
        subImgSize=(500, 500),
        legends=pair_legends,
        useSVG=False,
        drawOptions=dopts
    )
    img.save(out_png)
    img.save(f"{brain_dir}/medium_sim_pairs_grid.png")
    
    print("[+] Done! Results saved.")

if __name__ == "__main__":
    main()
