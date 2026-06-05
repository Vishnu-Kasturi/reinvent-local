import os
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw

def neutralize_and_canonical(smi):
    if not smi or pd.isna(smi):
        return None
    try:
        m = Chem.MolFromSmiles(str(smi))
        if not m:
            return None
        for atom in m.GetAtoms():
            if atom.GetFormalCharge() != 0:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(0)
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except:
        return None

def main():
    repo = "/Users/vishnukasturi/Intern/reinvent-local"
    input_csv_path = f"{repo}/Navneet_Git/MolID_Epoch/top15_balanced_with_TL.csv"
    results_csv_path = f"{repo}/results/pd1_pdl1_mol2mol_sol_opt_1.csv"
    
    out_csv = f"{repo}/Navneet_Git/MolID_Epoch/top15_mol2mol_sol_opt_pairs.csv"
    out_png = f"{repo}/Navneet_Git/MolID_Epoch/top15_mol2mol_sol_opt_pairs.png"
    brain_csv = "/Users/vishnukasturi/.gemini/antigravity/brain/47052de4-b6d7-432f-a23a-37a447b1885e/top15_mol2mol_sol_opt_pairs.csv"
    brain_png = "/Users/vishnukasturi/.gemini/antigravity/brain/47052de4-b6d7-432f-a23a-37a447b1885e/top15_mol2mol_sol_opt_pairs.png"
    
    print("[*] Loading input top 15 compounds...")
    df_input = pd.read_csv(input_csv_path)
    print(f"[+] Loaded {len(df_input)} compounds from input CSV.")
    
    print("[*] Loading mol2mol optimization results...")
    df_res = pd.read_csv(results_csv_path)
    print(f"[+] Loaded {len(df_res)} generated compound rows from mol2mol results.")
    
    # Pre-canonicalize SMILES to ensure robust matching
    df_input['canonical_smiles'] = df_input['smiles'].apply(canonical)
    df_res['canonical_input'] = df_res['Input_SMILES'].apply(canonical)
    df_res['canonical_gen'] = df_res['SMILES'].apply(canonical)
    
    pairs_data = []
    pair_mols = []
    pair_legends = []
    
    for idx, row in df_input.iterrows():
        mol_id = row['mol_id']
        epoch = int(row['epoch'])
        lead_smi = row['canonical_smiles']
        lead_pic50 = row['pic50']
        lead_sol = row['solubility']
        
        lead_m = Chem.MolFromSmiles(lead_smi)
        if not lead_m:
            print(f"[!] Invalid lead SMILES for {mol_id}: {lead_smi}")
            continue
            
        lead_fp = AllChem.GetMorganFingerprintAsBitVect(lead_m, 2, 2048)
        
        # Filter generated pool for this lead's candidates
        df_cand = df_res[df_res['canonical_input'] == lead_smi].copy()
        
        if df_cand.empty:
            # Fallback: check if the original SMILES matches instead of canonical
            df_cand = df_res[df_res['Input_SMILES'] == row['smiles']].copy()
            
        if df_cand.empty:
            print(f"[!] No generated candidates found for lead {mol_id} ({lead_smi[:30]}...)")
            continue
            
        # Calculate Tanimoto similarity for each candidate
        cand_fps = []
        cand_sims = []
        valid_indices = []
        
        for c_idx, c_row in df_cand.iterrows():
            c_m = Chem.MolFromSmiles(c_row['SMILES'])
            if c_m:
                c_fp = AllChem.GetMorganFingerprintAsBitVect(c_m, 2, 2048)
                sim = DataStructs.TanimotoSimilarity(lead_fp, c_fp)
                cand_sims.append(sim)
                valid_indices.append(c_idx)
            else:
                cand_sims.append(-1)
                valid_indices.append(c_idx)
                
        df_cand['tanimoto'] = cand_sims
        df_cand = df_cand[df_cand['tanimoto'] >= 0.0]  # keep only valid RDKit molecules
        
        # Filters:
        # 1. Tanimoto >= 0.40
        # 2. SAScore (raw) <= 4.5
        # 3. PD1PDL1pIC50 (raw) >= 7.0 (or close to parent)
        # Try to find the best candidate with stepwise relaxation of pIC50 filter if necessary
        for pIC50_threshold in [7.5, 7.0, 6.5, 6.0, 5.0, 0.0]:
            df_filt = df_cand[
                (df_cand['tanimoto'] >= 0.40) & 
                (df_cand['SAScore (raw)'] <= 4.5) & 
                (df_cand['PD1PDL1pIC50 (raw)'] >= pIC50_threshold)
            ]
            if not df_filt.empty:
                break
                
        if df_filt.empty:
            # If still empty, relax SA score filter and Tanimoto filter as last resort
            df_filt = df_cand[df_cand['tanimoto'] >= 0.30]
            
        if df_filt.empty:
            # Absolute fallback: just pick highest solubility candidate
            df_filt = df_cand
            
        # Select the candidate with the highest solubility
        best_cand = df_filt.sort_values(by='PD1PDL1Sol (raw)', ascending=False).iloc[0]
        
        opt_smi = best_cand['SMILES']
        opt_pic50 = best_cand['PD1PDL1pIC50 (raw)']
        opt_sol = best_cand['PD1PDL1Sol (raw)']
        opt_sa = best_cand['SAScore (raw)']
        tan = best_cand['tanimoto']
        sol_imp = opt_sol - lead_sol
        
        pair_row = {
            "Rank": idx + 1,
            "mol_id": mol_id,
            "epoch": epoch,
            "lead_smiles": row['smiles'],
            "lead_predicted_pic50": lead_pic50,
            "lead_predicted_solubility": lead_sol,
            "optimized_smiles": opt_smi,
            "optimized_predicted_pic50": opt_pic50,
            "optimized_predicted_solubility": opt_sol,
            "optimized_sascore": opt_sa,
            "tanimoto_similarity": tan,
            "solubility_improvement": sol_imp
        }
        pairs_data.append(pair_row)
        
        # Prepare drawings
        opt_m = Chem.MolFromSmiles(opt_smi)
        
        # Add coordinates
        AllChem.Compute2DCoords(lead_m)
        AllChem.Compute2DCoords(opt_m)
        
        pair_mols.append(lead_m)
        pair_mols.append(opt_m)
        
        lead_legend = (
            f"Rank {idx+1} | {mol_id} (Lead Ep {epoch})\n"
            f"Pred pIC50: {lead_pic50:.2f}\n"
            f"Pred logS: {lead_sol:.2f}"
        )
        opt_legend = (
            f"Rank {idx+1} | Optimized Analogue\n"
            f"Pred pIC50: {opt_pic50:.2f}\n"
            f"Pred logS: {opt_sol:.2f} ({'+' if sol_imp >= 0 else ''}{sol_imp:.2f})\n"
            f"Tan: {tan:.3f} | SA: {opt_sa:.2f}"
        )
        
        pair_legends.append(lead_legend)
        pair_legends.append(opt_legend)
        
        print(f"[+] Lead {mol_id}: Solubility improved from {lead_sol:.2f} to {opt_sol:.2f} (diff: {sol_imp:+.2f}), Tanimoto: {tan:.3f}")
        
    # Save CSV
    df_pairs = pd.DataFrame(pairs_data)
    df_pairs.to_csv(out_csv, index=False)
    df_pairs.to_csv(brain_csv, index=False)
    print(f"[+] Saved paired CSV results to:\n    - {out_csv}\n    - {brain_csv}")
    
    # Draw image grid
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
    os.makedirs(os.path.dirname(brain_png), exist_ok=True)
    os.system(f"cp {out_png} {brain_png}")
    print(f"[+] Saved paired comparison grid PNG to:\n    - {out_png}\n    - {brain_png}")

if __name__ == "__main__":
    main()
