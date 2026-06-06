import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw

# Load REINVENT features plugin
sys.path.append('REINVENT4')
from reinvent_plugins.components.pd1_pdl1_features import compute_features

def main():
    repo = "/Users/vishnukasturi/Intern/reinvent-local"
    baseline_path = f"{repo}/Preprocess/Data_pd1_pdl1/pd1_pdl1_pic50_raw.csv"
    
    pic50_model_path = f"{repo}/Preprocess/final_acc/pd1_pdl1_pic50_final_acc_model.ubj"
    pic50_scaler_path = f"{repo}/Preprocess/final_acc/pd1_pdl1_pic50_final_acc_scaler.pkl"
    sol_model_path = f"{repo}/Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj"
    sol_scaler_path = f"{repo}/Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl"
    
    print("[*] Loading baseline dataset...")
    df_base = pd.read_csv(baseline_path, sep='\t')
    df_base = df_base.dropna(subset=['SMILES'])
    
    # Load predictor models
    print("[*] Loading pIC50 and solubility models...")
    bst_pic50 = xgb.Booster()
    bst_pic50.load_model(pic50_model_path)
    bst_sol = xgb.Booster()
    bst_sol.load_model(sol_model_path)
    
    base_mols = []
    base_fps = []
    base_smiles = []
    
    for _, row in df_base.iterrows():
        s = str(row['SMILES'])
        m = Chem.MolFromSmiles(s)
        if m:
            base_mols.append(m)
            base_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
            base_smiles.append(s)
            
    print(f"[+] Loaded {len(base_fps)} valid baseline molecules.")

    datasets = [
        {
            "name": "novel_ha38_46",
            "csv_path": f"{repo}/results/pd1_pdl1_rl_run7_top15_novel_ha38_46_docking_results.csv",
            "grid_png_src": f"{repo}/results/pd1_pdl1_rl_run7_top15_novel_ha38_46_grid.png",
            "grid_png_dst": f"{repo}/Navneet_Git/pd1_pdl1_rl_run7_top15_novel_ha38_46_grid.png",
            "out_png": f"{repo}/Navneet_Git/pd1_pdl1_rl_run7_top15_novel_ha38_46_pairs.png",
            "out_csv": f"{repo}/Navneet_Git/pd1_pdl1_rl_run7_top15_novel_ha38_46_pairs.csv",
            "brain_png": "/Users/vishnukasturi/.gemini/antigravity/brain/47052de4-b6d7-432f-a23a-37a447b1885e/pd1_pdl1_rl_run7_top15_novel_ha38_46_pairs.png",
            "brain_csv": "/Users/vishnukasturi/.gemini/antigravity/brain/47052de4-b6d7-432f-a23a-37a447b1885e/pd1_pdl1_rl_run7_top15_novel_ha38_46_pairs.csv",
        },
        {
            "name": "novel_ha35_43",
            "csv_path": f"{repo}/results/pd1_pdl1_rl_run7_top15_novel_ha35_43_docking_results.csv",
            "grid_png_src": f"{repo}/results/pd1_pdl1_rl_run7_top15_novel_ha35_43_grid.png",
            "grid_png_dst": f"{repo}/Navneet_Git/pd1_pdl1_rl_run7_top15_novel_ha35_43_grid.png",
            "out_png": f"{repo}/Navneet_Git/pd1_pdl1_rl_run7_top15_novel_ha35_43_pairs.png",
            "out_csv": f"{repo}/Navneet_Git/pd1_pdl1_rl_run7_top15_novel_ha35_43_pairs.csv",
            "brain_png": "/Users/vishnukasturi/.gemini/antigravity/brain/47052de4-b6d7-432f-a23a-37a447b1885e/pd1_pdl1_rl_run7_top15_novel_ha35_43_pairs.png",
            "brain_csv": "/Users/vishnukasturi/.gemini/antigravity/brain/47052de4-b6d7-432f-a23a-37a447b1885e/pd1_pdl1_rl_run7_top15_novel_ha35_43_pairs.csv",
        },
        {
            "name": "novel_ha",
            "csv_path": f"{repo}/results/pd1_pdl1_rl_run7_top15_novel_ha_docking_results.csv",
            "grid_png_src": f"{repo}/results/pd1_pdl1_rl_run7_top15_novel_ha_grid.png",
            "grid_png_dst": f"{repo}/Navneet_Git/pd1_pdl1_rl_run7_top15_novel_ha_grid.png",
            "out_png": f"{repo}/Navneet_Git/pd1_pdl1_rl_run7_top15_novel_ha_pairs.png",
            "out_csv": f"{repo}/Navneet_Git/pd1_pdl1_rl_run7_top15_novel_ha_pairs.csv",
            "brain_png": "/Users/vishnukasturi/.gemini/antigravity/brain/47052de4-b6d7-432f-a23a-37a447b1885e/pd1_pdl1_rl_run7_top15_novel_ha_pairs.png",
            "brain_csv": "/Users/vishnukasturi/.gemini/antigravity/brain/47052de4-b6d7-432f-a23a-37a447b1885e/pd1_pdl1_rl_run7_top15_novel_ha_pairs.csv",
        }
    ]

    for d in datasets:
        print(f"\n[*] Processing: {d['name']}")
        
        # Copy the original grid PNG to Navneet_Git/
        if os.path.exists(d['grid_png_src']):
            os.system(f"cp {d['grid_png_src']} {d['grid_png_dst']}")
            print(f"[+] Copied original grid to {d['grid_png_dst']}")
        else:
            print(f"[-] Warning: Source grid not found: {d['grid_png_src']}")
            
        df = pd.read_csv(d['csv_path'])
        
        # Determine baseline matches first
        base_matched_smiles = []
        for idx, (_, row) in enumerate(df.iterrows()):
            gen_smi = row['SMILES']
            gen_m = Chem.MolFromSmiles(gen_smi)
            if gen_m:
                gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_m, 2, 2048)
                sims = DataStructs.BulkTanimotoSimilarity(gen_fp, base_fps)
                max_idx = np.argmax(sims)
                base_matched_smiles.append(base_smiles[max_idx])
            else:
                base_matched_smiles.append("")
                
        # Predict properties for matched baseline SMILES
        valid_base_smiles = [s for s in base_matched_smiles if s != ""]
        base_pic50s = {}
        base_sols = {}
        if valid_base_smiles:
            # Predict pIC50
            X_p, m_p = compute_features(valid_base_smiles, pic50_scaler_path)
            preds_p = bst_pic50.predict(xgb.DMatrix(X_p[:, :2415]))
            
            # Predict solubility
            X_s, m_s = compute_features(valid_base_smiles, sol_scaler_path)
            preds_s = bst_sol.predict(xgb.DMatrix(X_s))
            
            pred_idx = 0
            for s in base_matched_smiles:
                if s == "":
                    continue
                if m_p[pred_idx]:
                    base_pic50s[s] = float(preds_p[pred_idx])
                else:
                    base_pic50s[s] = np.nan
                    
                if m_s[pred_idx]:
                    base_sols[s] = float(preds_s[pred_idx])
                else:
                    base_sols[s] = np.nan
                pred_idx += 1

        pair_mols = []
        pair_legends = []
        pair_data = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            gen_smi = row['SMILES']
            gen_m = Chem.MolFromSmiles(gen_smi)
            if not gen_m:
                continue
            
            # Find max Tanimoto baseline match
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_m, 2, 2048)
            sims = DataStructs.BulkTanimotoSimilarity(gen_fp, base_fps)
            
            max_idx = np.argmax(sims)
            max_tan = sims[max_idx]
            
            base_smi = base_matched_smiles[idx]
            base_m = Chem.MolFromSmiles(base_smi)
            base_pic50 = base_pic50s.get(base_smi, np.nan)
            base_sol = base_sols.get(base_smi, np.nan)
            
            # Record data for CSV
            pair_data.append({
                "Rank": idx + 1,
                "generated_smiles": gen_smi,
                "predicted_pic50": row['pIC50'],
                "predicted_solubility": row['logS'],
                "heavy_atoms": row['HeavyAtoms'],
                "docking_score": row['DockingScore'],
                "tanimoto_similarity": max_tan,
                "matched_baseline_smiles": base_smi,
                "matched_baseline_predicted_pic50": base_pic50,
                "matched_baseline_predicted_solubility": base_sol
            })
            
            # Prepare generated molecule coords and legend
            AllChem.Compute2DCoords(gen_m)
            pair_mols.append(gen_m)
            
            gen_legend = (
                f"Rank {idx+1} (Gen)\n"
                f"Pred pIC50: {row['pIC50']:.2f}\n"
                f"Pred logS: {row['logS']:.2f}\n"
                f"Tan: {max_tan:.3f}"
            )
            pair_legends.append(gen_legend)
            
            # Prepare baseline match molecule coords and legend
            if base_m:
                AllChem.Compute2DCoords(base_m)
                pair_mols.append(base_m)
            else:
                pair_mols.append(Chem.MolFromSmiles(""))
                
            base_legend = (
                f"Rank {idx+1} (Base Match)\n"
                f"Pred pIC50: {base_pic50:.2f}\n"
                f"Pred logS: {base_sol:.2f}"
            )
            pair_legends.append(base_legend)
            
        # Save CSV
        df_pairs = pd.DataFrame(pair_data)
        df_pairs.to_csv(d['out_csv'], index=False)
        print(f"[+] Saved pair CSV to {d['out_csv']}")
        
        # Copy CSV to brain dir
        os.makedirs(os.path.dirname(d['brain_csv']), exist_ok=True)
        df_pairs.to_csv(d['brain_csv'], index=False)
        
        # Configure drawing options (increased legend font size for high readability)
        dopts = Draw.rdMolDraw2D.MolDrawOptions()
        dopts.legendFontSize = 36
        
        # Draw grid image (3 pairs per row -> 6 columns, 5 rows)
        img = Draw.MolsToGridImage(
            pair_mols,
            molsPerRow=6,
            subImgSize=(500, 500),
            legends=pair_legends,
            useSVG=False,
            drawOptions=dopts
        )
        
        img.save(d['out_png'])
        print(f"[+] Saved pair grid to {d['out_png']}")
        
        # Copy to brain dir
        os.makedirs(os.path.dirname(d['brain_png']), exist_ok=True)
        os.system(f"cp {d['out_png']} {d['brain_png']}")

if __name__ == "__main__":
    main()
