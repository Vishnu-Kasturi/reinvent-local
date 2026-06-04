import os
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw

def main():
    repo = "/Users/vishnukasturi/Intern/reinvent-local"
    baseline_path = f"{repo}/Preprocess/Data_pd1_pdl1/pd1_pdl1_pic50_raw.csv"
    
    print("[*] Loading baseline dataset...")
    df_base = pd.read_csv(baseline_path, sep='\t')
    df_base = df_base.dropna(subset=['SMILES'])
    
    base_mols = []
    base_fps = []
    base_pic50s = []
    base_smiles = []
    
    for _, row in df_base.iterrows():
        s = str(row['SMILES'])
        m = Chem.MolFromSmiles(s)
        if m:
            base_mols.append(m)
            base_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
            base_pic50s.append(row['pic50'])
            base_smiles.append(s)
            
    print(f"[+] Loaded {len(base_fps)} valid baseline molecules.")

    datasets = [
        {
            "name": "with_TL",
            "csv_path": f"{repo}/Navneet_Git/top15_balanced_with_TL.csv",
            "out_png": f"{repo}/Navneet_Git/top15_balanced_with_TL_pairs.png",
            "brain_png": "/Users/vishnukasturi/.gemini/antigravity/brain/47052de4-b6d7-432f-a23a-37a447b1885e/top15_balanced_with_TL_pairs.png",
            "out_csv": f"{repo}/Navneet_Git/top15_balanced_with_TL_pairs.csv",
            "brain_csv": "/Users/vishnukasturi/.gemini/antigravity/brain/47052de4-b6d7-432f-a23a-37a447b1885e/top15_balanced_with_TL_pairs.csv"
        },
        {
            "name": "without_TL",
            "csv_path": f"{repo}/Navneet_Git/top15_balanced_without_TL.csv",
            "out_png": f"{repo}/Navneet_Git/top15_balanced_without_TL_pairs.png",
            "brain_png": "/Users/vishnukasturi/.gemini/antigravity/brain/47052de4-b6d7-432f-a23a-37a447b1885e/top15_balanced_without_TL_pairs.png",
            "out_csv": f"{repo}/Navneet_Git/top15_balanced_without_TL_pairs.csv",
            "brain_csv": "/Users/vishnukasturi/.gemini/antigravity/brain/47052de4-b6d7-432f-a23a-37a447b1885e/top15_balanced_without_TL_pairs.csv"
        }
    ]

    for d in datasets:
        print(f"\n[*] Processing top 15 dataset: {d['name']}")
        df = pd.read_csv(d['csv_path'])
        
        pair_mols = []
        pair_legends = []
        pair_data = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            gen_smi = row['canonical_smiles']
            gen_m = Chem.MolFromSmiles(gen_smi)
            if not gen_m:
                continue
            
            # Find max Tanimoto baseline match
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_m, 2, 2048)
            sims = DataStructs.BulkTanimotoSimilarity(gen_fp, base_fps)
            
            max_idx = np.argmax(sims)
            max_tan = sims[max_idx]
            
            base_smi = base_smiles[max_idx]
            base_m = Chem.MolFromSmiles(base_smi)
            base_pic50 = base_pic50s[max_idx]
            
            # Record data for CSV
            pair_data.append({
                "Rank": idx + 1,
                "generated_smiles": gen_smi,
                "predicted_pic50": row['pic50'],
                "predicted_solubility": row['solubility'],
                "docking_score": row['docking_score'],
                "combined_score": row['combined_score'],
                "tanimoto_similarity": max_tan,
                "matched_baseline_smiles": base_smi,
                "matched_baseline_original_pic50": base_pic50
            })
            
            # Prepare generated molecule coords and legend
            AllChem.Compute2DCoords(gen_m)
            pair_mols.append(gen_m)
            
            gen_legend = (
                f"Rank {idx+1} (Gen)\n"
                f"Pred pIC50: {row['pic50']:.2f}\n"
                f"Pred logS: {row['solubility']:.2f}\n"
                f"Tan: {max_tan:.3f}"
            )
            pair_legends.append(gen_legend)
            
            # Prepare baseline match molecule coords and legend
            if base_m:
                AllChem.Compute2DCoords(base_m)
                pair_mols.append(base_m)
            else:
                # Fallback if mol generation fails
                pair_mols.append(Chem.MolFromSmiles(""))
                
            base_legend = (
                f"Rank {idx+1} (Base Match)\n"
                f"Orig pIC50: {base_pic50:.2f}"
            )
            pair_legends.append(base_legend)
            
        # Save CSV
        df_pairs = pd.DataFrame(pair_data)
        df_pairs.to_csv(d['out_csv'], index=False)
        print(f"[+] Saved pair CSV to {d['out_csv']}")
        
        # Copy CSV to brain dir
        os.makedirs(os.path.dirname(d['brain_csv']), exist_ok=True)
        df_pairs.to_csv(d['brain_csv'], index=False)
        print(f"[+] Saved pair CSV to {d['brain_csv']}")
            
        # Draw grid image (3 pairs per row -> 6 columns, 5 rows)
        img = Draw.MolsToGridImage(
            pair_mols,
            molsPerRow=6,
            subImgSize=(300, 300),
            legends=pair_legends,
            useSVG=False
        )
        
        img.save(d['out_png'])
        print(f"[+] Saved pair grid to {d['out_png']}")
        
        # Copy to brain dir
        os.makedirs(os.path.dirname(d['brain_png']), exist_ok=True)
        os.system(f"cp {d['out_png']} {d['brain_png']}")
        print(f"[+] Copied pair grid to {d['brain_png']}")

if __name__ == "__main__":
    main()
