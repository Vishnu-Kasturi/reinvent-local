import os
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

def main():
    repo = "/Users/vishnukasturi/Intern/reinvent-local"
    baseline_path = f"{repo}/Preprocess/Data_pd1_pdl1/pd1_pdl1_pic50_raw.csv"
    
    # 1. Load baseline and precompute fingerprints
    print("[*] Loading baseline dataset...")
    df_base = pd.read_csv(baseline_path, sep='\t')
    base_fps = []
    for s in df_base['SMILES'].dropna():
        m = Chem.MolFromSmiles(str(s))
        if m:
            base_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
    print(f"[+] Loaded {len(base_fps)} baseline fingerprints.")

    # Datasets to process
    datasets = [
        {
            "name": "with_TL",
            "path": f"{repo}/Navneet_Git/RL_filtered_processed.csv",
            "out_png": f"{repo}/Navneet_Git/top15_balanced_with_TL.png",
            "brain_png": "/Users/vishnukasturi/.gemini/antigravity/brain/47052de4-b6d7-432f-a23a-37a447b1885e/top15_balanced_with_TL.png"
        },
        {
            "name": "without_TL",
            "path": f"{repo}/Navneet_Git/RL_without_TL_filtered_processed.csv",
            "out_png": f"{repo}/Navneet_Git/top15_balanced_without_TL.png",
            "brain_png": "/Users/vishnukasturi/.gemini/antigravity/brain/47052de4-b6d7-432f-a23a-37a447b1885e/top15_balanced_without_TL.png"
        }
    ]

    for d in datasets:
        print(f"\n[*] Processing dataset: {d['name']}")
        df = pd.read_csv(d['path'])
        
        # Calculate combined 1:1 MPO score: pIC50 + solubility
        df['combined_score'] = df['pic50'] + df['solubility']
        
        # Sort by combined score descending
        df_sorted = df.sort_values(by='combined_score', ascending=False)
        
        # Retrieve top 15
        top15 = df_sorted.head(15).copy()
        
        mols = []
        legends = []
        
        for idx, (_, row) in enumerate(top15.iterrows()):
            smi = row['canonical_smiles']
            m = Chem.MolFromSmiles(smi)
            if m:
                AllChem.Compute2DCoords(m)
                mols.append(m)
                
                # Calculate Tanimoto similarity to baseline
                fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
                sims = DataStructs.BulkTanimotoSimilarity(fp, base_fps)
                max_tan = max(sims) if sims else 0.0
                
                # Create legend
                legend = (
                    f"Rank {idx+1}\n"
                    f"pIC50: {row['pic50']:.2f} | logS: {row['solubility']:.2f}\n"
                    f"Docking: {row['docking_score']:.2f} kcal\n"
                    f"Tanimoto: {max_tan:.3f}"
                )
                legends.append(legend)
        
        # Draw molecules grid (3 rows of 5 columns)
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=5,
            subImgSize=(350, 320),
            legends=legends,
            useSVG=False
        )
        
        # Save to Navneet_Git
        img.save(d['out_png'])
        print(f"[+] Saved grid to {d['out_png']}")
        
        # Copy to brain dir
        os.makedirs(os.path.dirname(d['brain_png']), exist_ok=True)
        os.system(f"cp {d['out_png']} {d['brain_png']}")
        print(f"[+] Copied grid to {d['brain_png']}")

if __name__ == "__main__":
    main()
