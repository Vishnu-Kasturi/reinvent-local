#!/usr/bin/env python3
"""
cluster_tl_novel.py
===================
Performs Butina clustering on the filtered molecules in TL_novel_scafold.csv:
1. Computes Morgan Fingerprints (radius=2, 2048-bit).
2. Calculates pairwise Tanimoto distances.
3. Groups molecules using Butina clustering (cutoff=0.40).
4. Selects the centroid (representative) compound from the top 15 largest clusters.
5. Draws the representative molecules in a high-resolution grid PNG.
6. Copies the output image to the app data artifacts folder for rendering.
"""

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.ML.Cluster import Butina

def main():
    repo = "/Users/vishnukasturi/Intern/reinvent-local"
    csv_path = os.path.join(repo, "Preprocess/Data_pd1_pdl1/data_csvs/TL_novel_scafold.csv")
    output_png = os.path.join(repo, "results/TL_novel_clusters.png")
    artifacts_dir = "/Users/vishnukasturi/.gemini/antigravity-ide/brain/ca6de21d-68e4-4095-a68c-c9c39230f853/artifacts"
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    print(f"[*] Loaded {len(df)} molecules from {csv_path}")
    
    # 1. Parse molecules and compute fingerprints
    valid_mols = []
    valid_rows = []
    
    for idx, row in df.iterrows():
        smi = row['smiles']
        m = Chem.MolFromSmiles(str(smi))
        if m:
            valid_mols.append(m)
            valid_rows.append(row)
            
    num_mols = len(valid_mols)
    print(f"[+] Successfully parsed {num_mols} valid molecules.")
    
    if num_mols == 0:
        print("[ERROR] No valid molecules found for clustering.")
        sys.exit(1)
        
    print("[*] Generating Morgan fingerprints...")
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in valid_mols]
    
    # 2. Compute lower-triangular pairwise distance matrix (1 - Tanimoto)
    print("[*] Calculating pairwise distance matrix...")
    dists = []
    for i in range(1, num_mols):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        for s in sims:
            dists.append(1.0 - s)
            
    # 3. Perform Butina clustering (cutoff=0.40 -> ~60% similarity threshold)
    cutoff = 0.40
    print(f"[*] Running Butina clustering (cutoff={cutoff})...")
    clusters = Butina.ClusterData(dists, num_mols, cutoff, isDistData=True)
    
    # Sort clusters by size in descending order
    clusters = sorted(clusters, key=len, reverse=True)
    print(f"[+] Formed {len(clusters)} clusters.")
    print(f"[*] Largest cluster size: {len(clusters[0])}")
    
    # 4. Extract representative centroids of the top 12 or 15 clusters
    num_to_draw = min(15, len(clusters))
    draw_mols = []
    draw_legends = []
    
    for c_idx in range(num_to_draw):
        cluster = clusters[c_idx]
        centroid_idx = cluster[0]  # Centroid is the first item in the Butina cluster tuple
        centroid_mol = valid_mols[centroid_idx]
        centroid_row = valid_rows[centroid_idx]
        
        # Details
        pic50 = centroid_row.get('pic50', np.nan)
        logs = centroid_row.get('logS', np.nan)
        
        draw_mols.append(centroid_mol)
        draw_legends.append(
            f"Cluster {c_idx+1} (Size: {len(cluster)})\n"
            f"pIC50: {pic50:.2f}\n"
            f"logS: {logs:.2f}"
        )
        
    # 5. Draw molecular grid image
    print(f"[*] Drawing grid image for top {num_to_draw} cluster representatives...")
    dopts = Draw.rdMolDraw2D.MolDrawOptions()
    dopts.legendFontSize = 26
    dopts.bondLineWidth = 2.0
    
    img = Draw.MolsToGridImage(
        draw_mols,
        molsPerRow=3,
        subImgSize=(450, 420),
        legends=draw_legends,
        useSVG=False,
        drawOptions=dopts
    )
    
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    img.save(output_png)
    print(f"[+] Saved cluster grid image to: {output_png}")
    
    # 6. Copy to brain artifacts dir
    if os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir, exist_ok=True)
        art_path = os.path.join(artifacts_dir, "TL_novel_clusters.png")
        os.system(f"cp {output_png} {art_path}")
        print(f"[+] Successfully copied grid image to artifacts: {art_path}")
        
if __name__ == "__main__":
    main()
