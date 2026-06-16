#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.ML.Cluster import Butina

def cluster_and_draw(csv_path, output_png, cutoff=0.40, max_draw=15):
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    print(f"\n[*] Loaded {len(df)} molecules from {csv_path}")
    
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
        return
        
    print("[*] Generating Morgan fingerprints...")
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in valid_mols]
    
    print("[*] Calculating pairwise distance matrix...")
    dists = []
    for i in range(1, num_mols):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        for s in sims:
            dists.append(1.0 - s)
            
    print(f"[*] Running Butina clustering (cutoff={cutoff})...")
    if num_mols > 1:
        clusters = Butina.ClusterData(dists, num_mols, cutoff, isDistData=True)
        clusters = sorted(clusters, key=len, reverse=True)
    else:
        clusters = [(0,)]
        
    print(f"[+] Formed {len(clusters)} clusters.")
    print(f"[*] Largest cluster size: {len(clusters[0])}")
    
    num_to_draw = min(max_draw, len(clusters))
    draw_mols = []
    draw_legends = []
    
    for c_idx in range(num_to_draw):
        cluster = clusters[c_idx]
        centroid_idx = cluster[0]
        centroid_mol = valid_mols[centroid_idx]
        centroid_row = valid_rows[centroid_idx]
        
        chembl_id = centroid_row['chembl_id']
        
        draw_mols.append(centroid_mol)
        draw_legends.append(
            f"{chembl_id}\n"
            f"Cluster {c_idx+1} (Size: {len(cluster)})"
        )
        
    print(f"[*] Drawing grid image for top {num_to_draw} cluster representatives...")
    dopts = Draw.rdMolDraw2D.MolDrawOptions()
    dopts.legendFontSize = 20
    dopts.bondLineWidth = 2.0
    
    mols_per_row = min(3, num_to_draw) if num_to_draw > 0 else 1
    
    img = Draw.MolsToGridImage(
        draw_mols,
        molsPerRow=mols_per_row,
        subImgSize=(350, 320),
        legends=draw_legends,
        useSVG=False,
        drawOptions=dopts
    )
    
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    img.save(output_png)
    print(f"[+] Saved cluster grid image to: {output_png}")

def main():
    parser = argparse.ArgumentParser(description="Cluster screening hits using Butina algorithm")
    parser.add_argument("-i", "--input_csv", help="Single CSV file of hits to cluster (if not provided, clusters all hits in the current dir)")
    parser.add_argument("-o", "--output_png", help="Path to save the output cluster grid image")
    parser.add_argument("-c", "--cutoff", type=float, default=0.40, help="Tanimoto distance cutoff for Butina clustering")
    args = parser.parse_args()

    if args.input_csv:
        if not args.output_png:
            base = os.path.splitext(os.path.basename(args.input_csv))[0]
            args.output_png = f"{base}_clusters.png"
        cluster_and_draw(args.input_csv, args.output_png, cutoff=args.cutoff)
    else:
        # Auto-detect hits files in current directory
        hit_files = [f for f in os.listdir('.') if f.startswith('hits_') and f.endswith('.csv')]
        if not hit_files:
            print("[ERROR] No hits_*.csv files detected in the current directory. Please specify --input_csv.")
            sys.exit(1)
            
        print(f"[+] Detected hit CSV files for clustering: {hit_files}")
        for f in hit_files:
            base = os.path.splitext(f)[0]
            out_png = f"{base}_clusters.png"
            cluster_and_draw(f, out_png, cutoff=args.cutoff)

if __name__ == "__main__":
    main()
