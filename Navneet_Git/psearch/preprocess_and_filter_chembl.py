#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdReducedGraphs
from multiprocessing import Pool, cpu_count

# Reference active molecule mol57
MOL57_SMILES = "Cc1cc(C)c(-c2[nH]c3ncc(-c4cccc(C[N@H+]5CC[N@@H+](C)CC5)c4)cc3c2C)cc1C(=O)N1CCOCC1"

ref_erg_fp = None

def init_worker(ref_fp):
    global ref_erg_fp
    ref_erg_fp = ref_fp

def preprocess_and_calc_erg_sim(row):
    chembl_id, smiles = row
    if not smiles or pd.isna(smiles):
        return None
        
    try:
        # 1. Salt Stripping (select largest fragment by heavy atoms)
        parts = str(smiles).split('.')
        if len(parts) > 1:
            mols = [Chem.MolFromSmiles(p) for p in parts]
            mols = [m for m in mols if m is not None]
            if not mols:
                return None
            mols.sort(key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
            m = mols[0]
        else:
            m = Chem.MolFromSmiles(parts[0])
            if not m:
                return None
                
        # 2. Charge Neutralization
        for atom in m.GetAtoms():
            if atom.GetFormalCharge() != 0:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(0)
        Chem.SanitizeMol(m)
        neutral_smi = Chem.MolToSmiles(m)
        
        # 3. Compute ErG fingerprint
        fp = np.array(rdReducedGraphs.GetErGFingerprint(m))
        
        # 4. Calculate ErG Tanimoto Similarity
        dot_product = np.dot(ref_erg_fp, fp)
        denominator = np.dot(ref_erg_fp, ref_erg_fp) + np.dot(fp, fp) - dot_product
        if denominator == 0.0:
            sim = 0.0
        else:
            sim = float(dot_product / denominator)
            
        return neutral_smi, chembl_id, sim
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description="Preprocess ChEMBL database and filter top compounds by ErG similarity to mol57")
    parser.add_argument("-i", "--input", default="chembl_data.csv", help="Path to raw ChEMBL database CSV file (semicolon separated)")
    parser.add_argument("-o", "--output_smi", default="chembl_top10000_erg.smi", help="Path to output SMILES file for PSearch (.smi)")
    parser.add_argument("-s", "--output_csv", default="chembl_top10000_erg_scores.csv", help="Path to output detailed CSV scores (.csv)")
    parser.add_argument("-c", "--cores", type=int, default=8, help="Number of CPU cores to use")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    print("[*] Generating reference ErG fingerprint for mol57...")
    mol57 = Chem.MolFromSmiles(MOL57_SMILES)
    ref_fp = np.array(rdReducedGraphs.GetErGFingerprint(mol57))
    
    print(f"[*] Loading ChEMBL database from {args.input}...")
    df = pd.read_csv(args.input, sep=';', usecols=["Compound ChEMBL ID", "Smiles"])
    print(f"[+] Loaded {len(df)} compounds.")
    
    data_list = list(df[["Compound ChEMBL ID", "Smiles"]].itertuples(index=False, name=None))
    
    ncpu = min(cpu_count(), args.cores)
    print(f"[+] Processing in parallel using {ncpu} CPUs...")
    
    results = []
    with Pool(ncpu, initializer=init_worker, initargs=(ref_fp,)) as pool:
        for res in pool.imap_unordered(preprocess_and_calc_erg_sim, data_list, chunksize=1000):
            if res is not None:
                results.append(res)
                
    print(f"[+] Calculated ErG similarity for {len(results)} compounds.")
    
    df_sim = pd.DataFrame(results, columns=['smiles', 'mol_name', 'erg_sim'])
    df_sim.drop_duplicates(subset=['smiles'], inplace=True)
    df_sim.sort_values('erg_sim', ascending=False, inplace=True)
    df_sim.reset_index(drop=True, inplace=True)
    
    print("\nTop 15 most ErG-similar preprocessed ChEMBL compounds:")
    print(df_sim.head(15))
    
    df_top10000 = df_sim.head(10000).copy()
    df_top10000['activity'] = 1
    
    df_top10000.to_csv(args.output_csv, index=False)
    print(f"[+] Saved detailed CSV to: {args.output_csv}")
    
    df_top10000[['smiles', 'mol_name', 'activity']].to_csv(args.output_smi, sep='\t', index=False)
    print(f"[+] Saved formatted SMILES to: {args.output_smi}")

if __name__ == "__main__":
    main()
