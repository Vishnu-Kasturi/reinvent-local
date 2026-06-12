#!/usr/bin/env python3
"""
filter_tl_novel_scaffold.py
===========================
Filters molecules out of TL_novel_scafold.csv that contain the generalized biaryl query structure:
  Clc1c(*)cccc1c1cccc(c1Cl)*
Adjusts query properties using RDKit's AdjustQueryProperties (making atoms and bonds generic).
"""

import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def main():
    repo = "/Users/vishnukasturi/Intern/reinvent-local"
    csv_path = os.path.join(repo, "Preprocess/Data_pd1_pdl1/data_csvs/TL_novel_scafold.csv")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found at {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    print(f"[*] Loaded {len(df)} rows from {csv_path}")
    
    if 'smiles' not in df.columns:
        print("[ERROR] 'smiles' column not found in CSV.")
        sys.exit(1)
        
    # Define query SMILES
    query_smi = 'Clc1c(*)cccc1c1cccc(c1Cl)*'
    query_mol = Chem.MolFromSmiles(query_smi)
    
    if not query_mol:
        print(f"[ERROR] Could not parse query SMILES: {query_smi}")
        sys.exit(1)
        
    # Adjust query parameters to make atom and bond matching generic
    params = AllChem.AdjustQueryParameters()
    params.makeAtomsGeneric = True  # Allows atom types to match more broadly
    params.makeBondsGeneric = True  # Allows bond types to match broadly
    
    # Create the generalized query
    generalized_query = AllChem.AdjustQueryProperties(query_mol, params)
    
    # Filter molecules
    keep_indices = []
    removed_count = 0
    removed_examples = []
    
    for idx, row in df.iterrows():
        smi = row['smiles']
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            # Check for generalized substructure match
            if mol.HasSubstructMatch(generalized_query):
                removed_count += 1
                if len(removed_examples) < 5:
                    removed_examples.append(smi)
            else:
                keep_indices.append(idx)
        else:
            # Keep invalid SMILES (or skip them, but keeping is safer)
            keep_indices.append(idx)
            
    filtered_df = df.loc[keep_indices]
    
    print(f"[+] Removed {removed_count} molecules matching the query.")
    if removed_examples:
        print("[*] Sample removed molecules:")
        for ex in removed_examples:
            print(f"  - {ex}")
            
    print(f"[+] Remaining molecules: {len(filtered_df)}")
    
    # Save back to CSV
    filtered_df.to_csv(csv_path, index=False)
    print(f"[+] Filtered CSV successfully saved back to {csv_path}")

if __name__ == "__main__":
    main()
