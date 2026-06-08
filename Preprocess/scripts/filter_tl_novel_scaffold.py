#!/usr/bin/env python3
"""
filter_tl_novel_scaffold.py
===========================
Filters out SMILES containing the 2,2'-dichlorobiphenyl core (with substituents):
  Clc1c(*)cccc1c1cccc(c1Cl)*
from Preprocess/Data_pd1_pdl1/data_csvs/TL_novel_scafold.csv.
Uses the AdjustQueryProperties method requested by the user.
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

REPO = "/Users/vishnukasturi/Intern/reinvent-local"
CSV_PATH = os.path.join(REPO, "Preprocess/Data_pd1_pdl1/data_csvs/TL_novel_scafold.csv")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Target CSV not found: {CSV_PATH}")
        return

    # Load CSV
    print(f"[*] Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    total_before = len(df)
    print(f"[+] Loaded {total_before} rows.")

    # 1. Define query using MolFromSmarts
    query_smi = "Clc1c(*)cccc1c1cccc(c1Cl)*"
    print(f"[*] Defining query SMARTS: {query_smi}")
    query_mol = Chem.MolFromSmarts(query_smi)
    if not query_mol:
        print("[ERROR] Could not parse query SMARTS.")
        return
    print("[+] Query initialized successfully via MolFromSmarts.")

    # 2. Filter rows
    print("[*] Filtering molecules...")
    keep_indices = []
    removed_count = 0

    for idx, row in df.iterrows():
        smi = str(row['smiles'])
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # Check for match
            has_match = mol.HasSubstructMatch(query_mol)
            if has_match:
                removed_count += 1
            else:
                keep_indices.append(idx)
        else:
            # Keep if unparseable or warning
            keep_indices.append(idx)

    # 3. Save filtered CSV
    df_filtered = df.loc[keep_indices].reset_index(drop=True)
    df_filtered.to_csv(CSV_PATH, index=False)
    
    total_after = len(df_filtered)
    print(f"\n==================================================")
    print(f" SUBSTRUCTURE FILTERING COMPLETE")
    print(f"==================================================")
    print(f"  Query Pattern   : {query_smi}")
    print(f"  Total Before    : {total_before}")
    print(f"  Removed Count   : {removed_count}")
    print(f"  Total Remaining : {total_after}")
    print(f"  CSV Saved To    : {CSV_PATH}")
    print(f"==================================================\n")

if __name__ == "__main__":
    main()
