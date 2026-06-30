import os
import pandas as pd
from rdkit import Chem

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
    repo_dir = "/Users/vishnukasturi/Intern/reinvent-local"
    csv_file = os.path.join(repo_dir, "Preprocess/Data_pd1_pdl1/data_csvs/pd1_pdl1_preprocess_pic50_with_sol.csv")
    output_smi_neutral = os.path.join(repo_dir, "Navneet_Git/psearch.smi")
    output_smi_original = os.path.join(repo_dir, "Navneet_Git/psearch_original.smi")
    output_csv_ref = os.path.join(repo_dir, "Navneet_Git/psearchreferecnenew.csv")
    
    # Read the input csv
    print(f"[*] Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"[+] Loaded {len(df)} rows.")
    
    # Keep the original row index as a column
    df = df.reset_index(names='original_index')
    
    # Clean, neutralize and canonicalize
    print("[*] Neutralizing and canonicalizing SMILES...")
    df['neutral_smiles'] = df['smiles'].apply(neutralize_and_canonical)
    
    # Drop rows where sanitization/parsing failed
    df = df.dropna(subset=['neutral_smiles'])
    print(f"[+] Successfully parsed {len(df)} molecules.")
    
    # Deduplicate by neutralized SMILES to avoid duplication issues in conformer generation
    df = df.drop_duplicates(subset=['neutral_smiles'])
    print(f"[+] Unique molecules: {len(df)}")
    
    # Sort descending by pic50
    df_sorted = df.sort_values(by='pic50', ascending=False).reset_index(drop=True)
    
    # Select top 50 highest pic50
    top50 = df_sorted.head(50).copy()
    top50['activity'] = 1
    
    # Select bottom 50 lowest pic50
    bottom50 = df_sorted.tail(50).copy()
    bottom50['activity'] = 0
    
    # Combine the top 50 and bottom 50
    combined = pd.concat([top50, bottom50], ignore_index=True)
    
    # Assign names mol1 to mol100
    combined['mol_name'] = [f"mol{i}" for i in range(1, 101)]
    
    # Reorder columns to match the exact schema of novel_test_1.csv:
    # 4 columns: [original_index, mol_name, pic50_score, smiles]
    # No header, tab-separated
    ref_df = combined[[
        'original_index', 'mol_name', 'pic50', 'smiles'
    ]]
    
    # Check ranges of pic50
    print("\n--- Summary ---")
    print(f"Actives (mol1 - mol50) pic50 range: {combined.loc[combined['activity']==1, 'pic50'].min()} to {combined.loc[combined['activity']==1, 'pic50'].max()}")
    print(f"Inactives (mol51 - mol100) pic50 range: {combined.loc[combined['activity']==0, 'pic50'].min()} to {combined.loc[combined['activity']==0, 'pic50'].max()}")
    
    # Save output smiles files
    # 1. Neutralized SMILES (Standard for PSearch database creation)
    with open(output_smi_neutral, 'w') as f:
        f.write("smiles\tmol_name\tactivity\n")
        for _, r in combined.iterrows():
            f.write(f"{r['neutral_smiles']}\t{r['mol_name']}\t{r['activity']}\n")
            
    # 2. Original SMILES
    with open(output_smi_original, 'w') as f:
        f.write("smiles\tmol_name\tactivity\n")
        for _, r in combined.iterrows():
            f.write(f"{r['smiles']}\t{r['mol_name']}\t{r['activity']}\n")
            
    # 3. Reference CSV (tab-separated, headerless, matching novel_test_1.csv structure)
    ref_df.to_csv(output_csv_ref, sep='\t', header=False, index=False)
            
    print(f"\n[+] Successfully generated 100-molecule psearch datasets:")
    print(f"  - Neutralized SMILES: {output_smi_neutral}")
    print(f"  - Original SMILES: {output_smi_original}")
    print(f"  - Reference CSV (headerless, tab-separated): {output_csv_ref}")

if __name__ == "__main__":
    main()
