import os, sys, warnings
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

warnings.filterwarnings("ignore")

REPO_ROOT = "/Users/vishnukasturi/Intern/reinvent-local"
CSV_PATH = os.path.join(REPO_ROOT, "results", "jak2_rl_candidates_run6.csv")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found")
        sys.exit(1)
        
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} candidates for plotting.")
    
    # Drop duplicates by canonical smiles to be absolutely unique
    def get_canon(s):
        try:
            m = Chem.MolFromSmiles(s)
            return Chem.MolToSmiles(m) if m else None
        except:
            return None
            
    df['canonical'] = df['SMILES'].apply(get_canon)
    df = df.dropna(subset=['canonical']).drop_duplicates('canonical')
    print(f"Unique unique candidates: {len(df)}")
    
    # Define combined score
    df['combined_score'] = df['pIC50'] + 1.5 * df['logS']
    
    # Grid configuration
    mols_per_row = 4
    sub_img_size = (300, 250)
    
    # ── 1. Top 20 by pIC50 ──
    df_pic50 = df.sort_values(by='pIC50', ascending=False).head(20)
    mols_p, legends_p = [], []
    for idx, (_, row) in enumerate(df_pic50.iterrows()):
        m = Chem.MolFromSmiles(row['canonical'])
        if m:
            mols_p.append(m)
            legends_p.append(f"Rank {idx+1}\npIC50: {row['pIC50']:.2f}\nlogS: {row['logS']:.2f}\nSA: {row['SA']:.2f}")
    img_p = Draw.MolsToGridImage(mols_p, molsPerRow=mols_per_row, subImgSize=sub_img_size, legends=legends_p)
    out_p = os.path.join(RESULTS_DIR, "jak2_rl_top20_pic50.png")
    img_p.save(out_p)
    print(f"Saved top 20 pIC50 grid to {out_p}")
    
    # ── 2. Top 20 by logS ──
    df_sol = df.sort_values(by='logS', ascending=False).head(20)
    mols_s, legends_s = [], []
    for idx, (_, row) in enumerate(df_sol.iterrows()):
        m = Chem.MolFromSmiles(row['canonical'])
        if m:
            mols_s.append(m)
            legends_s.append(f"Rank {idx+1}\nlogS: {row['logS']:.2f}\npIC50: {row['pIC50']:.2f}\nSA: {row['SA']:.2f}")
    img_s = Draw.MolsToGridImage(mols_s, molsPerRow=mols_per_row, subImgSize=sub_img_size, legends=legends_s)
    out_s = os.path.join(RESULTS_DIR, "jak2_rl_top20_sol.png")
    img_s.save(out_s)
    print(f"Saved top 20 solubility grid to {out_s}")
    
    # ── 3. Top 20 Balanced (pIC50 >= 7.5, SA <= 4.0, sorted by pIC50 + 1.5*logS) ──
    df_filtered = df[(df['pIC50'] >= 7.5) & (df['SA'] <= 4.0)].copy()
    if len(df_filtered) < 20:
        print("Fallback: lowering filter to pIC50 >= 7.0 for balanced grid")
        df_filtered = df[(df['pIC50'] >= 7.0) & (df['SA'] <= 4.5)].copy()
        
    df_bal = df_filtered.sort_values(by='combined_score', ascending=False).head(20)
    mols_b, legends_b = [], []
    for idx, (_, row) in enumerate(df_bal.iterrows()):
        m = Chem.MolFromSmiles(row['canonical'])
        if m:
            mols_b.append(m)
            legends_b.append(f"Rank {idx+1}\nScore: {row['combined_score']:.2f}\npIC50: {row['pIC50']:.2f}\nlogS: {row['logS']:.2f}\nSA: {row['SA']:.2f}")
    img_b = Draw.MolsToGridImage(mols_b, molsPerRow=mols_per_row, subImgSize=sub_img_size, legends=legends_b)
    out_b = os.path.join(RESULTS_DIR, "jak2_rl_top20_balanced.png")
    img_b.save(out_b)
    print(f"Saved top 20 balanced grid to {out_b}")

if __name__ == "__main__":
    main()
