import os, sys, pickle, xgboost as xgb
import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

RES = 'results'
RL_CSV = f'{RES}/pd1_pdl1_pic50_only_rl_1.csv'
PIC50 = 'PD1PDL1pIC50_raw (raw)'
SA_C = 'SAScore (raw)'

# Load generated data
print("[*] Loading RL generated data...")
df = pd.read_csv(RL_CSV).dropna(subset=[PIC50])

# Keep potent ones to save inference time
df = df[df[PIC50] > 9.0].drop_duplicates('SMILES').copy()
smiles = df['SMILES'].tolist()

print(f"[*] Found {len(smiles)} potent, unique SMILES. Predicting solubility...")

# Compute FPs
fps = []
valid_smiles = []
valid_pic50 = []
valid_sa = []
for i, row in df.iterrows():
    m = Chem.MolFromSmiles(row['SMILES'])
    if m:
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        fps.append(np.array(fp))
        valid_smiles.append(row['SMILES'])
        valid_pic50.append(row[PIC50])
        valid_sa.append(row[SA_C])

X = np.array(fps)

# Load XGBoost model and scaler
model = xgb.Booster()
model.load_model('Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj')
with open('Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Predict
dmatrix = xgb.DMatrix(X)
raw_preds = model.predict(dmatrix)
sol_preds = scaler.inverse_transform(raw_preds.reshape(-1, 1)).flatten()

# Combine results
results = []
for s, p, sa, sol in zip(valid_smiles, valid_pic50, valid_sa, sol_preds):
    results.append({'SMILES': s, 'pIC50': p, 'SA': sa, 'logS': sol})

df_res = pd.DataFrame(results)

# Filter for solubility (logS > -3.5)
filtered = df_res[df_res['logS'] > -3.5].sort_values('pIC50', ascending=False)
print(f"[*] {len(filtered)} molecules passed the solubility filter!")

if len(filtered) == 0:
    print("[-] No molecules passed the filter. Try loosening the threshold.")
    sys.exit(0)

# Take top 12
top_hits = filtered.head(12)
mols = []
legends = []
for i, row in top_hits.iterrows():
    m = Chem.MolFromSmiles(row['SMILES'])
    mols.append(m)
    legends.append(f"pIC50: {row['pIC50']:.2f}\nlogS: {row['logS']:.2f}\nSA: {row['SA']:.2f}")

# Draw grid
img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 300), legends=legends, returnPNG=False)
img.save(f'{RES}/pd1_pdl1_pic50_sol_filtered_grid.png')

print(f"[+] Saved grid image to {RES}/pd1_pdl1_pic50_sol_filtered_grid.png")
