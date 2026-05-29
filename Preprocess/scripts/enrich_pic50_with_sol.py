"""
Enrich the processed pIC50 dataset with predicted solubility (logS)
using the pd1_pdl1_sol XGBoost model.
Saves result to: Preprocess/Data_pd1_pdl1/data_csvs/pd1_pdl1_preprocess_pic50_with_sol.csv
"""
import sys, os, pandas as pd, numpy as np, xgboost as xgb
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

sys.path.append('REINVENT4')
from reinvent_plugins.components.pd1_pdl1_features import compute_features

SOL_MODEL  = 'Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj'
SOL_SCALER = 'Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl'

bst = xgb.Booster()
bst.load_model(SOL_MODEL)

in_csv  = 'Preprocess/Data_pd1_pdl1/data_csvs/pd1_pdl1_preprocess_all.csv'
out_csv = 'Preprocess/Data_pd1_pdl1/data_csvs/pd1_pdl1_preprocess_pic50_with_sol.csv'

df = pd.read_csv(in_csv)
print(f"Loaded {len(df)} rows from {in_csv}")

smiles = df['smiles'].tolist()
X, mask = compute_features(smiles, SOL_SCALER)
d = xgb.DMatrix(X)
preds = bst.predict(d)

df['logS'] = [float(preds[i]) if mask[i] else np.nan for i in range(len(smiles))]

df.to_csv(out_csv, index=False)
print(f"\nSaved enriched dataset to: {out_csv}")
print(f"Columns: {list(df.columns)}")
print(f"logS stats:\n{df['logS'].describe().round(3)}")
print(f"Valid logS predictions: {df['logS'].notna().sum()}/{len(df)}")
