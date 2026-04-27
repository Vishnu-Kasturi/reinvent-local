"""
Script 2: PURE RANDOM SPLIT + Visualizations
=======================================================================
"""
import os
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_FILE  = "data.csv" 
SMILES_COL  = "Smiles"
PIC50_COL   = "pChEMBL Value"
OUTPUT_DIR  = "data_splits_random"  # <--- CHANGED DIRECTORY

LOW_THRESH  = 5.0
HIGH_THRESH = 9.0

RANDOM_SEED = 42
USE_SMOGN   = False
# ─────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1 – Loading data")
print("=" * 60)
df = pd.read_csv(INPUT_FILE, sep=";") 
df = df[[SMILES_COL, PIC50_COL]].copy()
df.rename(columns={SMILES_COL: "smiles", PIC50_COL: "pic50"}, inplace=True)
df.dropna(subset=["smiles", "pic50"], inplace=True)
df["pic50"] = pd.to_numeric(df["pic50"], errors="coerce")
df.dropna(subset=["pic50"], inplace=True)

print("\nSTEP 2 – Validating SMILES")
def safe_mol(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return mol if mol is not None else None
    except Exception:
        return None

df["mol"] = df["smiles"].apply(safe_mol)
df.dropna(subset=["mol"], inplace=True)

print("\nSTEP 3 – Filtering & deduplication")
df = df[(df["pic50"] >= 1.0) & (df["pic50"] <= 14.0)]
df["canon_smiles"] = df["mol"].apply(lambda m: Chem.MolToSmiles(m, canonical=True))
df = df.groupby("canon_smiles", as_index=False).agg(pic50=("pic50", "mean"), smiles=("smiles", "first"))
df["mol"] = df["canon_smiles"].apply(safe_mol)
df.dropna(subset=["mol"], inplace=True)

print("\nSTEP 3.5 – Plotting Original Distribution")
plt.figure(figsize=(10, 6))
sns.histplot(df["pic50"], bins=40, kde=True, color='skyblue', edgecolor='black')
plt.axvline(LOW_THRESH, color='red', linestyle='--', linewidth=2)
plt.axvline(HIGH_THRESH, color='green', linestyle='--', linewidth=2)
plt.title('Distribution of pIC50 (Before Splitting)', fontsize=14)
plt.xlabel('pIC50 Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig(os.path.join(OUTPUT_DIR, "dist_original.png"), bbox_inches='tight')
plt.close()

print("\nSTEP 4 – Feature engineering")
def compute_features(mol):
    physchem = {
        "mw"         : Descriptors.MolWt(mol),
        "logp"       : Descriptors.MolLogP(mol),
        "hbd"        : rdMolDescriptors.CalcNumHBD(mol),
        "hba"        : rdMolDescriptors.CalcNumHBA(mol),
        "tpsa"       : Descriptors.TPSA(mol),
        "rot_bonds"  : rdMolDescriptors.CalcNumRotatableBonds(mol),
        "rings"      : rdMolDescriptors.CalcNumRings(mol),
        "arom_rings" : rdMolDescriptors.CalcNumAromaticRings(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "frac_csp3"  : rdMolDescriptors.CalcFractionCSP3(mol),
        "stereo"     : len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)), # FIXED
    }
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_dict = {f"fp_{i}": int(b) for i, b in enumerate(fp.ToBitString())}
    return {**physchem, **fp_dict}

feature_list = df["mol"].apply(compute_features).tolist()
feat_df = pd.DataFrame(feature_list)
df = df.reset_index(drop=True)
df = pd.concat([df[["canon_smiles", "smiles", "pic50"]], feat_df], axis=1)
FEATURE_COLS = [c for c in df.columns if c not in ["canon_smiles", "smiles", "pic50", "mol"]]

print("\nSTEP 5 – Murcko scaffold extraction")
def get_scaffold(mol):
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core, canonical=True)
    except Exception:
        return "__no_scaffold__"

df["mol"] = df["canon_smiles"].apply(safe_mol)
df["scaffold"] = df["mol"].apply(get_scaffold)

print("\nSTEP 6 – Zone labelling")
def zone_label(p):
    if p < LOW_THRESH: return "low"
    elif p > HIGH_THRESH: return "high"
    else: return "bulk"
df["zone"] = df["pic50"].apply(zone_label)

# =====================================================================
print("\nSTEP 7 – PURE RANDOM SPLIT (70/15/15)")
# =====================================================================
# First split: 70% Train, 30% Temp (Val + Test)
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=RANDOM_SEED)
# Second split: Split the 30% Temp evenly into 15% Val and 15% Test
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=RANDOM_SEED)

print(f"  Train : {len(train_df):,}")
print(f"  Val   : {len(val_df):,}")
print(f"  Test  : {len(test_df):,}")

print("\nSTEP 8 – Scaling physicochemical features (fit on train)")
PHYSCHEM_COLS = ["mw","logp","hbd","hba","tpsa","rot_bonds","rings","arom_rings","heavy_atoms","frac_csp3","stereo"]
scaler = StandardScaler()
train_df[PHYSCHEM_COLS] = scaler.fit_transform(train_df[PHYSCHEM_COLS])
val_df[PHYSCHEM_COLS]   = scaler.transform(val_df[PHYSCHEM_COLS])
test_df[PHYSCHEM_COLS]  = scaler.transform(test_df[PHYSCHEM_COLS])

import pickle
with open(os.path.join(OUTPUT_DIR, "physchem_scaler.pkl"), "wb") as f: pickle.dump(scaler, f)

print("\nSTEP 10 – Saving splits")
SAVE_COLS = ["canon_smiles", "smiles", "pic50", "zone", "scaffold"] + FEATURE_COLS
train_df[SAVE_COLS].to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df[SAVE_COLS].to_csv(os.path.join(OUTPUT_DIR, "val.csv"),   index=False)
test_df[SAVE_COLS].to_csv(os.path.join(OUTPUT_DIR, "test.csv"),  index=False)
with open(os.path.join(OUTPUT_DIR, "feature_cols.txt"), "w") as f: f.write("\n".join(FEATURE_COLS))

print("\nSTEP 10.5 – Plotting Split Distributions")
plt.figure(figsize=(10, 6))
sns.kdeplot(train_df["pic50"], label=f"Train", fill=True, alpha=0.4, color="blue")
sns.kdeplot(val_df["pic50"], label=f"Validation", fill=True, alpha=0.4, color="orange")
sns.kdeplot(test_df["pic50"], label=f"Test", fill=True, alpha=0.4, color="green")
plt.axvline(LOW_THRESH, color='red', linestyle='--', alpha=0.5)
plt.axvline(HIGH_THRESH, color='green', linestyle='--', alpha=0.5)
plt.title('Random Split: pIC50 Density Distribution', fontsize=14)
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "dist_splits_kde.png"), bbox_inches='tight')
plt.close()

print("\nPreprocessing complete. Random splits saved.")