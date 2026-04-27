"""
Script 1: ChEMBL pIC50 Preprocessing + Zone-Stratified Scaffold Split + Visualizations
=======================================================================
Input  : ChEMBL raw CSV with columns [smiles, pic50] (Adjusted for custom dataset)
Output : train.csv, val.csv, test.csv  (all in OUTPUT_DIR)
         scaffold_split_report.txt
         dist_original.png, dist_splits_kde.png

Requirements:
    pip install pandas numpy rdkit scikit-learn matplotlib seaborn
    (SMOGN optional – only needed if USE_SMOGN=True)
"""

import os
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ─────────────────────────────────────────────
# CONFIG  –  edit these paths / thresholds
# ─────────────────────────────────────────────
# EXACT FILENAME OF YOUR DATASET
INPUT_FILE  = "data.csv" 
SMILES_COL  = "Smiles"           # Your dataset's smiles column
PIC50_COL   = "pChEMBL Value"    # Your dataset's target column
OUTPUT_DIR  = "data_splits"

LOW_THRESH  = 5.0     # pIC50 < LOW_THRESH  → low-potency extreme
HIGH_THRESH = 9.0     # pIC50 > HIGH_THRESH → high-potency extreme

TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
# test = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

RANDOM_SEED = 42
USE_SMOGN   = False   # set True to oversample low-potency tail in train set
# ─────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════
# STEP 1 – Load & basic sanity checks
# ══════════════════════════════════════════════
print("=" * 60)
print("STEP 1 – Loading data")
print("=" * 60)

# Added sep=";" because your dataset is semicolon-separated
df = pd.read_csv(INPUT_FILE, sep=";") 
print(f"  Raw rows         : {len(df):,}")

# Keep only required columns
df = df[[SMILES_COL, PIC50_COL]].copy()
df.rename(columns={SMILES_COL: "smiles", PIC50_COL: "pic50"}, inplace=True)

# Drop nulls
df.dropna(subset=["smiles", "pic50"], inplace=True)

# Coerce pic50 to float, drop non-numeric
df["pic50"] = pd.to_numeric(df["pic50"], errors="coerce")
df.dropna(subset=["pic50"], inplace=True)
print(f"  After null drop  : {len(df):,}")

# ══════════════════════════════════════════════
# STEP 2 – Validate SMILES & compute RDKit mol
# ══════════════════════════════════════════════
print("\nSTEP 2 – Validating SMILES")

def safe_mol(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return mol if mol is not None else None
    except Exception:
        return None

df["mol"] = df["smiles"].apply(safe_mol)
invalid = df["mol"].isna().sum()
print(f"  Invalid SMILES   : {invalid:,}")
df.dropna(subset=["mol"], inplace=True)
print(f"  Valid molecules  : {len(df):,}")

# ══════════════════════════════════════════════
# STEP 3 – pIC50 range filter & duplicate removal
# ══════════════════════════════════════════════
print("\nSTEP 3 – Filtering & deduplication")

# Physically unreasonable pIC50 values
df = df[(df["pic50"] >= 1.0) & (df["pic50"] <= 14.0)]
print(f"  After range filter [1,14] : {len(df):,}")

# Canonical SMILES for dedup
df["canon_smiles"] = df["mol"].apply(lambda m: Chem.MolToSmiles(m, canonical=True))
before_dedup = len(df)

# Keep mean pIC50 per canonical SMILES
df = (
    df.groupby("canon_smiles", as_index=False)
      .agg(pic50=("pic50", "mean"), smiles=("smiles", "first"))
)
# Recreate mol from canonical smiles
df["mol"] = df["canon_smiles"].apply(safe_mol)
df.dropna(subset=["mol"], inplace=True)
print(f"  Removed duplicates: {before_dedup - len(df):,}")
print(f"  Unique molecules  : {len(df):,}")

# ══════════════════════════════════════════════
# STEP 3.5 – Plot Original Distribution [NEW]
# ══════════════════════════════════════════════
print("\nSTEP 3.5 – Plotting Original Distribution")
plt.figure(figsize=(10, 6))
sns.histplot(df["pic50"], bins=40, kde=True, color='skyblue', edgecolor='black')
plt.axvline(LOW_THRESH, color='red', linestyle='--', linewidth=2, label=f'Low Threshold ({LOW_THRESH})')
plt.axvline(HIGH_THRESH, color='green', linestyle='--', linewidth=2, label=f'High Threshold ({HIGH_THRESH})')
plt.title('Distribution of pIC50 (Before Splitting)', fontsize=14)
plt.xlabel('pIC50 Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.savefig(os.path.join(OUTPUT_DIR, "dist_original.png"), bbox_inches='tight')
plt.close()
print(f"  → Saved original distribution plot to {OUTPUT_DIR}/dist_original.png")

# ══════════════════════════════════════════════
# STEP 4 – Morgan fingerprints + physicochemical features
# ══════════════════════════════════════════════
print("\nSTEP 4 – Feature engineering")

def compute_features(mol):
    # --- physicochemical ---
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
        "stereo"     : len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    }

    # --- ECFP4 (Morgan r=2, 2048 bits) ---
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_dict = {f"fp_{i}": int(b) for i, b in enumerate(fp.ToBitString())}

    return {**physchem, **fp_dict}

print("  Computing features (this may take ~1–2 min for 14k molecules)…")
feature_list = df["mol"].apply(compute_features).tolist()
feat_df = pd.DataFrame(feature_list)

# Concatenate with metadata
df = df.reset_index(drop=True)
df = pd.concat([df[["canon_smiles", "smiles", "pic50"]], feat_df], axis=1)

FEATURE_COLS = [c for c in df.columns if c not in ["canon_smiles", "smiles", "pic50", "mol"]]
print(f"  Feature columns  : {len(FEATURE_COLS):,}  ({len(FEATURE_COLS)-11} FP bits + 11 physicochemical)")

# ══════════════════════════════════════════════
# STEP 5 – Murcko scaffold assignment
# ══════════════════════════════════════════════
print("\nSTEP 5 – Murcko scaffold extraction")

def get_scaffold(mol):
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core, canonical=True)
    except Exception:
        return "__no_scaffold__"

df["mol"] = df["canon_smiles"].apply(safe_mol)
df["scaffold"] = df["mol"].apply(get_scaffold)

n_scaffolds = df["scaffold"].nunique()
print(f"  Unique scaffolds : {n_scaffolds:,}")

# ══════════════════════════════════════════════
# STEP 6 – Zone labelling
# ══════════════════════════════════════════════
print("\nSTEP 6 – Zone labelling")

def zone_label(p):
    if p < LOW_THRESH:
        return "low"
    elif p > HIGH_THRESH:
        return "high"
    else:
        return "bulk"

df["zone"] = df["pic50"].apply(zone_label)
zone_counts = df["zone"].value_counts()
print(f"  low  (< {LOW_THRESH}) : {zone_counts.get('low', 0):,}")
print(f"  bulk ({LOW_THRESH}–{HIGH_THRESH}): {zone_counts.get('bulk', 0):,}")
print(f"  high (> {HIGH_THRESH}) : {zone_counts.get('high', 0):,}")

# ══════════════════════════════════════════════
# STEP 7 – Zone-stratified scaffold split
# ══════════════════════════════════════════════
print("\nSTEP 7 – Zone-stratified scaffold split")

np.random.seed(RANDOM_SEED)

train_idx, val_idx, test_idx = [], [], []

for zone in ["low", "bulk", "high"]:
    zone_df = df[df["zone"] == zone].copy()
    scaffolds = zone_df["scaffold"].unique()
    np.random.shuffle(scaffolds)

    # Map scaffold → molecule indices
    scaf_to_idx = defaultdict(list)
    for i, row in zone_df.iterrows():
        scaf_to_idx[row["scaffold"]].append(i)

    n_scaf = len(scaffolds)
    n_train_scaf = int(np.floor(n_scaf * TRAIN_FRAC))
    n_val_scaf   = int(np.floor(n_scaf * VAL_FRAC))

    train_scaffolds = set(scaffolds[:n_train_scaf])
    val_scaffolds   = set(scaffolds[n_train_scaf : n_train_scaf + n_val_scaf])
    test_scaffolds  = set(scaffolds[n_train_scaf + n_val_scaf:])

    for scaf in train_scaffolds:
        train_idx.extend(scaf_to_idx[scaf])
    for scaf in val_scaffolds:
        val_idx.extend(scaf_to_idx[scaf])
    for scaf in test_scaffolds:
        test_idx.extend(scaf_to_idx[scaf])

    print(f"  [{zone:4s}] scaffolds: train={len(train_scaffolds):,}  "
          f"val={len(val_scaffolds):,}  test={len(test_scaffolds):,}")

train_df = df.loc[train_idx].copy()
val_df   = df.loc[val_idx].copy()
test_df  = df.loc[test_idx].copy()

print(f"\n  FINAL molecule counts:")
print(f"    Train : {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
print(f"    Val   : {len(val_df):,}   ({len(val_df)/len(df)*100:.1f}%)")
print(f"    Test  : {len(test_df):,}  ({len(test_df)/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════
# STEP 8 – Feature scaling (fit on train only)
# ══════════════════════════════════════════════
print("\nSTEP 8 – Scaling physicochemical features (fit on train)")

PHYSCHEM_COLS = ["mw","logp","hbd","hba","tpsa","rot_bonds",
                 "rings","arom_rings","heavy_atoms","frac_csp3","stereo"]

scaler = StandardScaler()
train_df[PHYSCHEM_COLS] = scaler.fit_transform(train_df[PHYSCHEM_COLS])
val_df[PHYSCHEM_COLS]   = scaler.transform(val_df[PHYSCHEM_COLS])
test_df[PHYSCHEM_COLS]  = scaler.transform(test_df[PHYSCHEM_COLS])

import pickle
with open(os.path.join(OUTPUT_DIR, "physchem_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("  Scaler saved → physchem_scaler.pkl")

# ══════════════════════════════════════════════
# STEP 9 – SMOGN oversampling (skipped by default)
# ══════════════════════════════════════════════
if USE_SMOGN:
    pass # (Your previous code remains valid here if you ever turn it on)

# ══════════════════════════════════════════════
# STEP 10 – Save splits
# ══════════════════════════════════════════════
print("\nSTEP 10 – Saving splits")

SAVE_COLS = ["canon_smiles", "smiles", "pic50", "zone", "scaffold"] + FEATURE_COLS

train_df[SAVE_COLS].to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df[SAVE_COLS].to_csv(os.path.join(OUTPUT_DIR, "val.csv"),   index=False)
test_df[SAVE_COLS].to_csv(os.path.join(OUTPUT_DIR, "test.csv"),  index=False)

# Also save feature column list
with open(os.path.join(OUTPUT_DIR, "feature_cols.txt"), "w") as f:
    f.write("\n".join(FEATURE_COLS))

# ══════════════════════════════════════════════
# STEP 10.5 – Plot Split Distributions [NEW]
# ══════════════════════════════════════════════
print("\nSTEP 10.5 – Plotting Split Distributions")
plt.figure(figsize=(10, 6))

# Plot Kernel Density Estimate (Smooth Curve) for each set
sns.kdeplot(train_df["pic50"], label=f"Train (n={len(train_df)})", fill=True, alpha=0.4, color="blue")
sns.kdeplot(val_df["pic50"], label=f"Validation (n={len(val_df)})", fill=True, alpha=0.4, color="orange")
sns.kdeplot(test_df["pic50"], label=f"Test (n={len(test_df)})", fill=True, alpha=0.4, color="green")

plt.axvline(LOW_THRESH, color='red', linestyle='--', alpha=0.5)
plt.axvline(HIGH_THRESH, color='green', linestyle='--', alpha=0.5)
plt.title('pIC50 Density Distribution (Train vs Val vs Test)', fontsize=14)
plt.xlabel('pIC50 Value', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "dist_splits_kde.png"), bbox_inches='tight')
plt.close()
print(f"  → Saved split distribution overlay plot to {OUTPUT_DIR}/dist_splits_kde.png")


# ══════════════════════════════════════════════
# STEP 11 – Split report
# ══════════════════════════════════════════════
report_lines = [
    "ChEMBL pIC50 Scaffold Split Report",
    "=" * 45,
    f"Total molecules     : {len(df):,}",
    f"Unique scaffolds    : {n_scaffolds:,}",
    "",
    "Split sizes:",
    f"  Train : {len(train_df):,}",
    f"  Val   : {len(val_df):,}",
    f"  Test  : {len(test_df):,}",
    ""
]

report_text = "\n".join(report_lines)
report_path = os.path.join(OUTPUT_DIR, "scaffold_split_report.txt")
with open(report_path, "w") as f:
    f.write(report_text)

print("\n" + "=" * 60)
print("Preprocessing complete. Plots and splits saved to 'data_splits' folder.")
print("=" * 60)