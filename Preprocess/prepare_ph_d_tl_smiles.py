#!/usr/bin/env python3
"""
prepare_ph_d_tl_smiles.py
==========================
Reads filtered_smiles_tanimoto.csv, deduplicates SMILES, does an 80/20
random train/val split, and writes .smi files used for TL.
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(HERE, ".."))
IN_CSV   = os.path.join(HERE, "Data_pd1_pdl1", "filtered_smiles_tanimoto.csv")
DATA_DIR = os.path.join(ROOT, "data")
OUT_TRAIN = os.path.join(DATA_DIR, "ph_d_tl_train.smi")
OUT_VAL   = os.path.join(DATA_DIR, "ph_d_tl_val.smi")
SEED      = 42

os.makedirs(DATA_DIR, exist_ok=True)

# ── Load & canonicalise ───────────────────────────────────────────────────────
df = pd.read_csv(IN_CSV)
print(f"[*] Loaded {len(df)} rows from {IN_CSV}")

smiles_col = "SMILES"
if smiles_col not in df.columns:
    # try lowercase
    smiles_col = [c for c in df.columns if c.lower() == "smiles"][0]

raw_smiles = df[smiles_col].dropna().tolist()

canon = []
for smi in raw_smiles:
    mol = Chem.MolFromSmiles(str(smi).strip())
    if mol:
        canon.append(Chem.MolToSmiles(mol))

# Deduplicate
canon = list(dict.fromkeys(canon))
print(f"[+] {len(canon)} unique valid SMILES after canonicalisation")

# ── 80 / 20 split ─────────────────────────────────────────────────────────────
train_smi, val_smi = train_test_split(canon, test_size=0.20, random_state=SEED)
print(f"[+] Train: {len(train_smi)}  |  Val: {len(val_smi)}")

# ── Write .smi files ──────────────────────────────────────────────────────────
with open(OUT_TRAIN, "w") as f:
    for s in train_smi:
        f.write(s + "\n")

with open(OUT_VAL, "w") as f:
    for s in val_smi:
        f.write(s + "\n")

print(f"[+] Saved train → {OUT_TRAIN}")
print(f"[+] Saved val   → {OUT_VAL}")
