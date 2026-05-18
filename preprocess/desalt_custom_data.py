"""
desalt_custom_data.py
-----------------------------
Reads the raw ChEMBL JAK2 dataset (data.csv), applies full salt removal
and SMILES standardization, then:
  1. Writes a clean combined SMILES file: REINVENT4/custom_data/custom_desalted_all.smi
  2. Produces fresh train/val splits:   REINVENT4/custom_data/custom_desalted_train.smi
                                        REINVENT4/custom_data/custom_desalted_val.smi

Usage:
    conda activate reinvent-qsar
    cd /Users/vishnukasturi/Intern/reinvent-local
    python preprocess/desalt_custom_data.py
"""

import os
import csv
import sys
import random
import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover, MolStandardize, Descriptors, rdMolDescriptors

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_CSV      = "REINVENT4/custom_data/data.csv"          # raw source
OUT_ALL       = "REINVENT4/custom_data/custom_desalted_all.smi"
OUT_TRAIN     = "REINVENT4/custom_data/custom_desalted_train.smi"
OUT_VAL       = "REINVENT4/custom_data/custom_desalted_val.smi"
TRAIN_RATIO   = 0.80
RANDOM_SEED   = 42

# REINVENT prior model supported token regex (same as used in process_custom.py)
SMILES_TOKENS_REGEX = re.compile(
    r"(\%\d{2}|Br|Cl|@@|->|c|n|o|s|p|S|F|C|N|O|P|B|I|[se]|\[|\]|"
    r"\(|\)|=|#|\+|-|\\|/|\.|[0-9])"
)

# ─── Utilities ────────────────────────────────────────────────────────────────
remover    = SaltRemover.SaltRemover()        # RDKit built-in salt removal
normalizer = MolStandardize.rdMolStandardize.Normalizer()
uc_chooser = MolStandardize.rdMolStandardize.LargestFragmentChooser()
uncharger  = MolStandardize.rdMolStandardize.Uncharger()


def standardize_mol(mol):
    """Full standardization pipeline: normalize → desalt → uncharge → canonical."""
    try:
        mol = normalizer.normalize(mol)
        mol = uc_chooser.choose(mol)       # keep largest fragment (removes salts/counterions)
        mol = uncharger.uncharge(mol)      # neutralize charges where possible
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def is_token_valid(smi):
    """Check all tokens in SMILES are supported by REINVENT prior."""
    found = "".join(SMILES_TOKENS_REGEX.findall(smi))
    return found == smi


def is_drug_like(mol):
    """Loose Lipinski + TPSA filter to remove junk fragments."""
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    ha   = mol.GetNumHeavyAtoms()
    return (
        150 <= mw <= 800 and   # loose MW window
        logp <= 7.0 and
        tpsa <= 160 and
        hbd <= 10 and
        hba <= 15 and
        ha >= 10               # no tiny fragments
    )


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)

    if not os.path.exists(DATA_CSV):
        print(f"[ERROR] Raw data file not found: {DATA_CSV}")
        sys.exit(1)

    print(f"[*] Reading raw data: {DATA_CSV}")

    raw_smiles = []
    seen = set()

    with open(DATA_CSV, newline="", encoding="utf-8") as f:
        # Detect separator (semicolon for ChEMBL exports)
        sample = f.read(2048)
        f.seek(0)
        sep = ";" if sample.count(";") > sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=sep)

        for row in reader:
            # Try different possible column names
            smi = (
                row.get("Smiles") or
                row.get("SMILES") or
                row.get("smiles") or
                row.get("canonical_smiles") or
                row.get("canon_smiles") or ""
            ).strip().strip('"')

            if not smi or smi == "nan":
                continue
            raw_smiles.append(smi)

    print(f"[*] Raw SMILES loaded: {len(raw_smiles)}")

    # ─── Process ──────────────────────────────────────────────────────────────
    clean = []
    n_invalid  = 0
    n_salt     = 0
    n_token    = 0
    n_druglike = 0

    for smi in raw_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid += 1
            continue

        # Salt removal + standardization
        mol = standardize_mol(mol)
        if mol is None:
            n_salt += 1
            continue

        # Canonical SMILES
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if not canon:
            continue

        # Drug-likeness filter
        if not is_drug_like(mol):
            n_druglike += 1
            continue

        # Token compatibility with REINVENT prior
        if not is_token_valid(canon):
            n_token += 1
            continue

        # Deduplicate
        if canon in seen:
            continue
        seen.add(canon)
        clean.append(canon)

    print(f"\n[*] Results:")
    print(f"    Raw input:          {len(raw_smiles)}")
    print(f"    Invalid SMILES:     {n_invalid}")
    print(f"    Salt/fragment fail: {n_salt}")
    print(f"    Not drug-like:      {n_druglike}")
    print(f"    Bad REINVENT token: {n_token}")
    print(f"    ✅ Clean unique:     {len(clean)}")

    if not clean:
        print("[ERROR] No valid SMILES after processing!")
        sys.exit(1)

    # ─── Train / Val split ────────────────────────────────────────────────────
    random.shuffle(clean)
    split_idx = int(len(clean) * TRAIN_RATIO)
    train = clean[:split_idx]
    val   = clean[split_idx:]

    print(f"\n[*] Train / Val split ({int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}):")
    print(f"    Train: {len(train)}")
    print(f"    Val:   {len(val)}")

    # ─── Write outputs ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_ALL), exist_ok=True)

    for path, data in [(OUT_ALL, clean), (OUT_TRAIN, train), (OUT_VAL, val)]:
        with open(path, "w") as f:
            for s in data:
                f.write(s + "\n")
        print(f"    Saved: {path}  ({len(data)} SMILES)")

    print("\n✅ Done! Files written to REINVENT4/custom_data/")
    print("   To retrain with desalted data, update jak2_tl.toml:")
    print("     smiles_file            = \"custom_data/custom_desalted_train.smi\"")
    print("     validation_smiles_file = \"custom_data/custom_desalted_val.smi\"")


if __name__ == "__main__":
    main()
