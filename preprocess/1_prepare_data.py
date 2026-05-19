"""
1_prepare_data.py  —  Step 1 of the JAK2 (or any target) pipeline
===================================================================
Two modes:
  A) Fetch from ChEMBL by target name (default)
  B) Read a local raw CSV                 (--input_csv)

Outputs (saved to datasets/<TARGET>/):
  raw_bioactivity.csv   — full raw ChEMBL response (mode A only)
  processed.csv         — desalted, deduplicated, with pIC50 + conflict flags
  clean_smiles.smi      — all clean SMILES (no pIC50, for reference)
  train.smi             — 80% split for Transfer Learning
  val.smi               — 20% split for Transfer Learning validation
  summary.txt           — run statistics

Also copies train.smi / val.smi directly to:
  data/custom_train.smi
  data/custom_val.smi

Duplicate SMILES Strategy:
  Same canonical SMILES with different pIC50 values are resolved using the
  MEDIAN pIC50. SMILES with spread (max-min) > 1.5 log units are flagged
  as 'high_conflict' in processed.csv for manual inspection.

Usage:
  # From ChEMBL:
  python preprocess/1_prepare_data.py JAK2
  python preprocess/1_prepare_data.py EGFR --min_pic50 7.0 --max_mw 600

  # From local CSV:
  python preprocess/1_prepare_data.py JAK2 --input_csv /path/to/data.csv \\
         --smiles_col Smiles --pic50_col "pChEMBL Value"
"""

import sys
import os
import re
import time
import math
import json
import random
import shutil
import argparse
import requests
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, MolStandardize, SaltRemover
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ─── Constants ────────────────────────────────────────────────────────────────
CHEMBL_API   = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE    = 1000
TRAIN_RATIO  = 0.80
RANDOM_SEED  = 42

# REINVENT prior tokenizer — only these characters are valid
TOKEN_RE = re.compile(
    r"(\%\d{2}|Br|Cl|@@|->|c|n|o|s|p|S|F|C|N|O|P|B|I|[se]|\[|\]|"
    r"\(|\)|=|#|\+|-|\\|/|\.|[0-9])"
)

# Conflict threshold: flag SMILES where max-min pIC50 > this
CONFLICT_THRESHOLD = 1.5

# ─── RDKit standardization utils ──────────────────────────────────────────────
_normalizer = MolStandardize.rdMolStandardize.Normalizer()
_chooser    = MolStandardize.rdMolStandardize.LargestFragmentChooser()
_uncharger  = MolStandardize.rdMolStandardize.Uncharger()


def standardize(smi: str):
    """
    Full desalting + standardization pipeline:
      1. Parse SMILES
      2. Normalize functional groups (tautomers, nitro, etc.)
      3. Keep largest fragment (removes salts, counterions)
      4. Uncharge (neutralize where possible)
      5. Strip stereochemistry (REINVENT prior is flat)
      6. Canonical SMILES
    Returns canonical SMILES string or None if any step fails.
    """
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    try:
        mol = _normalizer.normalize(mol)
        mol = _chooser.choose(mol)       # desalting: keep largest fragment
        mol = _uncharger.uncharge(mol)
        Chem.RemoveStereochemistry(mol)  # REINVENT prior vocab is flat
        Chem.SanitizeMol(mol)
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return canon if canon else None
    except Exception:
        return None


def token_ok(smi: str) -> bool:
    """Check all characters in SMILES are supported by REINVENT prior."""
    return "".join(TOKEN_RE.findall(smi)) == smi


def drug_like(smi: str, min_mw=150, max_mw=800, max_logp=7,
              max_tpsa=160, min_ha=10) -> bool:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    ha   = mol.GetNumHeavyAtoms()
    return (min_mw <= mw <= max_mw and logp <= max_logp
            and tpsa <= max_tpsa and ha >= min_ha)


# ─── ChEMBL API helpers ───────────────────────────────────────────────────────
def chembl_get(endpoint: str, params: dict, retries=3):
    url = f"{CHEMBL_API}/{endpoint}.json"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                print(f"  Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def find_target(name: str, organism: str = "Homo sapiens"):
    data = chembl_get("target", {
        "pref_name__icontains": name,
        "target_type":          "SINGLE PROTEIN",
        "limit":                20,
        "format":               "json"
    })
    results = []
    for t in data.get("targets", []):
        chembl_id = t["target_chembl_id"]
        pref_name = t.get("pref_name", "")
        org       = t.get("organism", "")
        uniprot   = ""
        for comp in t.get("target_components", []):
            for xref in comp.get("target_component_xrefs", []):
                if xref.get("xref_src_db") == "UniProt":
                    uniprot = xref.get("xref_id", "")
                    break
            if uniprot:
                break
        results.append((chembl_id, pref_name, org, uniprot))

    human_hits = [c for c in results if organism.lower() in c[2].lower()]
    return human_hits if human_hits else results


def fetch_bioactivity(target_chembl_id: str,
                      assay_types=("IC50", "Ki", "Kd"),
                      min_pic50=5.0):
    all_rows = []
    for atype in assay_types:
        offset = 0
        print(f"  Fetching {atype} data...", end="", flush=True)
        while True:
            params = {
                "target_chembl_id":      target_chembl_id,
                "standard_type":         atype,
                "standard_relation":     "=",
                "standard_units":        "nM",
                "pchembl_value__isnull": "false",
                "pchembl_value__gte":    min_pic50,
                "limit":                 PAGE_SIZE,
                "offset":                offset,
                "format":                "json"
            }
            data       = chembl_get("activity", params)
            activities = data.get("activities", [])
            if not activities:
                break
            for a in activities:
                pchembl = a.get("pchembl_value")
                smi     = a.get("canonical_smiles")
                if pchembl and smi:
                    all_rows.append({
                        "molecule_chembl_id": a.get("molecule_chembl_id", ""),
                        "smiles":             smi,
                        "standard_type":      atype,
                        "standard_value":     a.get("standard_value"),
                        "standard_units":     a.get("standard_units", "nM"),
                        "pchembl_value":      float(pchembl),
                        "assay_chembl_id":    a.get("assay_chembl_id", ""),
                    })
            if len(activities) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            time.sleep(0.2)
        type_count = sum(1 for r in all_rows if r["standard_type"] == atype)
        print(f"  {atype}: {type_count} records")
    return pd.DataFrame(all_rows)


# ─── Core deduplication with conflict logging ─────────────────────────────────
def deduplicate_with_median(df: pd.DataFrame, smiles_col: str,
                            pic50_col: str) -> pd.DataFrame:
    """
    Groups rows by canonical SMILES and resolves duplicate pIC50 values
    using the MEDIAN. Also computes spread and flags high-conflict entries.

    Returns a DataFrame with one row per unique canonical SMILES.
    """
    print("\n[*] Deduplicating SMILES using median pIC50...")

    grouped = df.groupby(smiles_col)[pic50_col].agg(
        pic50_median="median",
        pic50_mean="mean",
        pic50_max="max",
        pic50_min="min",
        n_measurements="count"
    ).reset_index()

    grouped["pic50_spread"] = grouped["pic50_max"] - grouped["pic50_min"]
    grouped["high_conflict"] = grouped["pic50_spread"] > CONFLICT_THRESHOLD
    grouped = grouped.rename(columns={
        smiles_col:      "raw_smiles",
        "pic50_median":  "pic50"
    })

    n_multi   = (grouped["n_measurements"] > 1).sum()
    n_conflict = grouped["high_conflict"].sum()

    print(f"    Total unique SMILES:       {len(grouped):,}")
    print(f"    SMILES with >1 measurement:{n_multi:,}")
    print(f"    High-conflict (spread>{CONFLICT_THRESHOLD}): {n_conflict:,}")
    if n_conflict > 0:
        print("    [!] High-conflict SMILES flagged in processed.csv for review")

    return grouped


# ─── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fetch & preprocess pIC50 data for REINVENT4 pipeline"
    )
    parser.add_argument("target",
                        type=str, help="Target name, e.g. JAK2, EGFR")
    parser.add_argument("--min_pic50",   type=float, default=5.0)
    parser.add_argument("--max_mw",      type=float, default=800.0)
    parser.add_argument("--organism",    type=str,   default="Homo sapiens")
    parser.add_argument("--out_dir",     type=str,   default="datasets")
    parser.add_argument("--assay_types", type=str,   default="IC50,Ki,Kd")
    # Local CSV mode
    parser.add_argument("--input_csv",   type=str,   default=None,
                        help="Path to local raw CSV (skips ChEMBL fetch)")
    parser.add_argument("--smiles_col",  type=str,   default="Smiles")
    parser.add_argument("--pic50_col",   type=str,   default="pChEMBL Value")
    args = parser.parse_args()

    target_name = args.target.strip().upper()
    assay_types = [a.strip() for a in args.assay_types.split(",")]
    out_dir     = os.path.join(args.out_dir, target_name)
    data_dir    = "data"   # central data folder at repo root
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    random.seed(RANDOM_SEED)

    print(f"\n{'='*60}")
    print(f"  REINVENT4 Data Preparation")
    print(f"  Target:     {target_name}")
    print(f"  min_pIC50:  {args.min_pic50}  |  max_MW: {args.max_mw}")
    print(f"{'='*60}\n")

    # ── Step 1: Load raw data ─────────────────────────────────────────────────
    if args.input_csv:
        print(f"[1/5] Loading local CSV: {args.input_csv}")
        sample = open(args.input_csv).read(2048)
        sep    = ";" if sample.count(";") > sample.count(",") else ","
        raw_df = pd.read_csv(args.input_csv, sep=sep)
        raw_df = raw_df.rename(columns={
            args.smiles_col: "smiles",
            args.pic50_col:  "pchembl_value"
        })[["smiles", "pchembl_value"]].dropna()
        raw_df["pchembl_value"] = pd.to_numeric(
            raw_df["pchembl_value"], errors="coerce")
        raw_df = raw_df.dropna(subset=["pchembl_value"])
        raw_df = raw_df[raw_df["pchembl_value"] >= args.min_pic50]
        print(f"  Loaded {len(raw_df):,} rows from CSV")
    else:
        print(f"[1/5] Searching ChEMBL for: {target_name}...")
        candidates = find_target(target_name, args.organism)
        if not candidates:
            print(f"[ERROR] No targets found for '{target_name}'")
            sys.exit(1)
        chembl_id, pref_name, organism, uniprot = candidates[0]
        print(f"  Selected: {pref_name}  ({chembl_id})")
        if len(candidates) > 1:
            for c in candidates[1:4]:
                print(f"    Other: {c[0]}  {c[1]}  ({c[2]})")

        print(f"\n[2/5] Fetching bioactivity from ChEMBL...")
        raw_df = fetch_bioactivity(chembl_id, assay_types=assay_types,
                                   min_pic50=args.min_pic50)
        if raw_df.empty:
            print(f"[ERROR] No bioactivity data found for {chembl_id}")
            sys.exit(1)
        raw_df.to_csv(os.path.join(out_dir, "raw_bioactivity.csv"), index=False)
        print(f"  Raw records: {len(raw_df):,}")

    # ── Step 2: Standardize SMILES ─────────────────────────────────────────────
    print(f"\n[3/5] Standardizing SMILES (desalt, normalize, uncharge)...")
    raw_df["canon_smiles"] = raw_df["smiles"].apply(standardize)
    n_fail = raw_df["canon_smiles"].isna().sum()
    raw_df = raw_df.dropna(subset=["canon_smiles"])
    print(f"  Failed standardization: {n_fail:,}")
    print(f"  Valid after desalting:  {len(raw_df):,}")

    # ── Step 3: Deduplicate with median pIC50 ─────────────────────────────────
    dedup_df = deduplicate_with_median(raw_df, "canon_smiles", "pchembl_value")

    # ── Step 4: Drug-likeness + token filter ──────────────────────────────────
    print(f"\n[4/5] Applying drug-likeness and REINVENT token filters...")
    n_drug, n_token = 0, 0
    keep = []
    for _, row in dedup_df.iterrows():
        smi = row["raw_smiles"]
        if not drug_like(smi, max_mw=args.max_mw):
            n_drug += 1
            continue
        if not token_ok(smi):
            n_token += 1
            continue
        keep.append(row)

    clean_df = pd.DataFrame(keep).reset_index(drop=True)
    clean_df = clean_df.rename(columns={"raw_smiles": "smiles"})
    # Round pIC50 for cleanliness
    clean_df["pic50"] = clean_df["pic50"].round(3)
    print(f"  Not drug-like:          {n_drug:,}")
    print(f"  Bad REINVENT tokens:    {n_token:,}")
    print(f"  ✅ Final clean unique:   {len(clean_df):,}")

    if clean_df.empty:
        print("[ERROR] No valid SMILES after processing!")
        sys.exit(1)

    clean_df.sort_values("pic50", ascending=False).to_csv(
        os.path.join(out_dir, "processed.csv"), index=False
    )

    # ── Step 5: Train/val split and write .smi files ──────────────────────────
    print(f"\n[5/5] Writing train/val SMILES splits...")
    smiles_list = clean_df["smiles"].tolist()
    random.shuffle(smiles_list)
    split_idx = int(len(smiles_list) * TRAIN_RATIO)
    train = smiles_list[:split_idx]
    val   = smiles_list[split_idx:]

    def write_smi(path, smiles):
        with open(path, "w") as f:
            for s in smiles:
                f.write(s + "\n")
        print(f"  Saved {len(smiles):>6,} SMILES → {path}")

    write_smi(os.path.join(out_dir, "clean_smiles.smi"), smiles_list)
    write_smi(os.path.join(out_dir, "train.smi"),        train)
    write_smi(os.path.join(out_dir, "val.smi"),          val)

    # Copy into central data/ folder (for TOML configs)
    shutil.copy(os.path.join(out_dir, "train.smi"),
                os.path.join(data_dir, "custom_train.smi"))
    shutil.copy(os.path.join(out_dir, "val.smi"),
                os.path.join(data_dir, "custom_val.smi"))
    print(f"  Copied → data/custom_train.smi + data/custom_val.smi")

    # ── Summary ────────────────────────────────────────────────────────────────
    pic50_vals    = clean_df["pic50"].values
    n_conflict    = clean_df["high_conflict"].sum() if "high_conflict" in clean_df.columns else 0
    target_label  = target_name if not args.input_csv else f"{target_name} (local CSV)"

    summary = f"""
REINVENT4 Data Preparation Summary
====================================
Target:           {target_label}
min_pIC50:        {args.min_pic50}
max_MW:           {args.max_mw}

Raw records:      {len(raw_df) + n_fail:,}
After desalting:  {len(raw_df):,}
Unique (deduped): {len(dedup_df):,}
  Not drug-like:  {n_drug:,}
  Bad tokens:     {n_token:,}
Final clean:      {len(clean_df):,}
High-conflict:    {int(n_conflict):,}  (spread > {CONFLICT_THRESHOLD} log units)

pIC50 stats:
  Min:    {pic50_vals.min():.2f}
  Max:    {pic50_vals.max():.2f}
  Mean:   {pic50_vals.mean():.2f}
  Median: {float(np.median(pic50_vals)):.2f}

Train: {len(train):,} SMILES
Val:   {len(val):,} SMILES

Output: {out_dir}/
"""
    print(summary)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary)

    print(f"✅ Done! All files saved to: {out_dir}/")


if __name__ == "__main__":
    main()
