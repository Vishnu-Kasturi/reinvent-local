"""
fetch_chembl_target.py
-----------------------
Fetch pIC50-active molecules for ANY protein target from ChEMBL
using its name (e.g., JAK2, JAK1, DRD2, EGFR, BRAF).

Usage:
    conda activate reinvent-qsar
    cd /Users/vishnukasturi/Intern/reinvent-local

    python preprocess/fetch_chembl_target.py JAK2
    python preprocess/fetch_chembl_target.py DRD2
    python preprocess/fetch_chembl_target.py EGFR --min_pic50 7.0 --max_mw 600

Output (saved to datasets/<TARGET_NAME>/):
    raw_bioactivity.csv          - full raw ChEMBL response
    clean_smiles.smi             - REINVENT-compatible SMILES
    train.smi                    - 80% train split
    val.smi                      - 20% validation split
    summary.txt                  - run summary statistics
"""

import sys
import os
import re
import time
import math
import json
import random
import argparse
import requests
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, MolStandardize, SaltRemover

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

# ─── RDKit standardization utils ─────────────────────────────────────────────
_remover    = SaltRemover.SaltRemover()
_normalizer = MolStandardize.rdMolStandardize.Normalizer()
_chooser    = MolStandardize.rdMolStandardize.LargestFragmentChooser()
_uncharger  = MolStandardize.rdMolStandardize.Uncharger()


def standardize(smi: str):
    """Desalt, normalize, uncharge, canonicalize (flat SMILES). Returns canonical SMILES or None."""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    try:
        mol = _normalizer.normalize(mol)
        mol = _chooser.choose(mol)
        mol = _uncharger.uncharge(mol)
        # Strip stereochemistry to match flat REINVENT prior vocabulary
        Chem.RemoveStereochemistry(mol)
        Chem.SanitizeMol(mol)
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return canon if canon else None
    except Exception:
        return None


def token_ok(smi: str) -> bool:
    """Check all characters in SMILES are supported by REINVENT prior."""
    return "".join(TOKEN_RE.findall(smi)) == smi


def drug_like(smi: str, min_mw=150, max_mw=800, max_logp=7, max_tpsa=160, min_ha=10) -> bool:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    ha   = mol.GetNumHeavyAtoms()
    return min_mw <= mw <= max_mw and logp <= max_logp and tpsa <= max_tpsa and ha >= min_ha


# ─── ChEMBL API helpers ───────────────────────────────────────────────────────
def chembl_get(endpoint: str, params: dict, retries=3):
    """GET wrapper with retry and rate-limit handling."""
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
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def find_target(name: str):
    """
    Search ChEMBL for a single-protein target by name.
    Returns list of (chembl_id, preferred_name, organism, uniprot_id).
    """
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
        organism  = t.get("organism", "")

        # Extract UniProt accession from target_components
        uniprot = ""
        for comp in t.get("target_components", []):
            for xref in comp.get("target_component_xrefs", []):
                if xref.get("xref_src_db") == "UniProt":
                    uniprot = xref.get("xref_id", "")
                    break
            if uniprot:
                break

        results.append((chembl_id, pref_name, organism, uniprot))

    return results


def fetch_bioactivity(target_chembl_id: str, assay_types=("IC50", "Ki", "Kd"),
                      min_pic50=5.0):
    """
    Fetch all bioactivity records for a target.
    Returns a DataFrame with columns: smiles, pchembl_value, standard_type,
    standard_value, standard_units, assay_chembl_id, molecule_chembl_id.
    """
    all_rows = []

    for atype in assay_types:
        offset = 0
        print(f"  Fetching {atype} data...", end="", flush=True)

        while True:
            params = {
                "target_chembl_id":          target_chembl_id,
                "standard_type":             atype,
                "standard_relation":         "=",
                "standard_units":            "nM",
                "pchembl_value__isnull":     "false",
                "pchembl_value__gte":        min_pic50,
                "limit":                     PAGE_SIZE,
                "offset":                    offset,
                "format":                    "json"
            }
            data = chembl_get("activity", params)
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

            count = data.get("page_meta", {}).get("total_count", "?")
            print(f" {len(all_rows)} records (total={count})", end="\r", flush=True)

            if len(activities) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            time.sleep(0.2)   # polite rate limiting

        print(f"  {atype}: {sum(1 for r in all_rows if r['standard_type']==atype)} records")

    return pd.DataFrame(all_rows)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fetch & preprocess pIC50 bioactivity data from ChEMBL for any protein target"
    )
    parser.add_argument("target",         type=str,  help="Target name, e.g. JAK2, DRD2, EGFR")
    parser.add_argument("--min_pic50",    type=float, default=5.0,
                        help="Minimum pIC50 to include (default: 5.0 = IC50 ≤ 10 µM)")
    parser.add_argument("--max_mw",       type=float, default=800.0,
                        help="Maximum molecular weight (default: 800)")
    parser.add_argument("--organism",     type=str,  default="Homo sapiens",
                        help="Filter by organism (default: Homo sapiens)")
    parser.add_argument("--out_dir",      type=str,  default="datasets",
                        help="Root output directory (default: datasets/)")
    parser.add_argument("--assay_types",  type=str,  default="IC50,Ki,Kd",
                        help="Comma-separated assay types (default: IC50,Ki,Kd)")
    args = parser.parse_args()

    target_name = args.target.strip().upper()
    assay_types = [a.strip() for a in args.assay_types.split(",")]
    out_dir     = os.path.join(args.out_dir, target_name)
    os.makedirs(out_dir, exist_ok=True)

    random.seed(RANDOM_SEED)

    print(f"\n{'='*60}")
    print(f"  ChEMBL Bioactivity Fetcher")
    print(f"  Target: {target_name}")
    print(f"  min_pIC50: {args.min_pic50}  |  max_MW: {args.max_mw}")
    print(f"  Organism: {args.organism}")
    print(f"{'='*60}\n")

    # ── Step 1: Find target ───────────────────────────────────────────────────
    print(f"[1/5] Searching ChEMBL for target: {target_name}...")
    candidates = find_target(target_name)

    if not candidates:
        print(f"[ERROR] No SINGLE PROTEIN targets found for '{target_name}'")
        sys.exit(1)

    # Filter by organism if specified
    human_hits = [c for c in candidates if args.organism.lower() in c[2].lower()]
    if human_hits:
        candidates = human_hits

    # Pick the first (most relevant) hit
    chembl_id, pref_name, organism, uniprot = candidates[0]

    print(f"  Selected: {pref_name}")
    print(f"  ChEMBL ID: {chembl_id}")
    print(f"  Organism:  {organism}")
    print(f"  UniProt:   {uniprot if uniprot else 'N/A'}")

    if len(candidates) > 1:
        print(f"\n  Other matches:")
        for c in candidates[1:5]:
            print(f"    {c[0]}  {c[1]}  ({c[2]})")

    # ── Step 2: Fetch bioactivity ─────────────────────────────────────────────
    print(f"\n[2/5] Fetching bioactivity data from ChEMBL...")
    raw_df = fetch_bioactivity(chembl_id, assay_types=assay_types,
                               min_pic50=args.min_pic50)

    if raw_df.empty:
        print(f"[ERROR] No bioactivity data found for {chembl_id}")
        sys.exit(1)

    raw_path = os.path.join(out_dir, "raw_bioactivity.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"  Raw records: {len(raw_df)}  → saved to {raw_path}")

    # ── Step 3: Deduplicate (keep max pIC50 per SMILES) ───────────────────────
    print(f"\n[3/5] Deduplicating by canonical SMILES (keep max pIC50)...")
    raw_df = raw_df.sort_values("pchembl_value", ascending=False)
    raw_df = raw_df.drop_duplicates(subset="smiles", keep="first")
    print(f"  After dedup: {len(raw_df)} unique SMILES")

    # ── Step 4: Standardize & filter ─────────────────────────────────────────
    print(f"\n[4/5] Standardizing SMILES (desalt, normalize, filter)...")
    clean    = []
    n_fail   = 0
    n_token  = 0
    n_drug   = 0
    seen     = set()

    for _, row in raw_df.iterrows():
        canon = standardize(row["smiles"])
        if canon is None:
            n_fail += 1
            continue

        if not drug_like(canon, max_mw=args.max_mw):
            n_drug += 1
            continue

        if not token_ok(canon):
            n_token += 1
            continue

        if canon in seen:
            continue
        seen.add(canon)

        clean.append({
            "smiles":    canon,
            "pIC50":     round(row["pchembl_value"], 3),
            "assay_type": row["standard_type"],
        })

    print(f"  Failed standardization: {n_fail}")
    print(f"  Not drug-like:          {n_drug}")
    print(f"  Bad REINVENT tokens:    {n_token}")
    print(f"  ✅ Clean unique:         {len(clean)}")

    if not clean:
        print("[ERROR] No valid SMILES after processing!")
        sys.exit(1)

    clean_df = pd.DataFrame(clean).sort_values("pIC50", ascending=False)
    clean_df.to_csv(os.path.join(out_dir, "processed.csv"), index=False)

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
        print(f"  Saved {len(smiles):>5} SMILES → {path}")

    write_smi(os.path.join(out_dir, "clean_smiles.smi"), smiles_list)
    write_smi(os.path.join(out_dir, "train.smi"),        train)
    write_smi(os.path.join(out_dir, "val.smi"),          val)

    # ── Summary ───────────────────────────────────────────────────────────────
    pic50_vals = clean_df["pIC50"].values
    summary = f"""
ChEMBL Bioactivity Fetch Summary
==================================
Target name:      {target_name}
Preferred name:   {pref_name}
ChEMBL ID:        {chembl_id}
UniProt ID:       {uniprot if uniprot else 'N/A'}
Organism:         {organism}

Raw records:      {len(raw_df) + n_fail + n_token + n_drug}
After dedup:      {len(raw_df)}
Clean unique:     {len(clean)}

pIC50 stats:
  Min:  {pic50_vals.min():.2f}
  Max:  {pic50_vals.max():.2f}
  Mean: {pic50_vals.mean():.2f}
  Median: {float(np.median(pic50_vals)):.2f}

Train: {len(train)} SMILES
Val:   {len(val)} SMILES

Output directory: {out_dir}/
"""
    print(summary)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary)

    print(f"✅ Done! All files saved to: {out_dir}/")
    print(f"\n   To use for TL, update jak2_tl.toml:")
    print(f"     smiles_file            = \"../../datasets/{target_name}/train.smi\"")
    print(f"     validation_smiles_file = \"../../datasets/{target_name}/val.smi\"")


if __name__ == "__main__":
    main()
