#!/usr/bin/env python3
"""
fetch_chembl.py
===============
Given a protein target name (e.g., EGFR), queries UniProt to get the primary
accession ID, then queries the ChEMBL API for bioactivity data (IC50/Ki/Kd) 
for that target.

Usage:
  python fetch_chembl.py --target EGFR --out data/egfr_raw.csv
"""
import argparse
import os
import requests
import pandas as pd
from tqdm import tqdm
import math

def get_uniprot_id(protein_name):
    url = f"https://rest.uniprot.org/uniprotkb/search?query=protein_name:{protein_name} AND organism_id:9606&format=json&size=1"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        if data.get("results"):
            return data["results"][0]["primaryAccession"]
    return None

def fetch_chembl_data(uniprot_id):
    # 1. Get ChEMBL Target ID
    target_url = f"https://www.ebi.ac.uk/chembl/api/data/target?target_components.accession={uniprot_id}&format=json"
    res = requests.get(target_url).json()
    if not res.get("targets"):
        print(f"[!] No ChEMBL target found for UniProt {uniprot_id}")
        return pd.DataFrame()
    
    target_chembl_id = res["targets"][0]["target_chembl_id"]
    print(f"[*] Found ChEMBL Target ID: {target_chembl_id}")
    
    # 2. Fetch bioactivities
    url = (
        f"https://www.ebi.ac.uk/chembl/api/data/activity?target_chembl_id={target_chembl_id}"
        f"&standard_type__in=IC50,Ki,Kd&standard_relation__in==,<,<=&format=json&limit=1000"
    )
    
    activities = []
    while url:
        resp = requests.get(url).json()
        activities.extend(resp.get("activities", []))
        url = resp.get("page_meta", {}).get("next")
        if url:
            url = "https://www.ebi.ac.uk/chembl" + url
            
    if not activities:
        return pd.DataFrame()
        
    records = []
    for act in activities:
        val = act.get("standard_value")
        units = act.get("standard_units")
        smi = act.get("canonical_smiles")
        if val is not None and smi and units == "nM":
            try:
                v = float(val)
                if v > 0:
                    pic50 = -math.log10(v * 1e-9)
                    records.append({"smiles": smi, "pic50": pic50, "type": act.get("standard_type")})
            except:
                pass
                
    return pd.DataFrame(records).drop_duplicates(subset=["smiles"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Target protein name (e.g. EGFR)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()
    
    print(f"[*] Fetching data for {args.target}...")
    uniprot = get_uniprot_id(args.target)
    if not uniprot:
        print(f"[!] Could not resolve UniProt ID for {args.target}")
        return
        
    print(f"[*] UniProt ID: {uniprot}")
    df = fetch_chembl_data(uniprot)
    
    if df.empty:
        print("[!] No valid bioactivity data found.")
        return
        
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[*] Saved {len(df)} records to {args.out}")

if __name__ == "__main__":
    main()
