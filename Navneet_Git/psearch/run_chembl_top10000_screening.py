#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Run virtual screening of conformer database against pharmacophore models")
    parser.add_argument("-d", "--database", default="chembl_top10000_erg.dat", help="Path to conformer database (.dat)")
    parser.add_argument("-i", "--smi", default="chembl_top10000_erg.smi", help="Path to SMILES file (.smi)")
    parser.add_argument("-q", "--queries", default="psearch_models/models/", help="Path to query models directory")
    parser.add_argument("-o", "--output", default="screening_results_top10000", help="Directory to save raw screening results")
    parser.add_argument("-p", "--psearch_path", default="", help="Path to local psearch source directory if not installed in env")
    args = parser.parse_args()

    # Append local psearch path if provided
    if args.psearch_path:
        sys.path.append(os.path.abspath(args.psearch_path))

    try:
        from psearch.database import DB
        from psearch.screen_db import read_models, screen, save_results
    except ImportError:
        print("[ERROR] Could not import psearch. Make sure psearch is installed in your python environment or provide --psearch_path.")
        sys.exit(1)

    if not os.path.exists(args.database):
        print(f"[ERROR] Database file not found: {args.database}")
        sys.exit(1)
    if not os.path.exists(args.smi):
        print(f"[ERROR] SMILES file not found: {args.smi}")
        sys.exit(1)
    if not os.path.exists(args.queries):
        print(f"[ERROR] Queries directory not found: {args.queries}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    
    # Read compound names from the smi file
    df_smi = pd.read_csv(args.smi, sep='\t')
    smi_dict = dict(zip(df_smi['mol_name'], df_smi['smiles']))
    comp_names = df_smi['mol_name'].tolist()
    print(f"[+] Loaded {len(comp_names)} compound names from {args.smi}")
    
    # Run the custom screening loop
    db = DB(args.database, flag='r')
    bin_step = db.get_bin_step()
    
    # Get all query xyz/pma files
    queries = [os.path.join(args.queries, q) for q in os.listdir(args.queries) if q.endswith('.xyz') or q.endswith('.pma')]
    print(f"[+] Found {len(queries)} query models to screen: {[os.path.basename(q) for q in queries]}")
    
    models = read_models(queries, args.output, bin_step, min_features=None)
    
    # Clear output files first
    for model in models:
        if os.path.isfile(model.output_filename):
            os.remove(model.output_filename)
            
    print("[*] Screening compounds against models...")
    matched_count = 0
    
    for idx, comp_name in enumerate(comp_names, 1):
        try:
            res = screen(mol_name=comp_name, db=db, models=models, output_sdf=False, match_first_conf=True)
            if res:
                save_results(res, output_sdf=False, db=db)
                matched_count += 1
        except Exception as e:
            pass
            
    print(f"[+] Screening complete. Found matches for {matched_count} compounds.")
    print(f"[+] Raw results saved in {args.output}")
    
    # Map raw matches back to detailed CSVs for hits
    for q in os.listdir(args.output):
        if q.endswith('.txt'):
            path = os.path.join(args.output, q)
            model_name = q.replace(".txt", "")
            
            # Format output csv name
            hit_csv = f"hits_{model_name}_top10000.csv"
            
            hits = []
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path) as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            chembl_id = parts[0]
                            stereo_id = parts[1]
                            conf_id = parts[2]
                            
                            smiles = smi_dict.get(chembl_id, "")
                            hits.append({
                                'chembl_id': chembl_id,
                                'stereo_id': stereo_id,
                                'conf_id': conf_id,
                                'smiles': smiles
                            })
                            
            df_hits = pd.DataFrame(hits)
            if not df_hits.empty:
                df_hits.to_csv(hit_csv, index=False)
                print(f"[+] Saved {len(df_hits)} hits for {model_name} to {hit_csv}")
            else:
                pd.DataFrame(columns=['chembl_id','stereo_id','conf_id','smiles']).to_csv(hit_csv, index=False)
                print(f"[!] No hits found for {model_name}. Saved empty csv to {hit_csv}")

if __name__ == "__main__":
    main()
