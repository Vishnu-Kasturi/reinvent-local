#!/usr/bin/env python3
"""
generic_molecular_analysis.py
=============================
A fully configurable and reusable script to analyze molecular datasets from
Transfer Learning (TL), Reinforcement Learning (RL), and Mol2Mol optimization runs.

Features (all individually toggleable):
1. KDE Plots: Property distributions (pIC50, logS/Solubility, SAScore, QED).
2. Tanimoto Plots: Pairwise or max Tanimoto similarity distributions against a reference dataset.
3. RDKit PNG grids: Visual structure grids with property text.
4. Butina Clustering: Groups compounds by fingerprint similarity and draws representatives of the top clusters.
5. Mol2Mol-specific shifts: Dumbbell plots showing 1-to-1 property shifts and lead-vs-analogue structure grids.

How to customize:
Modify the CONFIGURATION SETTINGS block below to set your paths, toggles, and column names.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, QED, rdMolDescriptors
from rdkit.ML.Cluster import Butina
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Try to load SA Score scorer
try:
    from rdkit import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
except Exception:
    sascorer = None

# ==============================================================================
# CONFIGURATION SETTINGS
# Change paths, toggles, and column names here to customize execution.
# ==============================================================================

# --- 1. Mode Selection ---
# Set the mode of analysis:
# - "TL"      : Transfer Learning checkpoints/outputs.
# - "RL"      : Reinforcement Learning agent outputs.
# - "MOL2MOL" : Mol2Mol optimization pairs/results.
MODE = "MOL2MOL"  # Options: "TL", "RL", "MOL2MOL"

# --- 2. Module Toggles ---
# Set to True or False to enable/disable specific analyses.
RUN_KDE_PLOTS         = True   # Plot density distributions for properties (pIC50, solubility, etc.)
RUN_TANIMOTO_PLOTS    = True   # Plot Tanimoto similarity histograms vs. Reference Dataset
RUN_RDKIT_PNG_GRID    = True   # Draw a 2D structure grid of top compounds
RUN_BUTINA_CLUSTERING = True   # Run Butina clustering and draw representative centroids
RUN_MOL2MOL_SHIFTS    = True   # (MOL2MOL ONLY) Plot 1-to-1 shifts (Dumbbell plots) and side-by-side grids

# --- 3. File Paths ---
# Main dataset containing generated / optimized / sampled molecules (CSV)
GENERATED_CSV_PATH = "/Users/vishnukasturi/Intern/reinvent-local/results/pd1_pdl1_mol2mol_1to1_optimized.csv"

# Reference dataset (e.g., training set SMILES for TL/RL, or original leads for Mol2Mol)
REFERENCE_CSV_PATH = "/Users/vishnukasturi/Intern/reinvent-local/Preprocess/Data_pd1_pdl1/data_csvs/TL_scafold_data.csv"

# Output folder for saving plots and structures
OUTPUT_DIR = "/Users/vishnukasturi/Intern/reinvent-local/results/generic_analysis"

# --- 4. Main Generated Columns Config ---
GEN_SMILES_COL = "Optimized_SMILES"      # SMILES column name in GENERATED_CSV_PATH
GEN_PIC50_COL  = "Optimized_pIC50"       # pIC50 column (Use None to disable or if not present)
GEN_SOL_COL    = "Optimized_Solubility"  # Solubility/logS column (Use None to disable or if not present)
GEN_SA_COL     = None                    # SAScore column (Use None to disable or if not present)
GEN_QED_COL    = None                    # QED column (Use None to disable or if not present)

# --- 5. Reference Columns Config ---
REF_SMILES_COL = "smiles"                # SMILES column name in REFERENCE_CSV_PATH
REF_PIC50_COL  = "pic50"                 # pIC50 column in REFERENCE_CSV_PATH (Use None if not present)
REF_SOL_COL    = "logS"                  # Solubility column in REFERENCE_CSV_PATH (Use None if not present)

# --- 6. Mol2Mol-Specific Columns (Only active if MODE == "MOL2MOL") ---
MOL2MOL_ID_COL          = "mol_id"       # Compound ID column
MOL2MOL_LEAD_SMILES_COL = "Parent_SMILES"  # Starting parent lead SMILES
MOL2MOL_LEAD_PIC50_COL  = "Parent_pIC50"   # Starting lead pIC50
MOL2MOL_LEAD_SOL_COL    = "Parent_Solubility" # Starting lead solubility

# --- 7. Plot/Analysis Parameters ---
TARGET_PIC50  = 8.5
TARGET_SOL    = -3.0
BUTINA_CUTOFF = 0.40  # Clustering distance cutoff (0.40 = ~60% Tanimoto similarity threshold)
MAX_TOP_PNGS  = 12    # Number of top molecules to draw in the standard structure grid
CLUSTERS_TO_DRAW = 12 # Number of cluster centroids to draw in the clustering grid

# ==============================================================================
# ANALYSIS PIPELINE IMPLEMENTATION
# ==============================================================================

def canonicalize(smi):
    """Sanitizes and canonicalizes a SMILES string."""
    if not smi or pd.isna(smi):
        return None
    try:
        m = Chem.MolFromSmiles(str(smi).strip())
        if m:
            return Chem.MolToSmiles(m)
    except:
        pass
    return None

def compute_sa_score(mol):
    """Computes SA score using RDKit's Contrib sascorer if available."""
    if sascorer and mol:
        try:
            return sascorer.calculateScore(mol)
        except:
            pass
    return np.nan

def compute_qed_score(mol):
    """Computes QED drug-likeness score."""
    if mol:
        try:
            return QED.qed(mol)
        except:
            pass
    return np.nan

def load_data():
    """Loads and standardizes input datasets."""
    print("\n" + "="*50)
    print(" 1. LOADING DATASETS")
    print("="*50)
    
    if not os.path.exists(GENERATED_CSV_PATH):
        raise FileNotFoundError(f"Generated CSV not found at: {GENERATED_CSV_PATH}")
    
    df_gen = pd.read_csv(GENERATED_CSV_PATH)
    print(f"Loaded {len(df_gen)} rows from Generated/Sampled file.")
    
    # Standardize generated SMILES
    df_gen['canonical_smi'] = df_gen[GEN_SMILES_COL].apply(canonicalize)
    df_gen = df_gen.dropna(subset=['canonical_smi']).reset_index(drop=True)
    print(f"  → Found {len(df_gen)} valid canonicalized molecules.")
    
    df_ref = None
    if os.path.exists(REFERENCE_CSV_PATH):
        df_ref = pd.read_csv(REFERENCE_CSV_PATH)
        print(f"Loaded {len(df_ref)} rows from Reference file.")
        df_ref['canonical_smi'] = df_ref[REF_SMILES_COL].apply(canonicalize)
        df_ref = df_ref.dropna(subset=['canonical_smi']).reset_index(drop=True)
        print(f"  → Found {len(df_ref)} valid canonicalized reference molecules.")
    else:
        print("[WARNING] Reference CSV path was not found or not specified. Skipping reference-dependent metrics.")
        
    return df_gen, df_ref

def run_kde(df_gen, df_ref):
    """Generates KDE density plots comparing generated and reference properties."""
    print("\n" + "="*50)
    print(" 2. GENERATING PROPERTY KDE PLOTS")
    print("="*50)
    
    sns.set_theme(style="whitegrid")
    
    # Determine active columns
    props_to_plot = []
    if GEN_PIC50_COL and GEN_PIC50_COL in df_gen.columns:
        props_to_plot.append(('pIC50', GEN_PIC50_COL, REF_PIC50_COL, TARGET_PIC50, "#2563eb")) # Sleek Blue
    if GEN_SOL_COL and GEN_SOL_COL in df_gen.columns:
        props_to_plot.append(('logS (Solubility)', GEN_SOL_COL, REF_SOL_COL, TARGET_SOL, "#16a34a")) # Green
    if GEN_SA_COL and GEN_SA_COL in df_gen.columns:
        props_to_plot.append(('SAScore', GEN_SA_COL, None, 4.0, "#dc2626")) # Red
    if GEN_QED_COL and GEN_QED_COL in df_gen.columns:
        props_to_plot.append(('QED', GEN_QED_COL, None, 0.5, "#9333ea")) # Purple

    if not props_to_plot:
        print("[!] No valid property columns configured for KDE plots. Skipping.")
        return

    n_plots = len(props_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]
        
    fig.suptitle(f"Molecular Property Distributions ({MODE} Analysis)", fontsize=14, weight='bold', y=0.98)
    
    for idx, (label, gen_col, ref_col, threshold, color) in enumerate(props_to_plot):
        ax = axes[idx]
        
        # Plot reference if exists
        if df_ref is not None and ref_col and ref_col in df_ref.columns:
            sns.kdeplot(df_ref[ref_col].dropna(), label="Reference Set", color="#7f8c8d", ls="--", lw=2, ax=ax, fill=True, alpha=0.1)
            
        # Plot generated
        sns.kdeplot(df_gen[gen_col].dropna(), label="Generated Pool", color=color, lw=2.5, ax=ax, fill=True, alpha=0.15)
        
        if threshold is not None:
            ax.axvline(threshold, color="#f1c40f", ls=":", lw=1.8, label=f"Threshold: {threshold}")
            
        ax.set_title(f"{label} Distribution", weight='bold', fontsize=11)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"{MODE.lower()}_property_kde.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved KDE plots to: {plot_path}")

def run_tanimoto(df_gen, df_ref):
    """Computes and plots Tanimoto similarities of generated molecules to the reference dataset."""
    print("\n" + "="*50)
    print(" 3. RUNNING TANIMOTO SIMILARITY AUDIT")
    print("="*50)
    
    if df_ref is None:
        print("[!] Reference dataset not loaded. Cannot run Tanimoto audit.")
        return
        
    print("[*] Generating fingerprints...")
    gen_mols = [Chem.MolFromSmiles(s) for s in df_gen['canonical_smi']]
    ref_mols = [Chem.MolFromSmiles(s) for s in df_ref['canonical_smi']]
    
    gen_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in gen_mols if m]
    ref_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in ref_mols if m]
    
    if not gen_fps or not ref_fps:
        print("[ERROR] Failed to calculate fingerprints. Skipping.")
        return
        
    print("[*] Calculating max Tanimoto similarities vs reference set...")
    max_similarities = []
    for fp in gen_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        max_similarities.append(max(sims))
        
    # Plot Tanimoto distribution
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 4.5))
    
    sns.histplot(max_similarities, bins=30, color="#d97706", alpha=0.8, edgecolor="white", kde=True)
    mean_sim = np.mean(max_similarities)
    med_sim = np.median(max_similarities)
    
    plt.axvline(mean_sim, color="#2c3e50", ls="--", lw=1.5, label=f"Mean: {mean_sim:.3f}")
    plt.axvline(med_sim, color="#555555", ls=":", lw=1.5, label=f"Median: {med_sim:.3f}")
    
    plt.title(f"Max Tanimoto Similarity of Generated Pool vs Reference Set", weight='bold', fontsize=12)
    plt.xlabel("Tanimoto Similarity")
    plt.ylabel("Frequency")
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"{MODE.lower()}_tanimoto_distribution.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[+] Saved Tanimoto distribution plot to: {plot_path}")

def generate_rdkit_grid(df_gen):
    """Generates 2D RDKit structures grid for the top/first N compounds."""
    print("\n" + "="*50)
    print(" 4. GENERATING RDKIT GRID IMAGES")
    print("="*50)
    
    num_to_draw = min(MAX_TOP_PNGS, len(df_gen))
    mols = []
    legends = []
    
    for i in range(num_to_draw):
        row = df_gen.iloc[i]
        smi = row['canonical_smi']
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            
            # Format label
            legend_txt = f"Rank {i+1}\n"
            if GEN_PIC50_COL and GEN_PIC50_COL in row:
                legend_txt += f"pIC50: {row[GEN_PIC50_COL]:.2f}\n"
            if GEN_SOL_COL and GEN_SOL_COL in row:
                legend_txt += f"logS: {row[GEN_SOL_COL]:.2f}\n"
            if GEN_SA_COL and GEN_SA_COL in row:
                legend_txt += f"SA: {row[GEN_SA_COL]:.2f}"
            legends.append(legend_txt.strip())
            
    if not mols:
        print("[!] No valid molecules to render. Skipping.")
        return
        
    dopts = Draw.rdMolDraw2D.MolDrawOptions()
    dopts.legendFontSize = 26
    dopts.bondLineWidth = 2.0
    
    cols = 4 if num_to_draw >= 4 else num_to_draw
    rows = math.ceil(num_to_draw / cols)
    
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=cols,
        subImgSize=(350, 320),
        legends=legends,
        useSVG=False,
        drawOptions=dopts
    )
    
    plot_path = os.path.join(OUTPUT_DIR, f"{MODE.lower()}_compounds_grid.png")
    img.save(plot_path)
    print(f"[+] Saved compound grid structures image to: {plot_path}")

def run_butina_clustering(df_gen):
    """Clusters generated molecules and outputs representative centroid structures."""
    print("\n" + "="*50)
    print(" 5. RUNNING BUTINA CLUSTERING ANALYSIS")
    print("="*50)
    
    mols = [Chem.MolFromSmiles(s) for s in df_gen['canonical_smi']]
    valid_mols = [m for m in mols if m]
    valid_rows = [df_gen.iloc[i] for i, m in enumerate(mols) if m]
    num_mols = len(valid_mols)
    
    if num_mols < 3:
        print("[!] Too few molecules to cluster. Skipping clustering.")
        return
        
    print(f"Clustering {num_mols} molecules...")
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in valid_mols]
    
    # Pairwise distances
    dists = []
    for i in range(1, num_mols):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        for s in sims:
            dists.append(1.0 - s)
            
    # Perform Butina clustering
    clusters = Butina.ClusterData(dists, num_mols, BUTINA_CUTOFF, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    
    print(f"[+] Clustering finished. Created {len(clusters)} clusters.")
    print(f"    Largest cluster contains {len(clusters[0])} compounds.")
    
    num_to_draw = min(CLUSTERS_TO_DRAW, len(clusters))
    draw_mols = []
    draw_legends = []
    
    for c_idx in range(num_to_draw):
        cluster = clusters[c_idx]
        centroid_idx = cluster[0]
        centroid_mol = valid_mols[centroid_idx]
        row = valid_rows[centroid_idx]
        
        legend_txt = f"Cluster {c_idx+1} (Size: {len(cluster)})\n"
        if GEN_PIC50_COL and GEN_PIC50_COL in row:
            legend_txt += f"pIC50: {row[GEN_PIC50_COL]:.2f}\n"
        if GEN_SOL_COL and GEN_SOL_COL in row:
            legend_txt += f"logS: {row[GEN_SOL_COL]:.2f}\n"
        draw_mols.append(centroid_mol)
        draw_legends.append(legend_txt.strip())
        
    dopts = Draw.rdMolDraw2D.MolDrawOptions()
    dopts.legendFontSize = 26
    dopts.bondLineWidth = 2.0
    
    cols = 4 if num_to_draw >= 4 else num_to_draw
    
    img = Draw.MolsToGridImage(
        draw_mols,
        molsPerRow=cols,
        subImgSize=(380, 350),
        legends=draw_legends,
        useSVG=False,
        drawOptions=dopts
    )
    
    plot_path = os.path.join(OUTPUT_DIR, f"{MODE.lower()}_cluster_centroids_grid.png")
    img.save(plot_path)
    print(f"[+] Saved cluster centroids grid image to: {plot_path}")

def run_mol2mol_comparison(df_gen):
    """Generates comparison plots and grids showing 1-to-1 shifts between starting leads and optimized analogues."""
    print("\n" + "="*50)
    print(" 6. MOL2MOL COMPARISON AUDIT (LEADS VS. OPTIMIZED)")
    print("="*50)
    
    # Check if necessary columns are present in df_gen
    required_cols = [MOL2MOL_LEAD_SMILES_COL, GEN_SMILES_COL]
    missing = [c for c in required_cols if c not in df_gen.columns]
    if missing:
        print(f"[ERROR] Missing columns for Mol2Mol comparison: {missing}. Skipping.")
        return
        
    df_shifts = df_gen.dropna(subset=[MOL2MOL_LEAD_SMILES_COL, GEN_SMILES_COL]).copy()
    
    if len(df_shifts) == 0:
        print("[!] No pairs available for shifts analysis. Skipping.")
        return
        
    # --- Dumbbell plots for property shifts ---
    # Determine which property shifts are available
    shifts_to_plot = []
    if MOL2MOL_LEAD_PIC50_COL in df_shifts.columns and GEN_PIC50_COL in df_shifts.columns:
        shifts_to_plot.append(('pIC50', MOL2MOL_LEAD_PIC50_COL, GEN_PIC50_COL, TARGET_PIC50))
    if MOL2MOL_LEAD_SOL_COL in df_shifts.columns and GEN_SOL_COL in df_shifts.columns:
        shifts_to_plot.append(('logS (Solubility)', MOL2MOL_LEAD_SOL_COL, GEN_SOL_COL, TARGET_SOL))
        
    if shifts_to_plot:
        # Reverse pairs for a clean top-to-bottom visualization
        df_plot = df_shifts.iloc[::-1].reset_index(drop=True)
        y_positions = np.arange(len(df_plot))
        mol_ids = df_plot[MOL2MOL_ID_COL] if MOL2MOL_ID_COL in df_plot.columns else [f"Lead {i+1}" for i in y_positions]
        
        n_plots = len(shifts_to_plot)
        fig, axes = plt.subplots(1, n_plots, figsize=(6.5 * n_plots, 7.5))
        if n_plots == 1:
            axes = [axes]
            
        fig.suptitle("Mol2Mol Optimization — Property Shift Audits (Parent vs. Optimized)", fontsize=14, weight='bold', y=0.98)
        
        for idx, (label, lead_col, opt_col, threshold) in enumerate(shifts_to_plot):
            ax = axes[idx]
            
            # Draw line connecting parents to optimized compounds
            ax.hlines(y=y_positions, xmin=df_plot[[lead_col, opt_col]].min(axis=1), xmax=df_plot[[lead_col, opt_col]].max(axis=1), color='#bdc3c7', alpha=0.8, lw=1.5)
            
            # Scatter plots for parent and optimized properties
            ax.scatter(df_plot[lead_col], y_positions, color="#3498db", s=80, label="Parent Lead", zorder=3)
            ax.scatter(df_plot[opt_col], y_positions, color="#e74c3c", s=80, label="Optimized Analogue", zorder=3)
            
            if threshold is not None:
                ax.axvline(threshold, color="#f1c40f", ls="--", lw=1.2, zorder=1)
                
            ax.set_yticks(y_positions)
            ax.set_yticklabels(mol_ids, weight='bold')
            ax.set_title(f"{label} Optimization Shifts", weight='bold', fontsize=11)
            ax.set_xlabel(label)
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.15)
            
        plt.tight_layout()
        dumbbell_path = os.path.join(OUTPUT_DIR, f"{MODE.lower()}_dumbbell_shifts.png")
        plt.savefig(dumbbell_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved shift dumbbell plot to: {dumbbell_path}")
        
    # --- RDKit Parent vs analogue structural grid ---
    pair_mols = []
    pair_legends = []
    
    # We display up to 6 pairs (12 molecules total)
    max_pairs_to_draw = min(6, len(df_shifts))
    for i in range(max_pairs_to_draw):
        row = df_shifts.iloc[i]
        lead_mol = Chem.MolFromSmiles(row[MOL2MOL_LEAD_SMILES_COL])
        opt_mol = Chem.MolFromSmiles(row[GEN_SMILES_COL])
        
        if lead_mol and opt_mol:
            AllChem.Compute2DCoords(lead_mol)
            AllChem.Compute2DCoords(opt_mol)
            pair_mols.extend([lead_mol, opt_mol])
            
            cid = row[MOL2MOL_ID_COL] if MOL2MOL_ID_COL in row else f"Pair {i+1}"
            
            # Lead Legend
            lead_txt = f"{cid} | Parent\n"
            if MOL2MOL_LEAD_PIC50_COL in row:
                lead_txt += f"pIC50: {row[MOL2MOL_LEAD_PIC50_COL]:.2f}\n"
            if MOL2MOL_LEAD_SOL_COL in row:
                lead_txt += f"logS: {row[MOL2MOL_LEAD_SOL_COL]:.2f}"
            pair_legends.append(lead_txt.strip())
            
            # Optimized Legend
            opt_txt = f"{cid} | Optimized\n"
            if GEN_PIC50_COL in row:
                opt_txt += f"pIC50: {row[GEN_PIC50_COL]:.2f}\n"
            if GEN_SOL_COL in row:
                opt_txt += f"logS: {row[GEN_SOL_COL]:.2f}"
            pair_legends.append(opt_txt.strip())
            
    if pair_mols:
        dopts = Draw.rdMolDraw2D.MolDrawOptions()
        dopts.legendFontSize = 28
        dopts.bondLineWidth = 2.2
        
        img = Draw.MolsToGridImage(
            pair_mols,
            molsPerRow=6, # 3 pairs per row (Lead, Opt, Lead, Opt, Lead, Opt)
            subImgSize=(400, 380),
            legends=pair_legends,
            useSVG=False,
            drawOptions=dopts
        )
        
        grid_path = os.path.join(OUTPUT_DIR, f"{MODE.lower()}_leads_vs_optimized_grid.png")
        img.save(grid_path)
        print(f"[+] Saved Mol2Mol comparison grid image to: {grid_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        df_gen, df_ref = load_data()
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load data: {e}")
        return
        
    # Run toggleable modules
    if RUN_KDE_PLOTS:
        run_kde(df_gen, df_ref)
    else:
        print("\n[skip] Skipping KDE plots as configured.")
        
    if RUN_TANIMOTO_PLOTS:
        run_tanimoto(df_gen, df_ref)
    else:
        print("\n[skip] Skipping Tanimoto distribution plots as configured.")
        
    if RUN_RDKIT_PNG_GRID:
        generate_rdkit_grid(df_gen)
    else:
        print("\n[skip] Skipping RDKit structure grids as configured.")
        
    if RUN_BUTINA_CLUSTERING:
        run_butina_clustering(df_gen)
    else:
        print("\n[skip] Skipping Butina clustering as configured.")
        
    if MODE == "MOL2MOL" and RUN_MOL2MOL_SHIFTS:
        run_mol2mol_comparison(df_gen)
    elif MODE != "MOL2MOL" and RUN_MOL2MOL_SHIFTS:
        print("\n[skip] Skipping Mol2Mol comparison (Only active in MOL2MOL mode).")
        
    print("\n" + "="*50)
    print(" GENERIC MOLECULAR ANALYSIS PIPELINE COMPLETED")
    print(f" Outputs saved to: {OUTPUT_DIR}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
