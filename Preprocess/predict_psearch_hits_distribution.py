#!/usr/bin/env python3
"""
predict_psearch_hits_distribution.py
======================================
1. Load all 16 PSearch hit CSVs from Preprocess/re/
2. Combine, deduplicate by SMILES
3. Predict pIC50 using the final PD1/PDL1 XGBoost model
4. Plot a rich distribution curve + summary stats
5. Save combined predictions to CSV

Usage:
    python Preprocess/predict_psearch_hits_distribution.py
"""

import os
import sys
import math
import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from glob import glob
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, QED, rdMolDescriptors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.join(HERE, "..")                      # reinvent-local/
CSV_DIR      = os.path.join(HERE, "re")
MODEL_PATH   = os.path.join(HERE, "final_acc", "pd1_pdl1_pic50_final_acc_model.ubj")
SCALER_PATH  = os.path.join(HERE, "final_acc", "pd1_pdl1_pic50_final_acc_scaler.pkl")
OUT_CSV      = os.path.join(HERE, "re", "combined_psearch_hits_pic50.csv")
OUT_PLOT     = os.path.join(HERE, "re", "psearch_hits_pic50_distribution.png")

# ── Feature extraction (identical to predict_pd1_properties.py) ───────────────
_RDKIT_EXCLUDE = {
    "Ipc", "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI",
    "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",  "BCUT2D_MRLOW"
}
_ALL_RDKIT      = [(n, f) for n, f in Descriptors.descList if n not in _RDKIT_EXCLUDE]
RDKIT_DESC_LIST = _ALL_RDKIT[:200]


def featurize(smiles: str) -> np.ndarray | None:
    """Return (1, 2415) feature vector or None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 200 RDKit descriptors
    rdk = []
    for _, fn in RDKIT_DESC_LIST:
        try:
            v = fn(mol)
            rdk.append(0.0 if (v is None or math.isnan(v) or math.isinf(v)) else float(v))
        except Exception:
            rdk.append(0.0)

    # 2048-bit ECFP4
    ecfp4 = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))

    # 167-bit MACCS
    maccs = list(MACCSkeys.GenMACCSKeys(mol))

    # Return 200 + 2048 + 167 = 2415 (physchem NOT included for pIC50 model)
    return np.array([rdk + ecfp4 + maccs], dtype=np.float32)


# ── Step 1: Load all 16 CSVs ───────────────────────────────────────────────────
print("\n" + "="*65)
print("  PSearch Hits — PD1/PDL1 pIC50 Distribution Analysis")
print("="*65)

csv_files = sorted(glob(os.path.join(CSV_DIR, "hits_psearch.*.csv")))
print(f"\n[1] Found {len(csv_files)} CSV files in {CSV_DIR}/")

dfs = []
for f in csv_files:
    tag = os.path.basename(f).replace("hits_psearch.", "").replace("_top10000.csv", "")
    df_i = pd.read_csv(f)
    df_i["source_file"] = tag
    dfs.append(df_i)
    print(f"    {tag:30s}  → {len(df_i):5d} rows")

df_all = pd.concat(dfs, ignore_index=True)
total_raw = len(df_all)
print(f"\n    Total rows (all files combined) : {total_raw:,}")

# ── Step 2: Deduplicate by canonical SMILES ───────────────────────────────────
print("\n[2] Deduplicating by canonical SMILES …")
df_all["canon_smiles"] = df_all["smiles"].apply(
    lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
)
df_all = df_all.dropna(subset=["canon_smiles"])
df_all = df_all.drop_duplicates(subset="canon_smiles").reset_index(drop=True)
total_unique = len(df_all)
print(f"    Unique molecules after dedup     : {total_unique:,}  (removed {total_raw - total_unique:,} duplicates)")

# ── Step 3: Load model ─────────────────────────────────────────────────────────
print("\n[3] Loading PD1/PDL1 pIC50 XGBoost model …")
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
print(f"    Model : {MODEL_PATH}")
print(f"    Scaler: {SCALER_PATH}")

# ── Step 4: Batch featurize & predict ─────────────────────────────────────────
print(f"\n[4] Featurizing and predicting {total_unique:,} molecules …")

pic50_preds = []
failed_idx  = []

for i, row in tqdm(df_all.iterrows(), total=total_unique, ncols=70, desc="    Predicting"):
    X = featurize(row["canon_smiles"])
    if X is None:
        pic50_preds.append(np.nan)
        failed_idx.append(i)
        continue
    X_scaled = X.copy()
    X_scaled[:, :200] = scaler.transform(X_scaled[:, :200])
    pic50_preds.append(float(model.predict(X_scaled)[0]))

df_all["pred_pIC50"] = pic50_preds
df_ok = df_all.dropna(subset=["pred_pIC50"]).reset_index(drop=True)

print(f"\n    Successfully predicted : {len(df_ok):,}")
print(f"    Featurization failures : {len(failed_idx)}")

# ── Step 5: Summary statistics ─────────────────────────────────────────────────
pic50 = df_ok["pred_pIC50"]
print("\n" + "─"*65)
print("  pIC50 Prediction Summary")
print("─"*65)
print(f"  Total molecules analysed  : {len(df_ok):,}")
print(f"  Mean pIC50                : {pic50.mean():.3f}")
print(f"  Median pIC50              : {pic50.median():.3f}")
print(f"  Std Dev                   : {pic50.std():.3f}")
print(f"  Min / Max                 : {pic50.min():.3f}  /  {pic50.max():.3f}")
print(f"  pIC50 ≥ 6 (moderate)      : {(pic50 >= 6).sum():,}  ({(pic50 >= 6).mean()*100:.1f}%)")
print(f"  pIC50 ≥ 7 (potent)        : {(pic50 >= 7).sum():,}  ({(pic50 >= 7).mean()*100:.1f}%)")
print(f"  pIC50 ≥ 8 (highly potent) : {(pic50 >= 8).sum():,}  ({(pic50 >= 8).mean()*100:.1f}%)")
print("─"*65)

# ── Step 6: Save CSV ───────────────────────────────────────────────────────────
df_ok.to_csv(OUT_CSV, index=False)
print(f"\n[5] Saved predictions to: {OUT_CSV}")

# ── Step 7: Rich distribution plot ───────────────────────────────────────────
print("\n[6] Generating distribution plot …")

# Dark premium style
plt.style.use("dark_background")
PURPLE = "#7c3aed"
CYAN   = "#06b6d4"
GREEN  = "#10b981"
AMBER  = "#f59e0b"
RED    = "#ef4444"
GRAY   = "#475569"

fig = plt.figure(figsize=(18, 11), facecolor="#0a0a0f")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34)

ax_main  = fig.add_subplot(gs[0, :])   # top-spanning KDE + hist
ax_box   = fig.add_subplot(gs[1, 0])   # box per query file
ax_violin= fig.add_subplot(gs[1, 1])   # violin overall
ax_ecdf  = fig.add_subplot(gs[1, 2])   # ECDF

common_bg = "#0f0f1a"
for ax in [ax_main, ax_box, ax_violin, ax_ecdf]:
    ax.set_facecolor(common_bg)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e1e30")

# ── Main: Histogram + KDE ─────────────────────────────────────────────────────
bins = np.linspace(pic50.min() - 0.2, pic50.max() + 0.2, 60)

ax_main.hist(pic50, bins=bins, color=PURPLE, alpha=0.35, density=True,
             linewidth=0, label="Histogram")

from scipy.stats import gaussian_kde
kde  = gaussian_kde(pic50, bw_method=0.25)
x_kd = np.linspace(pic50.min() - 0.5, pic50.max() + 0.5, 500)
y_kd = kde(x_kd)

ax_main.plot(x_kd, y_kd, color=CYAN, linewidth=2.5, label="KDE")
ax_main.fill_between(x_kd, y_kd, alpha=0.12, color=CYAN)

# Potency threshold shading
ax_main.axvspan(6, 7,  alpha=0.08, color=AMBER,  label="Moderate (6–7)")
ax_main.axvspan(7, 8,  alpha=0.10, color=GREEN,   label="Potent (7–8)")
ax_main.axvspan(8, 12, alpha=0.13, color=RED,     label="Highly potent (≥8)")

# Vertical lines
for val, col, lbl in [(pic50.mean(), PURPLE, f"Mean {pic50.mean():.2f}"),
                      (pic50.median(), CYAN, f"Median {pic50.median():.2f}")]:
    ax_main.axvline(val, color=col, linewidth=1.6, linestyle="--", alpha=0.85, label=lbl)

ax_main.set_xlabel("Predicted pIC50", fontsize=11)
ax_main.set_ylabel("Density", fontsize=11)
ax_main.set_title(
    f"PSearch Hits — PD1/PDL1 pIC50 Distribution  |  N = {len(df_ok):,} unique molecules",
    fontsize=14, fontweight="bold", color="#e2e8f0", pad=14
)
ax_main.legend(fontsize=8.5, framealpha=0.2, loc="upper right",
               labelcolor="#e2e8f0", facecolor="#16162a", edgecolor="#333")
ax_main.set_xlim(x_kd[0], x_kd[-1])

# Annotation box
stats_txt = (
    f"N = {len(df_ok):,}\n"
    f"μ = {pic50.mean():.3f}\n"
    f"σ = {pic50.std():.3f}\n"
    f"≥7: {(pic50>=7).mean()*100:.1f}%\n"
    f"≥8: {(pic50>=8).mean()*100:.1f}%"
)
ax_main.text(0.01, 0.95, stats_txt, transform=ax_main.transAxes,
             fontsize=8.5, va="top", ha="left", color="#e2e8f0",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#16162a",
                       edgecolor=PURPLE, alpha=0.8))

# ── Boxplot per source query ───────────────────────────────────────────────────
# Create short labels: t0_p0, t0_p1, ...
df_ok["query_label"] = df_ok["source_file"].str.replace(r"_f5", "", regex=True)
query_groups = {q: grp["pred_pIC50"].values for q, grp in df_ok.groupby("query_label")}
labels = sorted(query_groups.keys())
data_bp = [query_groups[l] for l in labels]

bp = ax_box.boxplot(data_bp, patch_artist=True, notch=False,
                    medianprops=dict(color=CYAN, linewidth=2),
                    whiskerprops=dict(color=GRAY),
                    capprops=dict(color=GRAY),
                    flierprops=dict(marker=".", markersize=2, color=GRAY, alpha=0.4))

colors_bp = plt.cm.plasma(np.linspace(0.2, 0.85, len(labels)))
for patch, col in zip(bp['boxes'], colors_bp):
    patch.set_facecolor((*col[:3], 0.45))
    patch.set_edgecolor("#444")

ax_box.set_xticks(range(1, len(labels) + 1))
ax_box.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.5)
ax_box.set_ylabel("Predicted pIC50", fontsize=9)
ax_box.set_title("pIC50 by Query Pharmacophore", fontsize=10,
                 color="#e2e8f0", fontweight="bold")
ax_box.axhline(7, color=GREEN, linewidth=1, linestyle=":", alpha=0.7)

# ── Violin plot overall ───────────────────────────────────────────────────────
parts = ax_violin.violinplot(pic50, positions=[1], showmedians=True,
                              showextrema=True, widths=0.7)
parts["cmedians"].set_color(CYAN)
parts["cmedians"].set_linewidth(2)
parts["cbars"].set_color(GRAY)
parts["cmins"].set_color(GRAY)
parts["cmaxes"].set_color(GRAY)
for pc in parts["bodies"]:
    pc.set_facecolor(PURPLE)
    pc.set_alpha(0.55)
    pc.set_edgecolor("#a78bfa")

ax_violin.axhline(6, color=AMBER, linewidth=1.2, linestyle="--", alpha=0.7, label="pIC50=6")
ax_violin.axhline(7, color=GREEN, linewidth=1.2, linestyle="--", alpha=0.7, label="pIC50=7")
ax_violin.axhline(8, color=RED,   linewidth=1.2, linestyle="--", alpha=0.7, label="pIC50=8")
ax_violin.set_xticks([1])
ax_violin.set_xticklabels(["All Hits"])
ax_violin.set_ylabel("Predicted pIC50", fontsize=9)
ax_violin.set_title("Overall Distribution (Violin)", fontsize=10,
                    color="#e2e8f0", fontweight="bold")
ax_violin.legend(fontsize=7.5, framealpha=0.2, loc="upper right",
                 labelcolor="#e2e8f0", facecolor="#16162a", edgecolor="#333")

# ── ECDF ─────────────────────────────────────────────────────────────────────
sorted_pic50 = np.sort(pic50)
cdf          = np.arange(1, len(sorted_pic50) + 1) / len(sorted_pic50)

ax_ecdf.plot(sorted_pic50, cdf, color=CYAN, linewidth=2.2)
ax_ecdf.fill_between(sorted_pic50, cdf, alpha=0.1, color=CYAN)

for thresh, col, lab in [(6, AMBER, "6"), (7, GREEN, "7"), (8, RED, "8")]:
    frac = (pic50 >= thresh).mean()
    ax_ecdf.axvline(thresh, color=col, linewidth=1.3, linestyle="--", alpha=0.8)
    ax_ecdf.text(thresh + 0.05, 0.04, f"≥{lab}\n{frac*100:.1f}%",
                 color=col, fontsize=7.5, va="bottom")

ax_ecdf.set_xlabel("Predicted pIC50", fontsize=9)
ax_ecdf.set_ylabel("Cumulative Fraction", fontsize=9)
ax_ecdf.set_title("Empirical CDF", fontsize=10, color="#e2e8f0", fontweight="bold")
ax_ecdf.set_ylim(0, 1.02)
ax_ecdf.set_xlim(sorted_pic50[0] - 0.1, sorted_pic50[-1] + 0.1)
ax_ecdf.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))

# ── Footer note ───────────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         f"PD1/PDL1 pIC50 model (XGBoost, test R²=0.859) · "
         f"Features: 200 RDKit + 2048 ECFP4 + 167 MACCS · {len(csv_files)} PSearch query files",
         ha="center", fontsize=8, color=GRAY)

plt.savefig(OUT_PLOT, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"    Saved plot to: {OUT_PLOT}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  FINAL SUMMARY")
print("="*65)
print(f"  CSV files merged          : {len(csv_files)}")
print(f"  Total rows (raw)          : {total_raw:,}")
print(f"  Unique molecules          : {total_unique:,}")
print(f"  Successfully predicted    : {len(df_ok):,}")
print(f"  pIC50 mean ± std          : {pic50.mean():.3f} ± {pic50.std():.3f}")
print(f"  Moderate hits (≥6)        : {(pic50>=6).sum():,}  ({(pic50>=6).mean()*100:.1f}%)")
print(f"  Potent hits (≥7)          : {(pic50>=7).sum():,}  ({(pic50>=7).mean()*100:.1f}%)")
print(f"  Highly potent hits (≥8)   : {(pic50>=8).sum():,}  ({(pic50>=8).mean()*100:.1f}%)")
print(f"\n  Output CSV  : {OUT_CSV}")
print(f"  Output Plot : {OUT_PLOT}")
print("="*65 + "\n")
