#!/usr/bin/env python3
"""
rl_auto_validate.py
===================
Waits for the RL output CSV to be written, then runs:
  1. Top hits extraction (pIC50 > 8.5)
  2. Tanimoto histogram (RL vs raw dataset)
  3. KDE per RL step window (pIC50 evolution)
  4. Overall KDE comparison (RL optimized vs original data)
"""
import os, sys, time, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import gaussian_kde
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
RDLogger.DisableLog('rdApp.*')

REPO     = os.path.dirname(os.path.abspath(__file__))
RAW_CSV  = os.path.join(REPO, "Preprocess/Data_pd1_pdl1/pd1_pdl1_pic50_raw.csv")
RES_DIR  = os.path.join(REPO, "results")
RL_CSV   = os.path.join(RES_DIR, "pd1_pdl1_rl_toml_1.csv")

PIC50_COL = "PD1PDL1pIC50_raw (raw)"
SOL_COL   = "PD1PDL1Sol_raw (raw)"
SA_COL    = "SAScore (raw)"

# ── helpers ──────────────────────────────────────────────────────────────────
def get_fps(smiles_list):
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            out.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    return out

def get_props(smiles_list):
    mw, qed_s, sa_s = [], [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            mw.append(Descriptors.MolWt(mol))
            qed_s.append(QED.qed(mol))
            try:    sa_s.append(sascorer.calculateScore(mol))
            except: sa_s.append(np.nan)
        else:
            mw.append(np.nan); qed_s.append(np.nan); sa_s.append(np.nan)
    return mw, qed_s, sa_s

# ── Wait for RL to finish writing ────────────────────────────────────────────
print("[*] Waiting for RL output CSV to be stable...")
last_size = -1
stable_count = 0
while True:
    if os.path.exists(RL_CSV):
        sz = os.path.getsize(RL_CSV)
        if sz == last_size:
            stable_count += 1
        else:
            stable_count = 0
        last_size = sz
        # stable for 10 seconds = done writing
        if stable_count >= 2:
            print(f"[+] RL CSV stable at {sz/1e6:.1f} MB. Proceeding...")
            break
    time.sleep(5)

time.sleep(5)  # extra buffer

# ── Load data ─────────────────────────────────────────────────────────────────
df_rl = pd.read_csv(RL_CSV).dropna(subset=[PIC50_COL])
max_step = df_rl["step"].max()
cutoff   = int(max_step * 0.8)
df_opt   = df_rl[df_rl["step"] > cutoff].copy()
df_opt["pic50"] = df_opt[PIC50_COL]
df_opt["logS"]  = df_opt[SOL_COL]
print(f"[*] RL total rows: {len(df_rl)} | Optimized (last 20%): {len(df_opt)}")

df_raw = pd.read_csv(RAW_CSV, sep="\t")
df_raw.columns = [c.strip().lower() for c in df_raw.columns]
raw_smiles   = df_raw["smiles"].dropna().tolist()
raw_pic50    = df_raw["pic50"].dropna().values

# ── 1. TOP HITS ───────────────────────────────────────────────────────────────
df_hits = df_opt[df_opt["pic50"] > 8.5].drop_duplicates("SMILES").sort_values("pic50", ascending=False)
print(f"[*] Hits pIC50 > 8.5: {len(df_hits)}")
top30 = df_hits.head(30)[["SMILES", "pic50", "logS", SA_COL, "Score"]].rename(columns={SA_COL: "SA"})
top30.to_csv(os.path.join(RES_DIR, "pd1_pdl1_e30_rl_top30.csv"), index=False)
print(top30.head(10).to_string(index=False))

# ── 2. TANIMOTO ───────────────────────────────────────────────────────────────
print("[*] Computing Tanimoto...")
raw_fps = get_fps(raw_smiles)
opt_fps = get_fps(df_opt["SMILES"].tolist())
max_tans, exact = [], 0
for fp in opt_fps:
    sims = DataStructs.BulkTanimotoSimilarity(fp, raw_fps)
    ms   = max(sims)
    max_tans.append(ms)
    if ms >= 0.999: exact += 1
mean_tan  = np.mean(max_tans)
med_tan   = np.median(max_tans)
copy_rate = exact / len(opt_fps)
print(f"    Mean={mean_tan:.3f} | Median={med_tan:.3f} | CopyRate={copy_rate:.1%}")

fig_tan, ax = plt.subplots(figsize=(9, 5))
ax.hist(max_tans, bins=35, color="#5b9bd5", alpha=0.78, edgecolor="white", lw=0.3)
ax.axvline(mean_tan, color="#111", ls="--", lw=1.8, label=f"Mean: {mean_tan:.3f}")
ax.axvline(med_tan,  color="#555", ls=":",  lw=1.8, label=f"Median: {med_tan:.3f}")
ax.axvspan(0.999, 1.02, color="red", alpha=0.12, label=f"Copies: {copy_rate:.1%}")
ax.set_title("PD1-PDL1 RL (Epoch 30) — Tanimoto vs. Raw Dataset", fontsize=12, weight="bold")
ax.set_xlabel("Max Tanimoto Similarity"); ax.set_ylabel("Count")
ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.03, 0.97, f"N={len(opt_fps)}\nMean={mean_tan:.3f}\nCopy={copy_rate:.1%}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.7, ec="gray"))
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "pd1_pdl1_e30_rl_tanimoto.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[+] Saved: pd1_pdl1_e30_rl_tanimoto.png")

# ── 3. KDE PER RL STEP WINDOW ────────────────────────────────────────────────
print("[*] Plotting pIC50 KDE per step window...")
n_bins     = 10
bin_edges  = np.linspace(1, max_step + 1, n_bins + 1)
bin_labels = [f"Steps {int(bin_edges[i])}-{int(bin_edges[i+1]-1)}" for i in range(n_bins)]
colors     = cm.plasma(np.linspace(0.05, 0.95, n_bins))
xs         = np.linspace(4, 11, 300)

fig_kde, axes = plt.subplots(2, 5, figsize=(22, 8), sharey=False)
fig_kde.suptitle("PD1-PDL1 RL (Epoch 30) — pIC50 KDE Evolution per Step Window",
                 fontsize=13, weight="bold", y=1.01)
for idx, ax in enumerate(axes.flatten()):
    lo = bin_edges[idx]; hi = bin_edges[idx + 1]
    vals = df_rl.loc[(df_rl["step"] >= lo) & (df_rl["step"] < hi), PIC50_COL].dropna().values
    kde_raw = gaussian_kde(raw_pic50)
    ax.fill_between(xs, kde_raw(xs), alpha=0.25, color="#2ecc71", label="Original")
    ax.plot(xs, kde_raw(xs), color="#2ecc71", lw=1.5)
    if len(vals) > 5:
        kde_rl = gaussian_kde(vals)
        ax.fill_between(xs, kde_rl(xs), alpha=0.5, color=colors[idx])
        ax.plot(xs, kde_rl(xs), color=colors[idx], lw=2, label=f"RL (n={len(vals)})")
    ax.axvline(8.5, color="gold", ls="--", lw=1.2)
    mean_v  = np.mean(vals) if len(vals) else 0
    pct_h   = (vals > 8.5).mean() * 100 if len(vals) else 0
    ax.set_title(bin_labels[idx], fontsize=9, weight="bold")
    ax.set_xlabel("pIC50", fontsize=8); ax.set_ylabel("Density", fontsize=8)
    ax.tick_params(labelsize=7); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6.5, loc="upper left"); ax.set_xlim(4, 11)
    ax.text(0.97, 0.97, f"Mean={mean_v:.2f}\n>8.5: {pct_h:.1f}%",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.5, ec="none"), color="white")
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "pd1_pdl1_e30_rl_step_kde.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[+] Saved: pd1_pdl1_e30_rl_step_kde.png")

# ── 4. OVERALL KDE (Optimized vs Original) ────────────────────────────────────
print("[*] Plotting overall KDE comparison...")
raw_mw, raw_qed, raw_sa = get_props(raw_smiles)
opt_mw, opt_qed, opt_sa = get_props(df_opt["SMILES"].tolist())

df_plot = pd.concat([
    pd.DataFrame({"MW": raw_mw, "QED": raw_qed, "SA": raw_sa,
                  "pic50": raw_pic50[:len(raw_mw)].tolist(), "Source": "Original Data"}),
    pd.DataFrame({"MW": opt_mw, "QED": opt_qed, "SA": opt_sa,
                  "pic50": df_opt["pic50"].tolist(), "Source": "RL Generated (E30)"})
], ignore_index=True)

pal = {"Original Data": "#2ecc71", "RL Generated (E30)": "#e74c3c"}
fig_ov, axes_ov = plt.subplots(1, 5, figsize=(22, 5))
fig_ov.suptitle("PD1-PDL1 RL (Epoch 30) — Generated vs. Original Property Distributions",
                fontsize=13, weight="bold")

for ax, col, title, vline in zip(
        axes_ov,
        ["pic50", "MW", "QED", "SA", "logS"],
        ["pIC50", "Molecular Weight", "QED Score", "SA Score", "Solubility (logS)"],
        [8.5, 500, 0.6, 4.0, -3.0]):
    if col == "logS":
        data_rl = df_opt["logS"].dropna()
        ax.hist(data_rl, bins=30, color="#e74c3c", alpha=0.6, density=True, edgecolor="white", lw=0.2, label="RL")
        try:
            kde = gaussian_kde(data_rl); xs2 = np.linspace(data_rl.min(), data_rl.max(), 200)
            ax.plot(xs2, kde(xs2), color="#e74c3c", lw=2)
        except: pass
    else:
        sns.kdeplot(data=df_plot, x=col, hue="Source", common_norm=False, fill=True,
                    alpha=0.35, ax=ax, palette=pal)
    ax.axvline(vline, color="gold", ls="--", lw=1.3)
    ax.set_title(title, weight="bold"); ax.set_xlabel(col)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "pd1_pdl1_e30_rl_kde_overall.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[+] Saved: pd1_pdl1_e30_rl_kde_overall.png")

print("\n============================")
print(" ALL VALIDATION PLOTS DONE!")
print("============================")
print(f"  Tanimoto  : results/pd1_pdl1_e30_rl_tanimoto.png")
print(f"  Step KDE  : results/pd1_pdl1_e30_rl_step_kde.png")
print(f"  Overall   : results/pd1_pdl1_e30_rl_kde_overall.png")
print(f"  Top30 CSV : results/pd1_pdl1_e30_rl_top30.csv")
