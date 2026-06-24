#!/usr/bin/env python3
"""
evaluate_pdl1_ph_d_tl.py
=========================
Samples 500 molecules from every 10-epoch checkpoint of pdl1_ph_d_tl.model,
then for each epoch produces:
  1. Tanimoto histogram vs. the filtered_smiles_tanimoto training dataset.
  2. KDE plots for pIC50 (predicted) and logS (predicted) vs. the baseline.
Also saves a per-epoch tanimoto summary CSV.

Usage:
    conda run -n reinvent4 python Preprocess/evaluate_pdl1_ph_d_tl.py
"""

import os, sys, subprocess, math, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED
from tqdm import tqdm

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── SA Score ──────────────────────────────────────────────────────────────────
try:
    from rdkit import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    HAS_SA = True
except Exception:
    HAS_SA = False

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.abspath(os.path.join(HERE, ".."))
MODELS_DIR  = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
CONFIGS_DIR = os.path.join(ROOT, "REINVENT4", "configs")
REINVENT4   = os.path.join(ROOT, "REINVENT4")

# Reference dataset (training set used for this run)
REF_CSV    = os.path.join(HERE, "Data_pd1_pdl1", "filtered_smiles_tanimoto.csv")

# XGBoost models for property prediction
PIC50_MODEL  = os.path.join(HERE, "final_acc", "pd1_pdl1_pic50_final_acc_model.ubj")
PIC50_SCALER = os.path.join(HERE, "final_acc", "pd1_pdl1_pic50_final_acc_scaler.pkl")
SOL_MODEL    = os.path.join(HERE, "final_acc", "pd1_pdl1_sol_final_acc_model.ubj")
SOL_SCALER   = os.path.join(HERE, "final_acc", "pd1_pdl1_sol_final_acc_scaler.pkl")

N_SAMPLE    = 500
MODEL_STEM  = "pdl1_ph_d_tl"
EPOCHS      = list(range(10, 160, 10))  # 10, 20, ..., 150

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Feature extractor ─────────────────────────────────────────────────────────
from rdkit.Chem import Descriptors, MACCSkeys, rdMolDescriptors

_RDKIT_EXCLUDE = {"Ipc", "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI",
                  "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW",
                  "BCUT2D_MRHI", "BCUT2D_MRLOW"}
RDKIT_DESC_LIST = [(n, f) for n, f in Descriptors.descList if n not in _RDKIT_EXCLUDE][:200]

def featurize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    rdk = []
    for _, fn in RDKIT_DESC_LIST:
        try:
            v = fn(mol)
            rdk.append(0.0 if (v is None or math.isnan(v) or math.isinf(v)) else float(v))
        except:
            rdk.append(0.0)
    ecfp4 = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
    maccs  = list(MACCSkeys.GenMACCSKeys(mol))
    try: qed_v = QED.qed(mol)
    except: qed_v = 0.0
    phys = [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcFractionCSP3(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)), qed_v,
    ]
    return np.array([rdk + ecfp4 + maccs + phys], dtype=np.float32)

# ── Load QSAR models ──────────────────────────────────────────────────────────
print("[*] Loading QSAR models...")
m_pic50 = xgb.XGBRegressor(); m_pic50.load_model(PIC50_MODEL)
m_sol   = xgb.XGBRegressor(); m_sol.load_model(SOL_MODEL)
with open(PIC50_SCALER, "rb") as f: sc_pic50 = pickle.load(f)
with open(SOL_SCALER,  "rb") as f: sc_sol   = pickle.load(f)
cont_sol = list(range(200)) + list(range(200 + 2048 + 167, 2427))

def predict_batch(smiles_list):
    pic50s, logss = [], []
    for s in smiles_list:
        X = featurize(s)
        if X is None: continue
        Xp = X[:, :2415].copy(); Xp[:, :200] = sc_pic50.transform(Xp[:, :200])
        pic50s.append(float(m_pic50.predict(Xp)[0]))
        Xs = X.copy(); Xs[:, cont_sol] = sc_sol.transform(Xs[:, cont_sol])
        logss.append(float(m_sol.predict(Xs)[0]))
    return pic50s, logss

# ── Load reference dataset ────────────────────────────────────────────────────
print("[*] Loading reference dataset (filtered_smiles_tanimoto)...")
df_ref = pd.read_csv(REF_CSV)
ref_smiles = df_ref["SMILES"].dropna().tolist()
ref_mols   = [Chem.MolFromSmiles(str(s)) for s in ref_smiles]
ref_fps    = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in ref_mols if m]
print(f"  Reference FPs: {len(ref_fps)}")

# Baseline property distributions from reference (for KDE comparison)
ref_pic50, ref_lgs = predict_batch(ref_smiles[:300])  # subsample 300 for speed
print(f"  Reference baseline: {len(ref_pic50)} pIC50, {len(ref_lgs)} logS")

# ── Sampling helper ───────────────────────────────────────────────────────────
run_env = os.environ.copy()
run_env["PYTHONPATH"] = REINVENT4

def sample_checkpoint(chkpt_path, epoch):
    """Sample N_SAMPLE SMILES from a checkpoint, save & return CSV path."""
    out_csv = os.path.join(RESULTS_DIR, f"pdl1_ph_d_tl_sample_e{epoch}.csv")
    if os.path.exists(out_csv):
        print(f"  Epoch {epoch:3d}: existing CSV found, skipping sampling.")
        return out_csv

    toml_path = os.path.join(CONFIGS_DIR, f"_ph_d_tl_sample_e{epoch}.toml")
    toml_content = f"""run_type = "sampling"
device   = "mps"
json_out_config = "_ph_d_tl_sample_e{epoch}.json"

[parameters]
model_file       = "{chkpt_path}"
output_file      = "{out_csv}"
num_smiles       = {N_SAMPLE}
unique_molecules = true
sample_strategy  = "multinomial"
temperature      = 1.0
"""
    with open(toml_path, "w") as f:
        f.write(toml_content)

    subprocess.run(
        ["conda", "run", "-n", "reinvent4", "reinvent", toml_path],
        cwd=REINVENT4,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=run_env
    )
    return out_csv if os.path.exists(out_csv) else None

# ── Main evaluation loop ──────────────────────────────────────────────────────
summary_rows = []
all_tanimoto_data = {}   # epoch → list of max Tanimoto scores
all_prop_data    = {}    # epoch → {"pic50": [...], "lgs": [...]}

for epoch in EPOCHS:
    chkpt = os.path.join(MODELS_DIR, f"{MODEL_STEM}.model.{epoch}.chkpt")
    if not os.path.exists(chkpt):
        print(f"  [skip] Epoch {epoch}: checkpoint not found at {chkpt}")
        continue

    out_csv = sample_checkpoint(chkpt, epoch)
    if out_csv is None or not os.path.exists(out_csv):
        print(f"  [skip] Epoch {epoch}: no CSV produced")
        continue

    df_samp = pd.read_csv(out_csv).head(N_SAMPLE)
    smiles  = df_samp["SMILES"].dropna().tolist()
    gen_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
               for s in smiles if Chem.MolFromSmiles(s)]

    # Tanimoto vs reference
    max_tans = []
    for fp in gen_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        max_tans.append(max(sims))
    all_tanimoto_data[epoch] = max_tans

    mean_t  = np.mean(max_tans); med_t = np.median(max_tans)
    nov_85  = np.mean(np.array(max_tans) < 0.85) * 100
    nov_70  = np.mean(np.array(max_tans) < 0.70) * 100
    print(f"  Epoch {epoch:3d}: n={len(gen_fps)} | "
          f"Mean={mean_t:.3f} | Median={med_t:.3f} | "
          f"Novel<0.85={nov_85:.1f}% | Novel<0.70={nov_70:.1f}%")

    # Property prediction
    pic50s, logss = predict_batch(smiles)
    all_prop_data[epoch] = {"pic50": pic50s, "lgs": logss}

    summary_rows.append({
        "Epoch": epoch, "N_Generated": len(gen_fps),
        "Mean_MaxTanimoto": round(mean_t, 4), "Median_MaxTanimoto": round(med_t, 4),
        "Novelty_T085": round(nov_85, 2), "Novelty_T070": round(nov_70, 2),
        "Mean_pIC50": round(np.nanmean(pic50s), 4) if pic50s else None,
        "Mean_logS":  round(np.nanmean(logss), 4)  if logss  else None,
    })

# ── Save summary CSV ──────────────────────────────────────────────────────────
df_sum = pd.DataFrame(summary_rows)
sum_csv = os.path.join(RESULTS_DIR, "pdl1_ph_d_tl_epoch_tanimoto_summary.csv")
df_sum.to_csv(sum_csv, index=False)
print(f"\n[+] Summary CSV → {sum_csv}")
print(df_sum.to_string(index=False))

# ── Tanimoto histogram grid ───────────────────────────────────────────────────
print("\n[*] Plotting Tanimoto histograms...")
sns.set_theme(style="whitegrid")
n_epochs = len(all_tanimoto_data)
if n_epochs > 0:
    import matplotlib.cm as cm
    colors = cm.plasma(np.linspace(0.1, 0.9, n_epochs))
    ncols  = 5
    nrows  = math.ceil(n_epochs / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 4), sharex=True)
    fig.suptitle(
        "pdl1_ph_d TL — Max Tanimoto Histograms per Epoch vs. filtered_smiles_tanimoto baseline",
        fontsize=14, weight="bold", y=1.01
    )
    bins = np.linspace(0, 1, 35)
    for idx, (epoch, max_tans) in enumerate(sorted(all_tanimoto_data.items())):
        ax = axes.flatten()[idx]
        mean_t = np.mean(max_tans); nov_85 = np.mean(np.array(max_tans) < 0.85) * 100
        ax.hist(max_tans, bins=bins, color=colors[idx], edgecolor="white", lw=0.5, alpha=0.85)
        ax.axvline(mean_t, color="#2c3e50", ls="--", lw=1.5, label=f"Mean: {mean_t:.2f}")
        ax.axvline(0.85, color="red", ls=":", lw=1.2, label="T=0.85")
        ax.set_title(f"Epoch {epoch} (Novel<0.85={nov_85:.1f}%)", fontsize=10, weight="bold")
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=7, loc="upper left")
    for i in range(n_epochs, nrows * ncols):
        fig.delaxes(axes.flatten()[i])
    plt.tight_layout()
    tan_plot = os.path.join(RESULTS_DIR, "pdl1_ph_d_tl_epoch_tanimoto.png")
    plt.savefig(tan_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Tanimoto histogram grid → {tan_plot}")

# ── KDE property evolution plots ──────────────────────────────────────────────
print("[*] Plotting KDE property evolution...")
if len(all_prop_data) > 0:
    selected = sorted(all_prop_data.keys())
    colors_k = cm.viridis(np.linspace(0.15, 0.85, len(selected)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "pdl1_ph_d TL — pIC50 & logS KDE Evolution vs. filtered_smiles_tanimoto baseline",
        fontsize=13, weight="bold", y=1.01
    )

    # pIC50 KDE
    if ref_pic50:
        sns.kdeplot(ref_pic50, label="Reference baseline", color="gray", ls="--", lw=2, ax=axes[0])
    for idx, epoch in enumerate(selected):
        vals = all_prop_data[epoch]["pic50"]
        if vals:
            sns.kdeplot(vals, label=f"Epoch {epoch}", color=colors_k[idx], lw=1.5, ax=axes[0])
    axes[0].set_title("Predicted pIC50 Distribution", weight="bold")
    axes[0].set_xlabel("pIC50"); axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].spines[["top", "right"]].set_visible(False)

    # logS KDE
    if ref_lgs:
        sns.kdeplot(ref_lgs, label="Reference baseline", color="gray", ls="--", lw=2, ax=axes[1])
    for idx, epoch in enumerate(selected):
        vals = all_prop_data[epoch]["lgs"]
        if vals:
            sns.kdeplot(vals, label=f"Epoch {epoch}", color=colors_k[idx], lw=1.5, ax=axes[1])
    axes[1].set_title("Predicted logS (Solubility) Distribution", weight="bold")
    axes[1].set_xlabel("logS"); axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    kde_plot = os.path.join(RESULTS_DIR, "pdl1_ph_d_tl_epoch_kde_vs_baseline.png")
    plt.savefig(kde_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] KDE plot → {kde_plot}")

print("\n=== pdl1_ph_d TL Evaluation Complete ===")
