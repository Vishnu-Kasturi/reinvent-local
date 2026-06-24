#!/usr/bin/env python3
"""
evaluate_pdl1_ph_d_tl_run3.py
==============================
Samples 500 molecules from every 10-epoch checkpoint of pdl1_ph_d_tl_run3.model,
produces Tanimoto histograms and KDE property plots vs. filtered_smiles_tanimoto.
Run3: freeze_n_layers=2, lr=5e-5 (LSTM 1, LSTM 2 + head trainable).
"""

import os, sys, subprocess, math, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import matplotlib.cm as cm

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED, Descriptors, MACCSkeys, rdMolDescriptors

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.abspath(os.path.join(HERE, ".."))
MODELS_DIR  = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
CONFIGS_DIR = os.path.join(ROOT, "REINVENT4", "configs")
REINVENT4   = os.path.join(ROOT, "REINVENT4")

REF_CSV      = os.path.join(HERE, "Data_pd1_pdl1", "filtered_smiles_tanimoto.csv")
PIC50_MODEL  = os.path.join(HERE, "final_acc", "pd1_pdl1_pic50_final_acc_model.ubj")
PIC50_SCALER = os.path.join(HERE, "final_acc", "pd1_pdl1_pic50_final_acc_scaler.pkl")
SOL_MODEL    = os.path.join(HERE, "final_acc", "pd1_pdl1_sol_final_acc_model.ubj")
SOL_SCALER   = os.path.join(HERE, "final_acc", "pd1_pdl1_sol_final_acc_scaler.pkl")

N_SAMPLE   = 500
MODEL_STEM = "pdl1_ph_d_tl_run3"
RUN_LABEL  = "pdl1_ph_d TL run3 (freeze_n_layers=2, lr=5e-5)"
EPOCHS     = list(range(10, 160, 10))

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Feature extractor ─────────────────────────────────────────────────────────
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

print("[*] Loading reference dataset...")
df_ref     = pd.read_csv(REF_CSV)
ref_smiles = df_ref["SMILES"].dropna().tolist()
ref_fps    = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(str(s)), 2, 2048)
              for s in ref_smiles if Chem.MolFromSmiles(str(s))]
print(f"  Reference FPs: {len(ref_fps)}")
ref_pic50, ref_lgs = predict_batch(ref_smiles[:300])
print(f"  Reference baseline: {len(ref_pic50)} pIC50, {len(ref_lgs)} logS")

run_env = os.environ.copy()

def sample_checkpoint(chkpt_path, epoch):
    out_csv = os.path.join(RESULTS_DIR, f"{MODEL_STEM}_sample_e{epoch}.csv")
    if os.path.exists(out_csv):
        print(f"  Epoch {epoch:3d}: existing CSV found, skipping.")
        return out_csv
    toml_path = os.path.join(CONFIGS_DIR, f"_ph_d_tl_run3_sample_e{epoch}.toml")
    with open(toml_path, "w") as f:
        f.write(f"""run_type = "sampling"
device   = "mps"
json_out_config = "_ph_d_tl_run3_sample_e{epoch}.json"

[parameters]
model_file       = "{chkpt_path}"
output_file      = "{out_csv}"
num_smiles       = {N_SAMPLE}
unique_molecules = true
sample_strategy  = "multinomial"
temperature      = 1.0
""")
    subprocess.run(
        ["conda", "run", "-n", "reinvent4", "reinvent", toml_path],
        cwd=REINVENT4, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=run_env
    )
    return out_csv if os.path.exists(out_csv) else None

# ── Main loop ─────────────────────────────────────────────────────────────────
summary_rows, all_tanimoto_data, all_prop_data = [], {}, {}

for epoch in EPOCHS:
    chkpt = os.path.join(MODELS_DIR, f"{MODEL_STEM}.model.{epoch}.chkpt")
    if not os.path.exists(chkpt):
        print(f"  [skip] Epoch {epoch}: checkpoint not found"); continue
    out_csv = sample_checkpoint(chkpt, epoch)
    if out_csv is None or not os.path.exists(out_csv):
        print(f"  [skip] Epoch {epoch}: no CSV produced"); continue
    smiles  = pd.read_csv(out_csv).head(N_SAMPLE)["SMILES"].dropna().tolist()
    gen_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
               for s in smiles if Chem.MolFromSmiles(s)]
    max_tans = [max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps)) for fp in gen_fps]
    all_tanimoto_data[epoch] = max_tans
    mean_t = np.mean(max_tans); med_t = np.median(max_tans)
    nov_85 = np.mean(np.array(max_tans) < 0.85) * 100
    nov_70 = np.mean(np.array(max_tans) < 0.70) * 100
    print(f"  Epoch {epoch:3d}: n={len(gen_fps)} | Mean={mean_t:.3f} | Median={med_t:.3f} | "
          f"Novel<0.85={nov_85:.1f}% | Novel<0.70={nov_70:.1f}%")
    pic50s, logss = predict_batch(smiles)
    all_prop_data[epoch] = {"pic50": pic50s, "lgs": logss}
    summary_rows.append({
        "Epoch": epoch, "N_Generated": len(gen_fps),
        "Mean_MaxTanimoto": round(mean_t, 4), "Median_MaxTanimoto": round(med_t, 4),
        "Novelty_T085": round(nov_85, 2), "Novelty_T070": round(nov_70, 2),
        "Mean_pIC50": round(np.nanmean(pic50s), 4) if pic50s else None,
        "Mean_logS":  round(np.nanmean(logss),  4) if logss  else None,
    })

# ── Save summary ──────────────────────────────────────────────────────────────
df_sum = pd.DataFrame(summary_rows)
sum_csv = os.path.join(RESULTS_DIR, f"{MODEL_STEM}_epoch_tanimoto_summary.csv")
df_sum.to_csv(sum_csv, index=False)
print(f"\n[+] Summary CSV → {sum_csv}")
print(df_sum.to_string(index=False))

# ── Tanimoto histogram grid ───────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
n_epochs = len(all_tanimoto_data)
if n_epochs > 0:
    colors = cm.plasma(np.linspace(0.1, 0.9, n_epochs))
    ncols = 5; nrows = math.ceil(n_epochs / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 4), sharex=True)
    fig.suptitle(f"{RUN_LABEL}\nMax Tanimoto per Epoch vs. filtered_smiles_tanimoto",
                 fontsize=13, weight="bold", y=1.01)
    bins = np.linspace(0, 1, 35)
    for idx, (epoch, max_tans) in enumerate(sorted(all_tanimoto_data.items())):
        ax = axes.flatten()[idx]
        mean_t = np.mean(max_tans); nov_85 = np.mean(np.array(max_tans) < 0.85) * 100
        ax.hist(max_tans, bins=bins, color=colors[idx], edgecolor="white", lw=0.5, alpha=0.85)
        ax.axvline(mean_t, color="#2c3e50", ls="--", lw=1.5, label=f"Mean: {mean_t:.2f}")
        ax.axvline(0.85, color="red", ls=":", lw=1.2, label="T=0.85")
        ax.set_title(f"Epoch {epoch} (Novel<0.85={nov_85:.1f}%)", fontsize=10, weight="bold")
        ax.set_xlim(0, 1.05); ax.legend(fontsize=7, loc="upper left")
    for i in range(n_epochs, nrows * ncols):
        fig.delaxes(axes.flatten()[i])
    plt.tight_layout()
    tan_plot = os.path.join(RESULTS_DIR, f"{MODEL_STEM}_epoch_tanimoto.png")
    plt.savefig(tan_plot, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[+] Tanimoto histogram grid → {tan_plot}")

# ── KDE property evolution ────────────────────────────────────────────────────
if len(all_prop_data) > 0:
    selected = sorted(all_prop_data.keys())
    colors_k = cm.viridis(np.linspace(0.15, 0.85, len(selected)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"{RUN_LABEL}\npIC50 & logS KDE vs. filtered_smiles_tanimoto baseline",
                 fontsize=12, weight="bold", y=1.01)
    if ref_pic50:
        sns.kdeplot(ref_pic50, label="Reference", color="gray", ls="--", lw=2, ax=axes[0])
    for idx, epoch in enumerate(selected):
        vals = all_prop_data[epoch]["pic50"]
        if vals: sns.kdeplot(vals, label=f"E{epoch}", color=colors_k[idx], lw=1.5, ax=axes[0])
    axes[0].set_title("Predicted pIC50", weight="bold"); axes[0].set_xlabel("pIC50")
    axes[0].legend(fontsize=7); axes[0].spines[["top","right"]].set_visible(False)
    if ref_lgs:
        sns.kdeplot(ref_lgs, label="Reference", color="gray", ls="--", lw=2, ax=axes[1])
    for idx, epoch in enumerate(selected):
        vals = all_prop_data[epoch]["lgs"]
        if vals: sns.kdeplot(vals, label=f"E{epoch}", color=colors_k[idx], lw=1.5, ax=axes[1])
    axes[1].set_title("Predicted logS", weight="bold"); axes[1].set_xlabel("logS")
    axes[1].legend(fontsize=7); axes[1].spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    kde_plot = os.path.join(RESULTS_DIR, f"{MODEL_STEM}_epoch_kde_vs_baseline.png")
    plt.savefig(kde_plot, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[+] KDE plot → {kde_plot}")

print(f"\n=== {RUN_LABEL} Evaluation Complete ===")
