import os, sys, glob, re, pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.cm as cm, seaborn as sns
import xgboost as xgb
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import QED
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
RDLogger.DisableLog('rdApp.*')

# Import both feature extractors from REINVENT4
sys.path.append('REINVENT4')
from reinvent_plugins.components.nophyschem_features import compute_features as compute_pic50_features
from reinvent_plugins.components.pd1_pdl1_features import compute_features as compute_sol_features

PIC50_MODEL = 'Preprocess/final_acc/pd1_pdl1_pic50_final_acc_model.ubj'
PIC50_SCALER = 'Preprocess/final_acc/pd1_pdl1_pic50_final_acc_scaler.pkl'
SOL_MODEL = 'Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj'
SOL_SCALER = 'Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl'

bst_pic50 = xgb.Booster(); bst_pic50.load_model(PIC50_MODEL)
bst_sol   = xgb.Booster(); bst_sol.load_model(SOL_MODEL)

def predict_all(smiles_list):
    n = len(smiles_list)
    if n == 0:
        return {"pIC50": [], "logS": [], "SA": [], "QED": []}
    
    X_p, m_p = compute_pic50_features(smiles_list, PIC50_SCALER)
    preds_p   = bst_pic50.predict(xgb.DMatrix(X_p))
    
    X_s, m_s = compute_sol_features(smiles_list, SOL_SCALER)
    preds_s   = bst_sol.predict(xgb.DMatrix(X_s))
    
    sa_vals, qed_vals = [], []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        if mol:
            sa_vals.append(sascorer.calculateScore(mol))
            qed_vals.append(QED.qed(mol))
        else:
            sa_vals.append(np.nan); qed_vals.append(np.nan)
    
    return {
        "pIC50": [float(preds_p[i]) if m_p[i] else np.nan for i in range(n)],
        "logS":  [float(preds_s[i]) if m_s[i] else np.nan for i in range(n)],
        "SA": sa_vals, "QED": qed_vals
    }

# ── Load TL checkpoint sample files ────────────────────────────────────────────
files = sorted(glob.glob("results/pd1_pdl1_tl_run4_sample_e*.csv"),
               key=lambda f: int(re.findall(r'e(\d+)', f)[0]))
if not files:
    print("No TL run4 checkpoint sample files found!"); sys.exit(1)

checkpoints = [(int(re.findall(r'e(\d+)', f)[0]), f) for f in files if int(re.findall(r'e(\d+)', f)[0]) <= 100]
epoch_data = {}
for epoch, f_path in checkpoints:
    print(f"Predicting properties for Epoch {epoch}...")
    df = pd.read_csv(f_path)
    epoch_data[epoch] = predict_all(df['SMILES'].dropna().tolist())

# ── Load SINGLE baseline: enriched pIC50 dataset with predicted logS ───────────
print("\nLoading enriched pIC50 dataset as baseline...")
df_base = pd.read_csv('Preprocess/Data_pd1_pdl1/data_csvs/pd1_pdl1_preprocess_pic50_with_sol.csv')
base_smiles = df_base['smiles'].dropna().tolist()

raw_pic50  = df_base['pic50'].dropna().tolist()
raw_logS   = df_base['logS'].dropna().tolist()
raw_sa     = [sascorer.calculateScore(m) for s in base_smiles if (m := Chem.MolFromSmiles(str(s)))]
raw_qed    = [QED.qed(m) for s in base_smiles if (m := Chem.MolFromSmiles(str(s)))]
print(f"Baseline: {len(df_base)} pIC50 molecules with predicted logS\n")

# ── Plot ────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('PD1-PDL1 TL Run4 Epoch Progressions vs pIC50 Processed Baseline\n(logS from solubility model)', 
             fontsize=16, weight='bold', y=1.02)

colors = cm.plasma(np.linspace(0.1, 0.9, len(checkpoints)))

def plot_kde(ax, raw_vals, prop_key, title, xlabel, xlim=None):
    sns.kdeplot(raw_vals, label='pIC50 Dataset (baseline)', color='grey', ls='--', lw=2.5, ax=ax)
    for idx, (epoch, data) in enumerate(epoch_data.items()):
        vals = [v for v in data[prop_key] if pd.notna(v)]
        if vals:
            sns.kdeplot(vals, label=f'Epoch {epoch}', color=colors[idx], lw=1.8, ax=ax)
    ax.set_title(title, fontsize=13, weight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    if xlim: ax.set_xlim(xlim)

plot_kde(axes[0,0], raw_pic50, 'pIC50',  'pIC50 Affinity Evolution',               'Predicted pIC50')
plot_kde(axes[0,1], raw_logS,  'logS',   'Solubility (logS) Evolution',             'Predicted logS (pIC50 baseline)')
plot_kde(axes[1,0], raw_sa,    'SA',     'Synthetic Accessibility Evolution',       'SAScore')
plot_kde(axes[1,1], raw_qed,   'QED',    'Drug-Likeness (QED) Evolution',           'QED', xlim=(0, 1))

plt.tight_layout()
outpath = 'results/pd1_pdl1_tl_run4_epoch_kde_vs_pic50_baseline.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved to: {outpath}")
