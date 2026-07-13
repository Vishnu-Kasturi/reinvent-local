import os, sys, warnings, subprocess, tempfile
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import QED

warnings.filterwarnings("ignore")

# Load sascorer
try:
    sys.path.append("/Users/vishnukasturi/Intern/reinvent-local/REINVENT4/reinvent_plugins/components/SAScore")
    import sascorer
    HAS_SA = True
except ImportError:
    HAS_SA = False

sys.path.append("/Users/vishnukasturi/Intern/reinvent-local/REINVENT4")
from reinvent_plugins.components.jak2_final_acc_features import compute_features as compute_jak2_features
from reinvent_plugins.components.pd1_pdl1_features import compute_features as compute_sol_features

REPO_ROOT = "/Users/vishnukasturi/Intern/reinvent-local"
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
REF_CSV = os.path.join(REPO_ROOT, "Preprocess", "Data_jak2", "processed_all_with_sol.csv")
RL_CSV = os.path.join(REPO_ROOT, "results", "jak2_rl_candidates_run6.csv")
TL_MODEL = os.path.join(REPO_ROOT, "models", "jak2_tl.model")

def sample_model(model_path, n=500):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_smi = os.path.join(tmpdir, "sampled.smi")
        toml_content = f'''
run_type = "sampling"
[parameters]
model_file  = "{model_path}"
output_file = "{out_smi}"
num_smiles  = {n}
unique_molecules = true
randomize_smiles = true
'''
        toml_path = os.path.join(tmpdir, "sample.toml")
        with open(toml_path, "w") as f:
            f.write(toml_content)
        
        subprocess.run(
            ["conda", "run", "-n", "reinvent4", "reinvent", "-l", os.path.join(tmpdir, "s.log"), toml_path],
            capture_output=True, cwd=os.path.join(REPO_ROOT, "REINVENT4")
        )
        if os.path.exists(out_smi):
            with open(out_smi) as f:
                lines = [l.strip().split(',')[0] for l in f if l.strip()]
            return [l for l in lines if Chem.MolFromSmiles(l)]
        return []

def predict_batch(smiles_list, m_jak2, m_sol):
    n = len(smiles_list)
    if n == 0:
        return [], [], [], []
        
    X_p, m_p = compute_jak2_features(smiles_list, '/Users/vishnukasturi/Intern/reinvent-local/Preprocess/final_acc/jak2_pic50_final_acc_scaler.pkl')
    preds_p = m_jak2.predict(xgb.DMatrix(X_p))
    pic50s = [float(preds_p[i]) if m_p[i] else np.nan for i in range(n)]
    
    X_s, m_s = compute_sol_features(smiles_list, '/Users/vishnukasturi/Intern/reinvent-local/Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl')
    preds_s = m_sol.predict(xgb.DMatrix(X_s))
    logss = [float(preds_s[i]) if m_s[i] else np.nan for i in range(n)]
    
    sas, qeds = [], []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            qeds.append(QED.qed(mol))
            if HAS_SA:
                try: sas.append(sascorer.calculateScore(mol))
                except: sas.append(np.nan)
            else:
                sas.append(np.nan)
        else:
            qeds.append(np.nan)
            sas.append(np.nan)
    return pic50s, logss, sas, qeds

def main():
    print("Loading models for RL evaluation...")
    m_jak2 = xgb.Booster(); m_jak2.load_model('/Users/vishnukasturi/Intern/reinvent-local/Preprocess/final_acc/jak2_pic50_final_acc_model.ubj')
    m_sol  = xgb.Booster(); m_sol.load_model('/Users/vishnukasturi/Intern/reinvent-local/Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj')
    
    # 1. Load Baseline Dataset
    print("Loading baseline dataset...")
    df_ref = pd.read_csv(REF_CSV)
    raw_pic50 = df_ref['pic50'].dropna().tolist()
    raw_logS  = df_ref['pred_logS'].dropna().tolist()
    raw_sa, raw_qed = [], []
    for s in df_ref['smiles'].dropna().tolist():
        mol = Chem.MolFromSmiles(str(s))
        if mol:
            raw_qed.append(QED.qed(mol))
            if HAS_SA:
                raw_sa.append(sascorer.calculateScore(mol))
                
    # 2. Sample and Predict TL Model (Epoch 150)
    print("Sampling TL model (Epoch 150)...")
    tl_smiles = sample_model(TL_MODEL, n=500)
    tl_pic50, tl_logS, tl_sa, tl_qed = predict_batch(tl_smiles, m_jak2, m_sol)
    
    # 3. Load RL Model Candidates
    print("Loading and analyzing RL candidates...")
    df_rl = pd.read_csv(RL_CSV)
    rl_pic50 = df_rl['pIC50'].dropna().tolist()
    rl_logS  = df_rl['logS'].dropna().tolist()
    rl_sa    = df_rl['SA'].dropna().tolist()
    rl_qed   = df_rl['QED'].dropna().tolist()
    
    # 4. Extract and Save Best Candidates (MPO Hits)
    hits = df_rl[(df_rl['pIC50'] >= 8.0) & (df_rl['logS'] >= -4.0) & (df_rl['SA'] <= 4.0)]
    best_path = os.path.join(RESULTS_DIR, "jak2_rl_best_candidates.csv")
    hits.to_csv(best_path, index=False)
    print(f"Saved {len(hits)} best candidates satisfying MPO targets to: {best_path}")
    
    # ── KDE Plotting ──────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('JAK2 Generative Evolution: Baseline vs TL vs Reinforcement Learning (RL)',
                 fontsize=16, weight='bold', y=1.02)
                 
    def plot_three_way(ax, baseline_vals, tl_vals, rl_vals, title, xlabel, xlim=None):
        sns.kdeplot(baseline_vals, label='JAK2 Baseline Dataset', color='#7f8c8d', ls='--', lw=2.5, ax=ax, cut=0)
        sns.kdeplot(tl_vals, label='Transfer Learning (Epoch 150)', color='#3498db', lw=2.0, ax=ax, cut=0)
        sns.kdeplot(rl_vals, label='Reinforcement Learning (MPO)', color='#e74c3c', lw=2.5, ax=ax, cut=0)
        
        ax.set_title(title, fontsize=13, weight='bold')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        if xlim: ax.set_xlim(xlim)
        ax.spines[['top', 'right']].set_visible(False)
        
    plot_three_way(axes[0,0], raw_pic50, tl_pic50, rl_pic50, 'pIC50 Affinity Shift', 'Predicted pIC50', (4, 12))
    plot_three_way(axes[0,1], raw_logS, tl_logS, rl_logS, 'Solubility (logS) Shift', 'Predicted logS', (-10, 2))
    plot_three_way(axes[1,0], raw_sa, tl_sa, rl_sa, 'Synthetic Accessibility (SA)', 'SAScore', (1, 8))
    plot_three_way(axes[1,1], raw_qed, tl_qed, rl_qed, 'Drug-Likeness (QED)', 'QED', (0, 1))
    
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "jak2_rl_kde_vs_baseline.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved RL vs Baseline KDE plot to: {out_png}")

if __name__ == "__main__":
    main()
