import os, sys, math, pickle, subprocess, json, tempfile, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, QED, DataStructs

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── SA Score (rdkit contrib) ──────────────────────────────────────────────────
try:
    from rdkit.Contrib.SA_Score import sascorer
    HAS_SA = True
except ImportError:
    try:
        sys.path.append(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "REINVENT4", "contrib"))
        from SA_Score import sascorer
        HAS_SA = True
    except ImportError:
        HAS_SA = False
        print("[WARN] SA Score not available — will be skipped")

REPO_ROOT    = "/Users/vishnukasturi/Intern/reinvent-local"
MODELS_DIR   = os.path.join(REPO_ROOT, "models")
RESULTS_DIR  = os.path.join(REPO_ROOT, "results")
REINVENT4    = os.path.join(REPO_ROOT, "REINVENT4")

# Import feature extractors from REINVENT4
sys.path.append(REINVENT4)
from reinvent_plugins.components.jak2_final_acc_features import compute_features as compute_jak2_features
from reinvent_plugins.components.pd1_pdl1_features import compute_features as compute_sol_features

JAK2_MODEL   = os.path.join(REPO_ROOT, "Preprocess", "final_acc", "jak2_pic50_final_acc_model.ubj")
JAK2_SCALER  = os.path.join(REPO_ROOT, "Preprocess", "final_acc", "jak2_pic50_final_acc_scaler.pkl")
SOL_MODEL    = os.path.join(REPO_ROOT, "Preprocess", "final_acc", "pd1_pdl1_sol_final_acc_model.ubj")
SOL_SCALER   = os.path.join(REPO_ROOT, "Preprocess", "final_acc", "pd1_pdl1_sol_final_acc_scaler.pkl")
TRAIN_SMI    = os.path.join(REPO_ROOT, "data", "jak2_tl_train.smi")
REF_CSV      = os.path.join(REPO_ROOT, "Preprocess", "Data_jak2", "processed_all_with_sol.csv")

N_SAMPLE     = 500   # molecules to sample per checkpoint

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading models …")
m_jak2 = xgb.Booster(); m_jak2.load_model(JAK2_MODEL)
m_sol  = xgb.Booster(); m_sol.load_model(SOL_MODEL)

# ── Load training fingerprints for Tanimoto ───────────────────────────────────
print("Computing training set fingerprints …")
with open(TRAIN_SMI) as f:
    train_smiles = [l.strip() for l in f if l.strip()]
train_fps = []
for s in train_smiles:
    m = Chem.MolFromSmiles(s)
    if m:
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
print(f"  Training FPs: {len(train_fps):,}")

def get_max_tanimotos(smiles_list):
    scores = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        scores.append(max(sims) if sims else 0.0)
    return scores

# ── Predict properties ─────────────────────────────────────────────────────────
def predict_batch(smiles_list):
    n = len(smiles_list)
    if n == 0:
        return [], [], [], []
        
    # 1. Predict pIC50 using the jak2 feature pipeline
    X_p, m_p = compute_jak2_features(smiles_list, JAK2_SCALER)
    preds_p = m_jak2.predict(xgb.DMatrix(X_p))
    pic50s = [float(preds_p[i]) if m_p[i] else np.nan for i in range(n)]
    
    # 2. Predict logS using the solubility feature pipeline
    X_s, m_s = compute_sol_features(smiles_list, SOL_SCALER)
    preds_s = m_sol.predict(xgb.DMatrix(X_s))
    logss = [float(preds_s[i]) if m_s[i] else np.nan for i in range(n)]
    
    sas, qeds = [], []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            sas.append(np.nan)
            qeds.append(np.nan)
            continue
        # SA
        if HAS_SA:
            try: sas.append(sascorer.calculateScore(mol))
            except: sas.append(np.nan)
        else:
            sas.append(np.nan)
        # QED
        try: qeds.append(QED.qed(mol))
        except: qeds.append(np.nan)
        
    return pic50s, logss, sas, qeds

# ── Sampling via REINVENT ─────────────────────────────────────────────────────
def sample_checkpoint(chkpt_path: str, n: int = N_SAMPLE) -> list[str]:
    """Use reinvent sampling to get N SMILES from a checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_smi = os.path.join(tmpdir, "sampled.smi")
        toml_content = f'''
run_type = "sampling"
[parameters]
model_file  = "{chkpt_path}"
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
            capture_output=True, cwd=REINVENT4
        )
        if os.path.exists(out_smi):
            with open(out_smi) as f:
                lines = [l.strip().split(',')[0] for l in f if l.strip()]
            return [l for l in lines if Chem.MolFromSmiles(l)]
        return []

def main():
    # Detect checkpoints from models/ directory
    import glob
    chkpts = sorted(glob.glob(os.path.join(MODELS_DIR, "jak2_tl.model.*.chkpt")))
    if not chkpts:
        print("[ERROR] No checkpoints found for jak2_tl.model.*.chkpt")
        sys.exit(1)
        
    import time
    # Filter checkpoints generated in the last 3 hours (today's run)
    today_chkpts = []
    three_hours_ago = time.time() - 3 * 3600
    for c in chkpts:
        if os.path.getmtime(c) > three_hours_ago:
            today_chkpts.append(c)
            
    if today_chkpts:
        print(f"Filtering to today's checkpoints: {[os.path.basename(c) for c in today_chkpts]}")
        chkpts_to_use = today_chkpts
    else:
        print("No checkpoints modified in the last 3 hours. Using all checkpoints.")
        chkpts_to_use = chkpts
        
    epochs = sorted(set([int(c.split(".")[-2]) for c in chkpts_to_use]))
    print(f"Found checkpoints at epochs: {epochs}")
    
    selected_epochs = sorted(list(set(epochs)))
    print(f"Selected epochs for evaluation: {selected_epochs}")
    
    epoch_max_tanimotos = {}
    epoch_data = {}
    
    for epoch in tqdm(selected_epochs, desc="Evaluating Checkpoints"):
        chkpt = os.path.join(MODELS_DIR, f"jak2_tl.model.{epoch}.chkpt")
        sampled = sample_checkpoint(chkpt, N_SAMPLE)
        if not sampled:
            print(f"  [!] Failed to sample from epoch {epoch}"); continue
            
        # Tanimoto similarity
        tans = get_max_tanimotos(sampled)
        epoch_max_tanimotos[epoch] = tans
        
        # Predicted properties
        pic50s, logss, sas, qeds = predict_batch(sampled)
        epoch_data[epoch] = {
            "pIC50": pic50s,
            "logS": logss,
            "SA": sas,
            "QED": qeds
        }
        
        mean_t = np.mean(tans)
        nov_85 = np.mean(np.array(tans) < 0.85) * 100
        print(f"  Epoch {epoch:3d} | n={len(sampled)} | Mean Max Tanimoto={mean_t:.3f} | Novelty<0.85={nov_85:.1f}% | Mean pIC50={np.nanmean(pic50s):.2f}")

    # Save summary data
    summary_rows = []
    for epoch in epoch_data:
        summary_rows.append({
            "Epoch": epoch,
            "Mean_MaxTanimoto": np.mean(epoch_max_tanimotos[epoch]),
            "Mean_pIC50": np.nanmean(epoch_data[epoch]["pIC50"]),
            "Mean_logS": np.nanmean(epoch_data[epoch]["logS"]),
            "Mean_QED": np.nanmean(epoch_data[epoch]["QED"]),
            "Mean_SA": np.nanmean(epoch_data[epoch]["SA"]) if HAS_SA else np.nan
        })
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(RESULTS_DIR, "jak2_tl_epoch_metrics_summary.csv"), index=False)
    print(f"\n{df_sum.to_string(index=False)}")

    # ── 1. Tanimoto similarity histograms grid ────────────────────────────────
    sns.set_theme(style="whitegrid")
    n_epochs = len(epoch_data)
    if n_epochs > 0:
        colors = cm.plasma(np.linspace(0.1, 0.9, n_epochs))
        ncols = 3
        nrows = int(np.ceil(n_epochs / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows*4), sharey=False, sharex=True)
        fig.suptitle('JAK2 Transfer Learning — Max Tanimoto Histograms to Training Set',
                     fontsize=14, weight='bold', y=1.01)
        bins = np.linspace(0, 1, 35)
        for idx, epoch in enumerate(sorted(epoch_data.keys())):
            ax = axes.flatten()[idx] if n_epochs > 1 else axes
            max_tans = epoch_max_tanimotos[epoch]
            mean_t = np.mean(max_tans)
            nov_85 = np.mean(np.array(max_tans) < 0.85) * 100
            ax.hist(max_tans, bins=bins, color=colors[idx], edgecolor='white', lw=0.5, alpha=0.85)
            ax.axvline(mean_t, color='#2c3e50', ls='--', lw=1.5, label=f'Mean: {mean_t:.2f}')
            ax.axvline(0.85, color='red', ls=':', lw=1.2, label='T=0.85')
            ax.set_title(f"Epoch {epoch} (Novel<0.85={nov_85:.1f}%)", fontsize=11, weight='bold')
            ax.set_xlim(0, 1.05)
            ax.legend(fontsize=8, loc='upper left')

        if n_epochs > 1:
            for i in range(n_epochs, nrows*ncols):
                fig.delaxes(axes.flatten()[i])

        plt.tight_layout()
        out_tanimoto = os.path.join(RESULTS_DIR, "jak2_tl_epoch_tanimoto.png")
        plt.savefig(out_tanimoto, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved Tanimoto plot to: {out_tanimoto}")

    # ── 2. Property KDE Evolution vs Baseline ──────────────────────────────────
    if n_epochs > 0:
        print("Generating KDE plots...")
        df_ref = pd.read_csv(REF_CSV)
        raw_pic50 = df_ref['pic50'].dropna().tolist()
        raw_logS  = df_ref['pred_logS'].dropna().tolist()
        raw_sa, raw_qed = [], []
        for s in df_ref['smiles'].dropna().tolist():
            mol = Chem.MolFromSmiles(str(s))
            if mol:
                raw_sa.append(sascorer.calculateScore(mol))
                raw_qed.append(QED.qed(mol))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('JAK2 Transfer Learning Property Evolution vs Baseline Dataset',
                     fontsize=16, weight='bold', y=1.02)

        colors_subset = cm.plasma(np.linspace(0.1, 0.9, n_epochs))

        def plot_kde(ax, baseline_vals, prop_key, title, xlabel, xlim=None):
            sns.kdeplot(baseline_vals, label='JAK2 Baseline Dataset', color='grey', ls='--', lw=2.5, ax=ax, cut=0)
            for idx, epoch in enumerate(sorted(epoch_data.keys())):
                vals = [v for v in epoch_data[epoch][prop_key] if pd.notna(v)]
                if vals:
                    sns.kdeplot(vals, label=f'Epoch {epoch}', color=colors_subset[idx], lw=2.0, ax=ax)
            
            ax.set_title(title, fontsize=13, weight='bold')
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.legend(fontsize=8, loc='upper right')
            if xlim: ax.set_xlim(xlim)
            ax.spines[['top', 'right']].set_visible(False)

        plot_kde(axes[0,0], raw_pic50, 'pIC50', 'pIC50 Affinity Evolution',               'Predicted pIC50', (4, 12))
        plot_kde(axes[0,1], raw_logS,  'logS',  'Solubility (logS) Evolution',             'Predicted logS', (-10, 2))
        plot_kde(axes[1,0], raw_sa,    'SA',    'Synthetic Accessibility Evolution',       'SAScore', (1, 8))
        plot_kde(axes[1,1], raw_qed,   'QED',   'Drug-Likeness (QED) Evolution',           'QED', (0, 1))

        plt.tight_layout()
        out_kde = os.path.join(RESULTS_DIR, "jak2_tl_epoch_kde_vs_baseline.png")
        plt.savefig(out_kde, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved KDE plot to: {out_kde}")

if __name__ == "__main__":
    main()
