import os, sys, glob, re, subprocess, pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.cm as cm, seaborn as sns
from tqdm import tqdm
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

REPO_ROOT   = os.path.abspath('/Users/vishnukasturi/Intern/reinvent-local')
MODELS_DIR  = os.path.join(REPO_ROOT, 'models')
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
CONFIGS_DIR = os.path.join(REPO_ROOT, 'REINVENT4', 'configs')

def morgan_fps(smiles_list):
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s))
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None)
    return fps

# ── Reference Dataset ────────────────────────────────────────────────────────
REF_CSV = 'Preprocess/Data_pd1_pdl1/data_csvs/pd1_pdl1_preprocess_all.csv'
df_ref  = pd.read_csv(REF_CSV)
ref_fps = [fp for fp in morgan_fps(df_ref['smiles'].dropna().tolist()) if fp is not None]
print(f"Reference: {len(ref_fps)} pIC50 fingerprints\n")

# ── Find checkpoints ─────────────────────────────────────────────────────────
checkpoints = []
for epoch in range(10, 110, 10):
    p = os.path.join(MODELS_DIR, f'pd1_pdl1_TL_run3.model.{epoch}.chkpt')
    if os.path.exists(p):
        checkpoints.append((epoch, p))
print(f"Found {len(checkpoints)} checkpoints: {[e for e,_ in checkpoints]}\n")

# ── Sample + compute Tanimoto ─────────────────────────────────────────────────
epoch_max_tanimotos = {}
summary_rows = []

for epoch, chkpt_path in checkpoints:
    out_csv  = os.path.join(RESULTS_DIR, f'pd1_pdl1_tl_run3_sample_e{epoch}.csv')
    toml_path = os.path.join(CONFIGS_DIR, f'_tl_run3_sample_e{epoch}.toml')
    
    toml_content = f"""run_type = "sampling"
device   = "cpu"
json_out_config = "_tl_run3_sample_e{epoch}.json"

[parameters]
model_file      = "{chkpt_path}"
output_file     = "{out_csv}"
num_smiles      = 600
unique_molecules = true
sample_strategy = "multinomial"
temperature     = 1.0
"""
    with open(toml_path, 'w') as f:
        f.write(toml_content)
    
    print(f"Epoch {epoch:3d}: Sampling 600 → target 500 valid...")
    subprocess.run(
        ['conda', 'run', '-n', 'reinvent-qsar', 'reinvent', toml_path],
        cwd=os.path.join(REPO_ROOT, 'REINVENT4'),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    if not os.path.exists(out_csv):
        print(f"  [!] Output not found for Epoch {epoch}, skipping.")
        continue
    
    df = pd.read_csv(out_csv)
    n_valid = len(df)
    
    df = df.head(500)
    df.to_csv(out_csv, index=False)
    
    smiles = df['SMILES'].dropna().tolist()
    gen_fps = [fp for fp in morgan_fps(smiles) if fp is not None]
    
    max_tans = []
    for fp in tqdm(gen_fps, desc=f"  Tanimoto E{epoch:3d}", leave=False):
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        max_tans.append(max(sims))
    
    epoch_max_tanimotos[epoch] = max_tans
    mean_t = np.mean(max_tans)
    med_t  = np.median(max_tans)
    nov_85 = np.mean(np.array(max_tans) < 0.85) * 100
    nov_70 = np.mean(np.array(max_tans) < 0.70) * 100
    
    print(f"          n_sampled={n_valid} → trimmed to {len(gen_fps)} | "
          f"Mean={mean_t:.3f} | Median={med_t:.3f} | "
          f"Novel<0.85: {nov_85:.1f}% | Novel<0.70: {nov_70:.1f}%")
    
    summary_rows.append({
        "Epoch": epoch, "N_Generated": len(gen_fps),
        "Mean_MaxTanimoto": round(mean_t, 4),
        "Median_MaxTanimoto": round(med_t, 4),
        "Novelty_T085": round(nov_85, 2),
        "Novelty_T070": round(nov_70, 2),
    })

# ── Save summary ──────────────────────────────────────────────────────────────
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv('results/pd1_pdl1_tl_run3_epoch_tanimoto_summary.csv', index=False)
print(f"\n{df_summary.to_string(index=False)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
colors = cm.plasma(np.linspace(0.1, 0.9, len(epoch_max_tanimotos)))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('PD1-PDL1 TL Run3 Epoch — Tanimoto Similarity vs pIC50 Dataset (500 samples each)',
             fontsize=15, weight='bold')

ax = axes[0]
for idx, (epoch, max_tans) in enumerate(epoch_max_tanimotos.items()):
    sns.kdeplot(max_tans, label=f'Epoch {epoch}', color=colors[idx], lw=2.0, ax=ax)
ax.axvline(0.85, color='gold', ls='--', lw=1.5, label='T=0.85 novelty cutoff')
ax.axvline(0.70, color='tomato', ls='--', lw=1.5, label='T=0.70 novelty cutoff')
ax.set_xlabel('Max Tanimoto Similarity to pIC50 Dataset', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Max Tanimoto Distribution per Epoch (KDE)', fontsize=13, weight='bold')
ax.set_xlim(0, 1)
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)

ax2 = axes[1]
epochs = [r['Epoch'] for r in summary_rows]
means  = [r['Mean_MaxTanimoto'] for r in summary_rows]
meds   = [r['Median_MaxTanimoto'] for r in summary_rows]
nov85  = [r['Novelty_T085'] for r in summary_rows]

ax2.plot(epochs, means, 'o-', color='#e74c3c', lw=2.5, ms=7, label='Mean Max Tanimoto')
ax2.plot(epochs, meds,  's--', color='#3498db', lw=2.0, ms=6, label='Median Max Tanimoto')
ax2.set_xlabel('TL Epoch', fontsize=12)
ax2.set_ylabel('Max Tanimoto to pIC50 Dataset', fontsize=12)
ax2.set_title('Tanimoto Progression across TL Epochs\n(500 samples per epoch)', fontsize=13, weight='bold')
ax2.set_ylim(0, 1)

ax3 = ax2.twinx()
ax3.plot(epochs, nov85, '^:', color='#27ae60', lw=2.0, ms=6, label='% Novel (T<0.85)')
ax3.set_ylabel('% Structurally Novel Molecules', fontsize=12, color='#27ae60')
ax3.tick_params(axis='y', labelcolor='#27ae60')
ax3.set_ylim(0, 100)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='lower right')
ax2.spines[['top']].set_visible(False)

plt.tight_layout()
plt.savefig('results/pd1_pdl1_tl_run3_epoch_tanimoto.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to: results/pd1_pdl1_tl_run3_epoch_tanimoto.png")
