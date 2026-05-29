"""
Compute and plot Tanimoto similarity for each TL checkpoint
against the processed pIC50 dataset only.
Generates:
  - results/pd1_pdl1_tl_epoch_tanimoto.png  (overlaid histograms, one per epoch)
  - results/pd1_pdl1_tl_epoch_tanimoto_summary.csv  (mean/median/novelty per epoch)
"""
import glob, re, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

def morgan_fps(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        else:
            fps.append(None)
    return fps

# ── Load reference: processed pIC50 dataset ─────────────────────────────────
REF_CSV = 'Preprocess/Data_pd1_pdl1/data_csvs/pd1_pdl1_preprocess_all.csv'
df_ref  = pd.read_csv(REF_CSV)
ref_smiles = df_ref['smiles'].dropna().tolist()
print(f"Computing fingerprints for {len(ref_smiles)} pIC50 reference molecules...")
ref_fps_raw = morgan_fps(ref_smiles)
ref_fps = [fp for fp in ref_fps_raw if fp is not None]
print(f"  → {len(ref_fps)} valid reference fingerprints")

# ── Find TL checkpoint sample files ─────────────────────────────────────────
files = sorted(glob.glob("results/pd1_pdl1_tl_sample_e*.csv"),
               key=lambda f: int(re.findall(r'e(\d+)', f)[0]))
if not files:
    print("No TL checkpoint sample files found!"); sys.exit(1)

checkpoints = [(int(re.findall(r'e(\d+)', f)[0]), f) for f in files]
print(f"\nFound {len(checkpoints)} checkpoints: {[e for e, _ in checkpoints]}\n")

# ── Compute max Tanimoto per molecule for each epoch ────────────────────────
epoch_max_tanimotos = {}
summary_rows = []

for epoch, f_path in checkpoints:
    df = pd.read_csv(f_path)
    smiles = df['SMILES'].dropna().tolist()
    gen_fps = [fp for fp in morgan_fps(smiles) if fp is not None]
    
    max_tans = []
    for fp in tqdm(gen_fps, desc=f"  Epoch {epoch:3d}", leave=False):
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        max_tans.append(max(sims))
    
    epoch_max_tanimotos[epoch] = max_tans
    mean_t  = np.mean(max_tans)
    med_t   = np.median(max_tans)
    nov_85  = np.mean(np.array(max_tans) < 0.85) * 100
    nov_70  = np.mean(np.array(max_tans) < 0.70) * 100
    
    print(f"  Epoch {epoch:3d} | n={len(gen_fps):4d} | "
          f"Mean={mean_t:.3f} | Median={med_t:.3f} | "
          f"Novel<0.85: {nov_85:.1f}% | Novel<0.70: {nov_70:.1f}%")
    
    summary_rows.append({
        "Epoch": epoch,
        "N_Generated": len(gen_fps),
        "Mean_MaxTanimoto": round(mean_t, 4),
        "Median_MaxTanimoto": round(med_t, 4),
        "Novelty_T085": round(nov_85, 2),
        "Novelty_T070": round(nov_70, 2),
    })

# ── Save summary CSV ──────────────────────────────────────────────────────────
df_summary = pd.DataFrame(summary_rows)
summary_csv = 'results/pd1_pdl1_tl_epoch_tanimoto_summary.csv'
df_summary.to_csv(summary_csv, index=False)
print(f"\nSummary saved to: {summary_csv}")
print(df_summary.to_string(index=False))

# ── Plot overlaid histograms ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
colors = cm.plasma(np.linspace(0.1, 0.9, len(checkpoints)))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('PD1-PDL1 TL Epoch — Tanimoto Similarity vs pIC50 Dataset',
             fontsize=15, weight='bold')

# Left: overlaid KDE per epoch
ax = axes[0]
for idx, (epoch, max_tans) in enumerate(epoch_max_tanimotos.items()):
    sns.kdeplot(max_tans, label=f'Epoch {epoch}', color=colors[idx], lw=2.0, ax=ax)
ax.axvline(0.85, color='gold', ls='--', lw=1.5, label='T=0.85 novelty cutoff')
ax.axvline(0.70, color='tomato', ls='--', lw=1.5, label='T=0.70 novelty cutoff')
ax.set_xlabel('Max Tanimoto Similarity to pIC50 Dataset', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Max Tanimoto Distribution per Epoch (KDE)', fontsize=13, weight='bold')
ax.set_xlim(0, 1)
ax.legend(fontsize=9, loc='upper left')
ax.spines[['top','right']].set_visible(False)

# Right: Mean max Tanimoto progression across epochs
ax2 = axes[1]
epochs = [r['Epoch'] for r in summary_rows]
means  = [r['Mean_MaxTanimoto'] for r in summary_rows]
meds   = [r['Median_MaxTanimoto'] for r in summary_rows]
nov85  = [r['Novelty_T085'] for r in summary_rows]

ax2.plot(epochs, means, 'o-', color='#e74c3c', lw=2.5, ms=7, label='Mean Max Tanimoto')
ax2.plot(epochs, meds,  's--', color='#3498db', lw=2.0, ms=6, label='Median Max Tanimoto')
ax2.set_xlabel('TL Epoch', fontsize=12)
ax2.set_ylabel('Max Tanimoto to pIC50 Dataset', fontsize=12)
ax2.set_title('Tanimoto Progression across TL Epochs', fontsize=13, weight='bold')
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
outpath = 'results/pd1_pdl1_tl_epoch_tanimoto.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {outpath}")
