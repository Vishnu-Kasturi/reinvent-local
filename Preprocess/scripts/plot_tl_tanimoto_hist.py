"""
Plot Tanimoto similarity histograms for each TL checkpoint
against the processed pIC50 dataset.
Reads existing pd1_pdl1_tl_sample_e{epoch}.csv files.
"""
import glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

def morgan_fps(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None)
    return fps

# ── Reference fingerprints ───────────────────────────────────────────────────
print("Computing reference fingerprints...")
df_ref  = pd.read_csv('Preprocess/Data_pd1_pdl1/data_csvs/pd1_pdl1_preprocess_all.csv')
ref_fps = [fp for fp in morgan_fps(df_ref['smiles'].dropna().tolist()) if fp is not None]
print(f"  {len(ref_fps)} pIC50 reference fingerprints\n")

# ── Load checkpoint samples ──────────────────────────────────────────────────
files = sorted(glob.glob("results/pd1_pdl1_tl_run2_sample_e*.csv"),
               key=lambda f: int(re.findall(r'e(\d+)', f)[0]))
checkpoints = [(int(re.findall(r'e(\d+)', f)[0]), f) for f in files]
print(f"Found {len(checkpoints)} checkpoints\n")

epoch_max_tans = {}
for epoch, fpath in checkpoints:
    smiles = pd.read_csv(fpath)['SMILES'].dropna().tolist()
    gen_fps = [fp for fp in morgan_fps(smiles) if fp is not None]
    max_tans = []
    for fp in tqdm(gen_fps, desc=f"Epoch {epoch:3d}", leave=False):
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        max_tans.append(max(sims))
    epoch_max_tans[epoch] = max_tans
    print(f"  Epoch {epoch:3d} | n={len(max_tans)} | Mean={np.mean(max_tans):.3f} | Median={np.median(max_tans):.3f}")

# ── Plot: 2x5 grid of histograms (one per epoch) ─────────────────────────────
n_epochs = len(checkpoints)
colors   = cm.plasma(np.linspace(0.1, 0.9, n_epochs))

fig, axes = plt.subplots(int(np.ceil(n_epochs/5)), 5, figsize=(22, 9), sharey=False, sharex=True)
fig.suptitle('PD1-PDL1 TL Run2 Checkpoints — Max Tanimoto Similarity vs pIC50 Dataset\n(500 samples per epoch)',
             fontsize=16, weight='bold', y=1.02)

bins = np.linspace(0, 1, 35)

for idx, ((epoch, _), ax) in enumerate(zip(checkpoints, axes.flatten())):
    max_tans = epoch_max_tans[epoch]
    mean_t   = np.mean(max_tans)
    med_t    = np.median(max_tans)
    nov_85   = np.mean(np.array(max_tans) < 0.85) * 100

    ax.hist(max_tans, bins=bins, color=colors[idx], edgecolor='white',
            linewidth=0.5, alpha=0.85)
    ax.axvline(mean_t, color='white', ls='-', lw=2.0, label=f'Mean={mean_t:.3f}')
    ax.axvline(0.85,   color='gold',  ls='--', lw=1.5, label='T=0.85')
    ax.axvline(0.70,   color='tomato',ls='--', lw=1.2, label='T=0.70')

    ax.set_title(f'Epoch {epoch}', fontsize=12, weight='bold')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Max Tanimoto', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.text(0.04, 0.93, f'Mean: {mean_t:.3f}\nMedian: {med_t:.3f}\nNovel<0.85: {nov_85:.0f}%',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    ax.spines[['top','right']].set_visible(False)

# Shared legend on last axis
axes.flatten()[-1].legend(fontsize=8, loc='upper left')

plt.tight_layout()
outpath = 'results/pd1_pdl1_tl_run2_epoch_tanimoto_hist.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath}")
