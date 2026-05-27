import os, sys, time, glob, pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.cm as cm, seaborn as sns
from scipy.stats import gaussian_kde
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

RES = 'results'
RL_CSV = f'{RES}/pd1_pdl1_rl_toml_1.csv'
RAW_CSV = 'Preprocess/Data_pd1_pdl1/pd1_pdl1_pic50_raw.csv'
TL_CSV = f'{RES}/pd1_pdl1_tl_sample_e50.csv'
PIC50 = 'PD1PDL1pIC50_raw (raw)'
OUT_PNG = f'{RES}/pd1_pdl1_live_rl_monitor.png'

# Load Baseline Data
df_raw = pd.read_csv(RAW_CSV, sep='\t')
df_raw.columns = [c.strip().lower() for c in df_raw.columns]
raw_pic50 = df_raw['pic50'].dropna().values

df_tl = pd.read_csv(TL_CSV)
tl_smiles = df_tl['SMILES'].dropna().tolist()

def get_fps(sl):
    out=[]
    for s in sl:
        m=Chem.MolFromSmiles(str(s))
        if m: out.append(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048))
    return out

tl_fps = get_fps(tl_smiles)

print('[*] Starting Live RL Monitor (Updates every 10 seconds)...')
last_rows = 0

while True:
    if not os.path.exists(RL_CSV):
        time.sleep(5)
        continue
    
    try:
        df = pd.read_csv(RL_CSV).dropna(subset=[PIC50])
    except Exception:
        time.sleep(5)
        continue

    if len(df) <= last_rows:
        time.sleep(10)
        continue
    
    last_rows = len(df)
    max_step = df['step'].max()
    
    # 1. Progression: Mean pIC50 per step
    steps = sorted(df['step'].unique())
    means = df.groupby('step')[PIC50].mean()
    
    # 2. Latest Window (last 30 steps) for KDE and Tanimoto
    window_start = max(1, max_step - 30)
    df_window = df[df['step'] >= window_start]
    window_pic50 = df_window[PIC50].values
    window_smiles = df_window['SMILES'].dropna().tolist()
    
    # 3. Tanimoto vs TL
    opt_fps = get_fps(window_smiles)
    max_tans=[]; exact=0
    for fp in opt_fps:
        if not fp or not tl_fps: continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, tl_fps)
        ms = max(sims)
        max_tans.append(ms)
        if ms >= 0.999: exact += 1
    
    mt = np.mean(max_tans) if max_tans else 0
    cr = (exact / len(opt_fps)) if opt_fps else 0

    # PLOT
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Live RL Monitoring — Step {max_step} | Total Molecules: {len(df)}', fontsize=14, weight='bold')

    # A) Progression
    axes[0].plot(means.index, means.values, color='#e74c3c', lw=2)
    axes[0].axhline(8.5, color='gold', ls='--', lw=1.5, label='pIC50=8.5')
    axes[0].set_title('Mean pIC50 Progression')
    axes[0].set_xlabel('RL Step')
    axes[0].set_ylabel('Mean pIC50')
    axes[0].legend()

    # B) pIC50 KDE (Latest Window)
    kde_r = gaussian_kde(raw_pic50)
    xs = np.linspace(4, 11, 200)
    axes[1].fill_between(xs, kde_r(xs), alpha=0.25, color='#2ecc71', label='Original')
    axes[1].plot(xs, kde_r(xs), color='#2ecc71', lw=1.5)
    
    if len(window_pic50) > 5:
        krl = gaussian_kde(window_pic50)
        axes[1].fill_between(xs, krl(xs), alpha=0.5, color='#e74c3c', label=f'Latest (Steps {window_start}-{max_step})')
        axes[1].plot(xs, krl(xs), color='#e74c3c', lw=2)
    
    axes[1].axvline(8.5, color='gold', ls='--', lw=1.5)
    axes[1].set_title('pIC50 Density (Latest 30 Steps)')
    axes[1].set_xlabel('pIC50')
    axes[1].legend()

    # C) Tanimoto vs TL
    if max_tans:
        axes[2].hist(max_tans, bins=30, color='#9b59b6', alpha=0.75, edgecolor='white')
        axes[2].axvline(mt, color='#111', ls='--', lw=2, label=f'Mean: {mt:.3f}')
        axes[2].set_title(f'Tanimoto vs TL (Copy Rate: {cr:.1%})')
        axes[2].set_xlabel('Max Tanimoto')
        axes[2].legend()

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f'[-] Updated live plot at Step {max_step}...')
    time.sleep(10)
