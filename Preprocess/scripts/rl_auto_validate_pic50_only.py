import os, sys, time, glob, pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.cm as cm, seaborn as sns
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import AllChem, Descriptors
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
RDLogger.DisableLog('rdApp.*')

RES     = 'results'
RAW_CSV = 'Preprocess/Data_pd1_pdl1/pd1_pdl1_pic50_raw.csv'
PIC50   = 'PD1PDL1pIC50_raw (raw)'
SA_C    = 'SAScore (raw)'

def get_fps(sl):
    out=[]
    for s in sl:
        m=Chem.MolFromSmiles(str(s))
        if m: out.append(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048))
    return out

def get_props(sl):
    mw,sa=[],[]
    for s in sl:
        m=Chem.MolFromSmiles(str(s))
        if m: 
            mw.append(Descriptors.MolWt(m))
            try: sa.append(sascorer.calculateScore(m))
            except: sa.append(np.nan)
        else: mw.append(np.nan); sa.append(np.nan)
    return mw,sa

RL_CSV = f'{RES}/pd1_pdl1_pic50_only_rl_1.csv'
print('[*] Waiting for RL to complete...')
last_sz=-1; stable=0
while True:
    if os.path.exists(RL_CSV):
        sz=os.path.getsize(RL_CSV)
        stable = stable+1 if sz==last_sz else 0
        last_sz=sz
        if stable>=2: 
            try:
                df_rl = pd.read_csv(RL_CSV).dropna(subset=[PIC50])
                if len(df_rl) > 100:
                    print(f'[+] CSV stable at {sz/1e6:.1f}MB'); break
            except pd.errors.EmptyDataError:
                pass
    time.sleep(10)

df_raw = pd.read_csv(RAW_CSV, sep='\t')
df_raw.columns=[c.strip().lower() for c in df_raw.columns]
raw_smi=df_raw['smiles'].dropna().tolist()
raw_pic50=df_raw['pic50'].dropna().values

max_st = df_rl['step'].max()
cutoff = int(max_st * 0.8)
df_opt = df_rl[df_rl['step'] > cutoff].copy()
opt_pic50 = df_opt[PIC50].values
opt_smi = df_opt['SMILES'].tolist()

hits = df_opt[df_opt[PIC50] > 8.5].drop_duplicates('SMILES').sort_values(PIC50, ascending=False)
hits.head(30)[['SMILES', PIC50, SA_C]].to_csv(f'{RES}/pd1_pdl1_pic50_only_top30.csv', index=False)

raw_mw, raw_sa = get_props(raw_smi)
opt_mw, opt_sa = get_props(opt_smi)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('PD1-PDL1 RL (pIC50 Focus) vs Original Raw Dataset', fontsize=15, weight='bold')

sns.kdeplot(raw_pic50, ax=axes[0], color='#2ecc71', fill=True, alpha=0.35, label='Orig. Raw Dataset')
sns.kdeplot(opt_pic50, ax=axes[0], color='#3498db', fill=True, alpha=0.35, label='RL Generated')
axes[0].axvline(8.5, color='gold', ls='--', lw=1.5); axes[0].set_title('pIC50', weight='bold'); axes[0].legend()

sns.kdeplot(raw_mw, ax=axes[1], color='#2ecc71', fill=True, alpha=0.35)
sns.kdeplot(opt_mw, ax=axes[1], color='#3498db', fill=True, alpha=0.35)
axes[1].axvline(500, color='gold', ls='--', lw=1.5); axes[1].set_title('Molecular Weight', weight='bold')

sns.kdeplot(raw_sa, ax=axes[2], color='#2ecc71', fill=True, alpha=0.35)
sns.kdeplot(opt_sa, ax=axes[2], color='#3498db', fill=True, alpha=0.35)
axes[2].axvline(4.0, color='gold', ls='--', lw=1.5); axes[2].set_title('Synthetic Accessibility', weight='bold')

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{RES}/pd1_pdl1_pic50_only_kde.png', dpi=150, bbox_inches='tight')
plt.close()

opt_fps = get_fps(opt_smi)
raw_fps = get_fps(raw_smi)

max_tans=[]; exact=0
for fp in opt_fps:
    if not fp or not raw_fps: continue
    sims = DataStructs.BulkTanimotoSimilarity(fp, raw_fps)
    ms = max(sims)
    max_tans.append(ms)
    if ms >= 0.999: exact += 1

mt = np.mean(max_tans)
cr = exact / len(opt_fps)

fig2, ax = plt.subplots(figsize=(9,5))
ax.hist(max_tans, bins=35, color='#e67e22', alpha=0.78, edgecolor='white', lw=0.3)
ax.axvline(mt, color='#111', ls='--', lw=1.8, label=f'Mean: {mt:.3f}')
ax.axvspan(0.999, 1.02, color='red', alpha=0.12, label=f'Copies: {cr:.1%}')
ax.set_title('RL Molecules vs Original Raw Dataset — Tanimoto Similarity', fontsize=12, weight='bold')
ax.legend()
plt.savefig(f'{RES}/pd1_pdl1_pic50_only_tanimoto.png', dpi=150, bbox_inches='tight')
plt.close()

print('ALL DONE!')
