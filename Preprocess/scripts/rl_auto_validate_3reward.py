import os, sys, time, glob, pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.cm as cm, seaborn as sns
from scipy.stats import gaussian_kde
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
RDLogger.DisableLog('rdApp.*')

RES     = 'results'
RAW_CSV = 'Preprocess/Data_pd1_pdl1/pd1_pdl1_pic50_raw.csv'
TL_CSV  = f'{RES}/pd1_pdl1_tl_sample_e50.csv'
PIC50   = 'PD1PDL1pIC50_raw (raw)'
SOL     = 'PD1PDL1Sol_raw (raw)'
SA_C    = 'SAScore (raw)'

def get_fps(sl):
    out=[]
    for s in sl:
        m=Chem.MolFromSmiles(str(s))
        if m: out.append(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048))
    return out

def get_props(sl):
    mw,qe,sa=[],[],[]
    for s in sl:
        m=Chem.MolFromSmiles(str(s))
        if m: mw.append(Descriptors.MolWt(m)); qe.append(QED.qed(m)); sa.append(sascorer.calculateScore(m) if sascorer else np.nan)
        else: mw.append(np.nan); qe.append(np.nan); sa.append(np.nan)
    return mw,qe,sa

RL_CSV = f'{RES}/pd1_pdl1_rl_toml_1.csv'
print('[*] Waiting for RL to complete...')
last_sz=-1; stable=0
while True:
    if os.path.exists(RL_CSV):
        sz=os.path.getsize(RL_CSV)
        stable = stable+1 if sz==last_sz else 0
        last_sz=sz
        if stable>=2: 
            try:
                df_rl = pd.read_csv(RL_CSV).dropna(subset=[PIC50, SOL])
                if len(df_rl) > 100:
                    print(f'[+] CSV stable at {sz/1e6:.1f}MB'); break
            except pd.errors.EmptyDataError:
                pass
    time.sleep(10)
df_raw = pd.read_csv(RAW_CSV, sep='\t')
df_raw.columns=[c.strip().lower() for c in df_raw.columns]
raw_smi=df_raw['smiles'].dropna().tolist(); raw_pic50=df_raw['pic50'].dropna().values

max_st=df_rl['step'].max(); cutoff=int(max_st*0.8)
df_opt=df_rl[df_rl['step']>cutoff].copy()
df_opt['pic50']=df_opt[PIC50]; df_opt['logS']=df_opt[SOL]

# Top hits (pIC50 > 8.5 AND logS > -3)
hits = df_opt[(df_opt['pic50']>8.5) & (df_opt['logS']>-3)].drop_duplicates('SMILES').sort_values('pic50',ascending=False)
hits.head(30)[['SMILES','pic50','logS',SA_C]].to_csv(f'{RES}/pd1_pdl1_3reward_top30.csv',index=False)

# Properties
raw_mw,raw_qed,raw_sa=get_props(raw_smi)
opt_mw,opt_qed,opt_sa=get_props(df_opt['SMILES'].tolist())
opt_sol = df_opt['logS'].values

# 1. 5-Panel KDE
fig, axes = plt.subplots(1, 5, figsize=(24, 5))
fig.suptitle('PD1-PDL1 3-Reward RL — All Properties KDE', fontsize=15, weight='bold')

sns.kdeplot(raw_pic50, ax=axes[0], color='#2ecc71', fill=True, alpha=0.35, label='Original')
sns.kdeplot(df_opt['pic50'], ax=axes[0], color='#e74c3c', fill=True, alpha=0.35, label='RL Generated')
axes[0].axvline(8.5, color='gold', ls='--', lw=1.5); axes[0].set_title('pIC50', weight='bold'); axes[0].legend()

sns.kdeplot(raw_mw, ax=axes[1], color='#2ecc71', fill=True, alpha=0.35)
sns.kdeplot(opt_mw, ax=axes[1], color='#e74c3c', fill=True, alpha=0.35)
axes[1].axvline(500, color='gold', ls='--', lw=1.5); axes[1].set_title('Molecular Weight', weight='bold')

sns.kdeplot(raw_qed, ax=axes[2], color='#2ecc71', fill=True, alpha=0.35)
sns.kdeplot(opt_qed, ax=axes[2], color='#e74c3c', fill=True, alpha=0.35)
axes[2].axvline(0.6, color='gold', ls='--', lw=1.5); axes[2].set_title('QED Score', weight='bold')

sns.kdeplot(raw_sa, ax=axes[3], color='#2ecc71', fill=True, alpha=0.35)
sns.kdeplot(opt_sa, ax=axes[3], color='#e74c3c', fill=True, alpha=0.35)
axes[3].axvline(4.0, color='gold', ls='--', lw=1.5); axes[3].set_title('Synthetic Accessibility', weight='bold')

sns.kdeplot(opt_sol, ax=axes[4], color='#e74c3c', fill=True, alpha=0.35, label='RL Generated')
axes[4].axvline(-3.0, color='gold', ls='--', lw=1.5, label='logS = -3'); axes[4].set_title('Solubility (logS)', weight='bold'); axes[4].legend()

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(f'{RES}/pd1_pdl1_3reward_kde_all5.png', dpi=150, bbox_inches='tight'); plt.close()

# 2. Tanimoto vs Original
opt_fps=get_fps(df_opt['SMILES'].tolist())
raw_fps=get_fps(raw_smi)
max_tans=[]; exact=0
for fp in opt_fps:
    sims=DataStructs.BulkTanimotoSimilarity(fp,raw_fps); ms=max(sims); max_tans.append(ms)
    if ms>=0.999: exact+=1
mt=np.mean(max_tans); cr=exact/len(opt_fps)
fig2,ax=plt.subplots(figsize=(9,5))
ax.hist(max_tans,bins=35,color='#5b9bd5',alpha=0.78,edgecolor='white',lw=0.3)
ax.axvline(mt,color='#111',ls='--',lw=1.8,label=f'Mean:{mt:.3f}'); ax.axvspan(0.999,1.02,color='red',alpha=0.12,label=f'Copies:{cr:.1%}')
ax.set_title('PD1-PDL1 3-Reward RL — Tanimoto vs Raw Dataset',fontsize=12,weight='bold'); ax.legend()
plt.savefig(f'{RES}/pd1_pdl1_3reward_tanimoto_raw.png',dpi=150,bbox_inches='tight'); plt.close()

# 3. Tanimoto vs TL Epoch 50
df_tl = pd.read_csv(TL_CSV)
tl_fps=get_fps(df_tl['SMILES'].dropna().tolist())
max_tans=[]; exact=0
for fp in opt_fps:
    if not tl_fps: break
    sims=DataStructs.BulkTanimotoSimilarity(fp,tl_fps); ms=max(sims); max_tans.append(ms)
    if ms>=0.999: exact+=1
mt=np.mean(max_tans); cr=exact/len(opt_fps)
fig3,ax=plt.subplots(figsize=(9,5))
ax.hist(max_tans,bins=35,color='#9b59b6',alpha=0.78,edgecolor='white',lw=0.3)
ax.axvline(mt,color='#111',ls='--',lw=1.8,label=f'Mean:{mt:.3f}'); ax.axvspan(0.999,1.02,color='red',alpha=0.12,label=f'Copies:{cr:.1%}')
ax.set_title('PD1-PDL1 3-Reward RL — Tanimoto vs TL Dataset',fontsize=12,weight='bold'); ax.legend()
plt.savefig(f'{RES}/pd1_pdl1_3reward_tanimoto_tl.png',dpi=150,bbox_inches='tight'); plt.close()

print('ALL DONE!')
