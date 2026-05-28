import os, sys, pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.cm as cm, seaborn as sns
from scipy.stats import gaussian_kde
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import Descriptors, QED
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
RDLogger.DisableLog('rdApp.*')

RES = 'results'
df_rl = pd.read_csv(f'{RES}/sol_focus_rl_1.csv').dropna(subset=['PD1PDL1Sol_raw (raw)'])
max_st = df_rl['step'].max()
cutoff = int(max_st * 0.8)
df_opt = df_rl[df_rl['step'] > cutoff].copy()
df_opt['logS'] = df_opt['PD1PDL1Sol_raw (raw)']

# Load original SOL data
df_raw_sol = pd.read_csv('Preprocess/Data_pd1_pdl1/pd1_pdl1_sol.csv')
raw_sol = df_raw_sol['Y'].dropna().values

# Step KDE for logS (with original curve)
n=10; be=np.linspace(1, max_st+1, n+1); colors=cm.plasma(np.linspace(0.05, 0.95, n))
xs = np.linspace(-12, 6, 300)
fig2, axes2 = plt.subplots(2, 5, figsize=(22, 8))
fig2.suptitle('PD1-PDL1 Sol Focus RL — logS KDE per Step Window', fontsize=13, weight='bold', y=1.01)

kde_r = gaussian_kde(raw_sol)
for i, ax in enumerate(axes2.flatten()):
    lo = be[i]; hi = be[i+1]
    vals = df_rl.loc[(df_rl['step'] >= lo) & (df_rl['step'] < hi), 'PD1PDL1Sol_raw (raw)'].dropna().values
    
    # Plot original curve
    ax.fill_between(xs, kde_r(xs), alpha=0.25, color='#2ecc71', label='Original Sol Dataset')
    ax.plot(xs, kde_r(xs), color='#2ecc71', lw=1.5)
    
    if len(vals) > 5:
        krl = gaussian_kde(vals)
        ax.fill_between(xs, krl(xs), alpha=0.5, color=colors[i])
        ax.plot(xs, krl(xs), color=colors[i], lw=2, label=f'RL n={len(vals)}')
    
    ax.axvline(-3.0, color='gold', ls='--', lw=1.2)
    ax.set_title(f'Steps {int(lo)}-{int(hi-1)}', fontsize=9, weight='bold')
    ax.set_xlim(-10, 4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=6.5)

plt.tight_layout()
plt.savefig(f'{RES}/sol_focus_step_kde.png', dpi=150, bbox_inches='tight')
plt.close()

# Overall KDE
def get_props(sl):
    mw,qe,sa=[],[],[]
    for s in sl:
        m=Chem.MolFromSmiles(str(s))
        if m: mw.append(Descriptors.MolWt(m)); qe.append(QED.qed(m)); sa.append(sascorer.calculateScore(m) if sascorer else np.nan)
        else: mw.append(np.nan); qe.append(np.nan); sa.append(np.nan)
    return mw,qe,sa

df_raw_pic50 = pd.read_csv('Preprocess/Data_pd1_pdl1/pd1_pdl1_pic50_raw.csv', sep='\t')
df_raw_pic50.columns = [c.strip().lower() for c in df_raw_pic50.columns]
raw_smi = df_raw_pic50['smiles'].dropna().tolist()
raw_mw, raw_qed, raw_sa = get_props(raw_smi)
opt_mw, opt_qed, opt_sa = get_props(df_opt['SMILES'].tolist())

pal={'Original Dataset':'#2ecc71','RL Generated':'#e74c3c'}
df_p=pd.concat([pd.DataFrame({'MW':raw_mw,'QED':raw_qed,'SA':raw_sa,'Source':'Original Dataset'}),
                pd.DataFrame({'MW':opt_mw,'QED':opt_qed,'SA':opt_sa,'Source':'RL Generated'})],ignore_index=True)

fig3, axes3 = plt.subplots(1, 4, figsize=(18, 5))
fig3.suptitle('PD1-PDL1 Sol Focus RL — Generated vs Original Dataset Distributions', fontsize=13, weight='bold')

for ax, col, title, vl in zip(axes3[:3], ['MW','QED','SA'], ['Mol. Weight','QED Score','SA Score'], [500, 0.6, 4.0]):
    sns.kdeplot(data=df_p, x=col, hue='Source', common_norm=False, fill=True, alpha=0.35, ax=ax, palette=pal)
    ax.axvline(vl, color='gold', ls='--', lw=1.3)
    ax.set_title(title, weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# LogS subplot
ax = axes3[3]
# Original Sol
ax.fill_between(xs, kde_r(xs), alpha=0.25, color='#2ecc71', label='Original Dataset')
ax.plot(xs, kde_r(xs), color='#2ecc71', lw=1.5)

# RL Generated Sol
data_rl = df_opt['logS'].dropna()
try:
    krl = gaussian_kde(data_rl)
    ax.fill_between(xs, krl(xs), alpha=0.4, color='#e74c3c', label='RL Generated')
    ax.plot(xs, krl(xs), color='#e74c3c', lw=2)
except: pass

ax.axvline(-3.0, color='gold', ls='--', lw=1.3)
ax.set_title('Solubility (logS)', weight='bold')
ax.set_xlim(-10, 4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()

plt.tight_layout()
plt.savefig(f'{RES}/sol_focus_kde.png', dpi=150, bbox_inches='tight')
plt.close()
print('DONE')
