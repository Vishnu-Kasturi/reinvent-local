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

# Wait for RL to finish (CSV stable for 10s)
RL_CSV = f'{RES}/pd1_pdl1_rl_toml_1.csv'
print('[*] Waiting for RL to complete...')
last_sz=-1; stable=0
while True:
    if os.path.exists(RL_CSV):
        sz=os.path.getsize(RL_CSV)
        stable = stable+1 if sz==last_sz else 0
        last_sz=sz
        if stable>=2: print(f'[+] CSV stable at {sz/1e6:.1f}MB'); break
    time.sleep(5)
time.sleep(5)

# Load
df_rl = pd.read_csv(RL_CSV).dropna(subset=[PIC50])
df_raw = pd.read_csv(RAW_CSV, sep='\t')
df_raw.columns=[c.strip().lower() for c in df_raw.columns]
raw_smi=df_raw['smiles'].dropna().tolist(); raw_pic50=df_raw['pic50'].dropna().values
max_st=df_rl['step'].max(); cutoff=int(max_st*0.8)
df_opt=df_rl[df_rl['step']>cutoff].copy()
df_opt['pic50']=df_opt[PIC50]
print(f'Total rows: {len(df_rl)} | Max step: {max_st} | Optimized: {len(df_opt)}')

# Top hits
hits=df_opt[df_opt['pic50']>8.5].drop_duplicates('SMILES').sort_values('pic50',ascending=False)
print(f'Hits >8.5: {len(hits)}')
hits.head(30)[['SMILES','pic50',SA_C]].to_csv(f'{RES}/onlypic50_top30.csv',index=False)
print(hits.head(5)[['SMILES','pic50']].to_string(index=False))

# Tanimoto vs Original
print('[*] Tanimoto...')
raw_fps=get_fps(raw_smi); opt_fps=get_fps(df_opt['SMILES'].tolist())
max_tans=[]; exact=0
for fp in opt_fps:
    sims=DataStructs.BulkTanimotoSimilarity(fp,raw_fps); ms=max(sims); max_tans.append(ms)
    if ms>=0.999: exact+=1
mt=np.mean(max_tans); mdt=np.median(max_tans); cr=exact/len(opt_fps)
print(f'Mean={mt:.3f} Median={mdt:.3f} CopyRate={cr:.1%}')

fig,ax=plt.subplots(figsize=(9,5))
ax.hist(max_tans,bins=35,color='#5b9bd5',alpha=0.78,edgecolor='white',lw=0.3)
ax.axvline(mt,color='#111',ls='--',lw=1.8,label=f'Mean:{mt:.3f}'); ax.axvline(mdt,color='#555',ls=':',lw=1.8,label=f'Med:{mdt:.3f}')
ax.axvspan(0.999,1.02,color='red',alpha=0.12,label=f'Copies:{cr:.1%}')
ax.set_title('PD1-PDL1 RL — Tanimoto vs Raw Dataset',fontsize=12,weight='bold')
ax.set_xlabel('Max Tanimoto'); ax.set_ylabel('Count'); ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(f'{RES}/onlypic50_tanimoto.png',dpi=150,bbox_inches='tight'); plt.close()
print('Saved tanimoto')

# Step KDE
n=10; be=np.linspace(1,max_st+1,n+1); colors=cm.plasma(np.linspace(0.05,0.95,n)); xs=np.linspace(4,11,300)
fig2,axes2=plt.subplots(2,5,figsize=(22,8))
fig2.suptitle('PD1-PDL1 RL — pIC50 KDE per Step Window',fontsize=13,weight='bold',y=1.01)
for i,ax in enumerate(axes2.flatten()):
    lo=be[i]; hi=be[i+1]; vals=df_rl.loc[(df_rl['step']>=lo)&(df_rl['step']<hi),PIC50].dropna().values
    kde_r=gaussian_kde(raw_pic50); ax.fill_between(xs,kde_r(xs),alpha=0.25,color='#2ecc71',label='Original'); ax.plot(xs,kde_r(xs),color='#2ecc71',lw=1.5)
    if len(vals)>5:
        krl=gaussian_kde(vals); ax.fill_between(xs,krl(xs),alpha=0.5,color=colors[i]); ax.plot(xs,krl(xs),color=colors[i],lw=2,label=f'RL n={len(vals)}')
    ax.axvline(8.5,color='gold',ls='--',lw=1.2); mv=np.mean(vals) if len(vals) else 0; ph=(vals>8.5).mean()*100 if len(vals) else 0
    ax.set_title(f'Steps {int(lo)}-{int(hi-1)}',fontsize=9,weight='bold'); ax.set_xlabel('pIC50',fontsize=8); ax.set_ylabel('Density',fontsize=8)
    ax.tick_params(labelsize=7); ax.set_xlim(4,11); ax.legend(fontsize=6.5); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.text(0.97,0.97,f'Mean={mv:.2f}\n>8.5:{ph:.1f}%',transform=ax.transAxes,fontsize=7.5,va='top',ha='right',bbox=dict(boxstyle='round,pad=0.3',fc='black',alpha=0.5,ec='none'),color='white')
plt.tight_layout(); plt.savefig(f'{RES}/onlypic50_step_kde.png',dpi=150,bbox_inches='tight'); plt.close()
print('Saved step KDE')

# Overall KDE
print('[*] Overall KDE...')
raw_mw,raw_qed,raw_sa=get_props(raw_smi); opt_mw,opt_qed,opt_sa=get_props(df_opt['SMILES'].tolist())
pal={'Original Dataset':'#2ecc71','RL Generated':'#e74c3c'}
df_p=pd.concat([pd.DataFrame({'MW':raw_mw,'QED':raw_qed,'SA':raw_sa,'pic50':raw_pic50[:len(raw_mw)].tolist(),'Source':'Original Dataset'}),
                pd.DataFrame({'MW':opt_mw,'QED':opt_qed,'SA':opt_sa,'pic50':df_opt['pic50'].tolist(),'Source':'RL Generated'})],ignore_index=True)
fig3,axes3=plt.subplots(1,4,figsize=(18,5))
fig3.suptitle('PD1-PDL1 RL — Generated vs Original Dataset Distributions',fontsize=13,weight='bold')
for ax,col,title,vl in zip(axes3,['pic50','MW','QED','SA'],['pIC50','Mol. Weight','QED Score','SA Score'],[8.5,500,0.6,4.0]):
    sns.kdeplot(data=df_p,x=col,hue='Source',common_norm=False,fill=True,alpha=0.35,ax=ax,palette=pal)
    ax.axvline(vl,color='gold',ls='--',lw=1.3); ax.set_title(title,weight='bold'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(f'{RES}/onlypic50_kde.png',dpi=150,bbox_inches='tight'); plt.close()
print('Saved overall KDE')

print('=== ALL DONE ===')
