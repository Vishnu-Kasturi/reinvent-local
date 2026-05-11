"""
qsar3.py  -  RF x XGBoost Ensemble + Extended Fingerprint Suite
================================================================
Same ensemble architecture as qsar.py (RF_balanced, RF_extreme,
XGB_conservative, XGB_aggressive + Ridge meta-learner) but trained
on enriched features:

  Physicochemical (12): MolWt, ExactMW, MolLogP, NumHDonors,
    NumHAcceptors, TPSA, NumRotatableBonds, NumRings,
    NumAromaticRings, HeavyAtoms, FracCSP3, Stereocenters
  ECFP4  : Morgan r=2, 2048-bit (AllChem.GetMorganFingerprintAsBitVect)
  MACCS  : 167-bit substructure keys
  RDKit  : Topological path FP, 2048-bit
  FCFP4  : Feature-based Morgan r=2, 2048-bit (useFeatures=True)
  Total  : 6,323 features

Input  : data_splits/train.csv, val.csv, test.csv
Output : models/*_qsar3.joblib, models/meta_ridge_qsar3.joblib
         models/physchem_scaler_qsar3.pkl
         results/evaluation_report_qsar3.txt
         results/feature_importance_qsar3.csv
         results/evaluation_plots_qsar3.png
"""

import os, warnings, pickle
from time import time

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import RDKFingerprint

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, f1_score, classification_report, roc_curve,
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── CONFIG ────────────────────────────────────────────────────
DATA_DIR      = "data_splits"
MODEL_DIR     = "models"
RESULTS_DIR   = "results"
ACTIVE_THRESH = 6.0
RANDOM_SEED   = 42
N_JOBS        = -1
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── FEATURE NAMES ─────────────────────────────────────────────
PHYSCHEM_COLS = [
    "mw","exact_mw","logp","hbd","hba",
    "tpsa","rot_bonds","rings","arom_rings",
    "heavy_atoms","frac_csp3","stereo",
]
ECFP4_COLS = [f"ecfp4_{i}" for i in range(2048)]
RDKFP_COLS = [f"rdkfp_{i}" for i in range(2048)]
ALL_FEATURE_COLS = PHYSCHEM_COLS + ECFP4_COLS + RDKFP_COLS   # 4,108 total


# ── FEATURE COMPUTATION ───────────────────────────────────────
def safe_mol(smi):
    try:
        m = Chem.MolFromSmiles(str(smi))
        return m
    except Exception:
        return None


def compute_features(mol):
    pc = [
        Descriptors.MolWt(mol),
        Descriptors.ExactMolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcFractionCSP3(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    ]
    ecfp4 = list(map(int, AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048).ToBitString()))
    rdkfp = list(map(int, RDKFingerprint(mol, fpSize=2048).ToBitString()))
    return pc + ecfp4 + rdkfp


def build_X(df_raw):
    smiles_col = "smiles" if "smiles" in df_raw.columns else df_raw.columns[0]
    rows, valid_idx = [], []
    for i, smi in enumerate(df_raw[smiles_col]):
        mol = safe_mol(smi)
        if mol is not None:
            rows.append(compute_features(mol))
            valid_idx.append(i)
    feat = pd.DataFrame(rows, columns=ALL_FEATURE_COLS, index=df_raw.index[valid_idx])
    return feat, df_raw.iloc[valid_idx]


# ── STEP 1: LOAD & FEATURISE ──────────────────────────────────
print("=" * 65)
print("STEP 1 - Loading splits and computing extended features")
print("=" * 65)

for split in ["train", "val", "test"]:
    print(f"  Processing {split}...")
    raw = pd.read_csv(os.path.join(DATA_DIR, f"{split}.csv"))
    feat, meta = build_X(raw)
    feat.insert(0, "pic50", meta["pic50"].values)
    if "zone" in meta.columns:
        feat.insert(1, "zone", meta["zone"].values)
    if split == "train":
        train_df = feat
    elif split == "val":
        val_df = feat
    else:
        test_df = feat

print(f"  Train {len(train_df):,} | Val {len(val_df):,} | Test {len(test_df):,}")
print(f"  Feature dim: {len(ALL_FEATURE_COLS):,}")

# ── STEP 2: SCALE PHYSICOCHEMICAL ─────────────────────────────
print("\nSTEP 2 - Scaling physicochemical features")
scaler = StandardScaler()
train_df[PHYSCHEM_COLS] = scaler.fit_transform(train_df[PHYSCHEM_COLS])
val_df[PHYSCHEM_COLS]   = scaler.transform(val_df[PHYSCHEM_COLS])
test_df[PHYSCHEM_COLS]  = scaler.transform(test_df[PHYSCHEM_COLS])

with open(os.path.join(MODEL_DIR, "physchem_scaler_qsar3.pkl"), "wb") as f:
    pickle.dump(scaler, f)

X_train = train_df[ALL_FEATURE_COLS].values.astype(np.float32)
y_train = train_df["pic50"].values
X_val   = val_df[ALL_FEATURE_COLS].values.astype(np.float32)
y_val   = val_df["pic50"].values
X_test  = test_df[ALL_FEATURE_COLS].values.astype(np.float32)
y_test  = test_df["pic50"].values
zone_test = test_df["zone"].values if "zone" in test_df.columns else np.array(["bulk"]*len(test_df))
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

# ── STEP 3: BASE MODELS ───────────────────────────────────────
print("\nSTEP 3 - Training base models")

base_models = {
    "RF_balanced_qsar3": RandomForestRegressor(
        n_estimators=500, max_depth=20, min_samples_split=4,
        min_samples_leaf=2, max_features="sqrt",
        n_jobs=N_JOBS, random_state=RANDOM_SEED,
    ),
    "RF_extreme_qsar3": RandomForestRegressor(
        n_estimators=600, max_depth=12, min_samples_split=2,
        min_samples_leaf=1, max_features=0.4, bootstrap=True,
        oob_score=True, n_jobs=N_JOBS, random_state=RANDOM_SEED+1,
    ),
    "XGB_conservative_qsar3": XGBRegressor(
        n_estimators=1000, learning_rate=0.02, max_depth=6,
        subsample=0.8, colsample_bytree=0.7, colsample_bylevel=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3, gamma=0.05,
        tree_method="hist", n_jobs=N_JOBS, random_state=RANDOM_SEED,
        early_stopping_rounds=50, eval_metric="rmse", verbosity=0,
    ),
    "XGB_aggressive_qsar3": XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=8,
        subsample=0.7, colsample_bytree=0.5, colsample_bylevel=0.6,
        reg_alpha=1.0, reg_lambda=0.5, min_child_weight=1, gamma=0.1,
        tree_method="hist", n_jobs=N_JOBS, random_state=RANDOM_SEED+2,
        early_stopping_rounds=50, eval_metric="rmse", verbosity=0,
    ),
}

trained_models, val_preds, test_preds = {}, {}, {}

for name, model in base_models.items():
    t0 = time()
    print(f"\n  [{name}]")
    if "XGB" in name:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print(f"    Best iter: {model.best_iteration}")
    else:
        model.fit(X_train, y_train)
        if hasattr(model, "oob_score_"):
            print(f"    OOB R²: {model.oob_score_:.4f}")

    vp = model.predict(X_val)
    tp = model.predict(X_test)
    print(f"    Val R²: {r2_score(y_val, vp):.4f}  RMSE: {mean_squared_error(y_val,vp)**0.5:.4f}  ({time()-t0:.0f}s)")
    trained_models[name] = model
    val_preds[name]  = vp
    test_preds[name] = tp
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))

# ── STEP 4: META-LEARNER ─────────────────────────────────────
print("\nSTEP 4 - Ridge meta-learner")
val_stack  = np.column_stack([val_preds[n]  for n in base_models])
test_stack = np.column_stack([test_preds[n] for n in base_models])
meta = Ridge(alpha=1.0)
meta.fit(val_stack, y_val)
y_pred = meta.predict(test_stack)
print(f"  Coefs: {dict(zip(base_models, meta.coef_.round(4)))}")
joblib.dump(meta, os.path.join(MODEL_DIR, "meta_ridge_qsar3.joblib"))

# ── HELPERS ──────────────────────────────────────────────────
def reg_metrics(yt, yp):
    return r2_score(yt,yp), mean_squared_error(yt,yp)**0.5, mean_absolute_error(yt,yp)

def clf_metrics(yt, yp, thresh=ACTIVE_THRESH):
    tb = (yt>=thresh).astype(int)
    pb = (yp>=thresh).astype(int)
    pp = np.clip((yp-(thresh-4))/4, 0, 1)
    f1 = f1_score(tb, pb, zero_division=0)
    try: auc = roc_auc_score(tb, pp)
    except: auc = float("nan")
    return f1, auc, tb, pp

# ── STEP 5: EVALUATION ───────────────────────────────────────
print("\nSTEP 5 - Test-set evaluation")
r2,rmse,mae = reg_metrics(y_test, y_pred)
f1,auc,y_true_bin,y_pred_prob = clf_metrics(y_test, y_pred)
print(f"  Ensemble  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

report = [
    "qsar3 - RF×XGB Ensemble + Extended Fingerprints - Evaluation Report",
    "="*65,
    f"Features: {len(ALL_FEATURE_COLS):,}  (12 PhysChem + 2048 ECFP4 + 2048 RDKit Topo)",
    f"Active threshold: pIC50 = {ACTIVE_THRESH}",
    "",
    "Global Test Metrics:",
    f"  {'Model':<28} {'R2':>7} {'RMSE':>7} {'MAE':>7} {'F1':>7} {'AUC':>7}",
]

all_preds = dict(test_preds)
all_preds["Ensemble"] = y_pred

for name, yp in all_preds.items():
    r2_,rm,ma = reg_metrics(y_test, yp)
    f1_,au,_,_ = clf_metrics(y_test, yp)
    row = f"  {name:<28} {r2_:>7.4f} {rm:>7.4f} {ma:>7.4f} {f1_:>7.4f} {au:>7.4f}"
    print(row); report.append(row)

report += ["","Per-Zone Metrics (Ensemble)", "-"*65,
           f"  {'Zone':<10}{'N':>6}{'R2':>8}{'RMSE':>8}{'MAE':>8}"]
print(f"\n  Per-zone:")
for zone in ["low","bulk","high","all"]:
    mask = np.ones(len(y_test),bool) if zone=="all" else (zone_test==zone)
    if mask.sum()<2: continue
    r2_,rm,ma = reg_metrics(y_test[mask], y_pred[mask])
    row = f"  {zone:<10}{mask.sum():>6}{r2_:>8.4f}{rm:>8.4f}{ma:>8.4f}"
    print(row); report.append(row)

cr = classification_report(y_true_bin,(y_pred>=ACTIVE_THRESH).astype(int),
                            target_names=["Inactive","Active"])
print(cr); report += ["",f"Classification (threshold={ACTIVE_THRESH})",cr]

# ── STEP 6: CROSS-VALIDATION ─────────────────────────────────
print("STEP 6 - 5-fold CV (XGB proxy)")
cv_m = XGBRegressor(n_estimators=500,learning_rate=0.05,max_depth=6,
                    subsample=0.8,colsample_bytree=0.7,tree_method="hist",
                    n_jobs=N_JOBS,random_state=RANDOM_SEED,verbosity=0)
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_r2   = cross_val_score(cv_m, X_trainval, y_trainval, cv=kf, scoring="r2", n_jobs=N_JOBS)
cv_rmse = -cross_val_score(cv_m, X_trainval, y_trainval, cv=kf,
                            scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
print(f"  CV R²={cv_r2.mean():.4f}±{cv_r2.std():.4f}  RMSE={cv_rmse.mean():.4f}±{cv_rmse.std():.4f}")
report += ["","5-Fold CV (XGB proxy on train+val)",
           f"  R²  : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}",
           f"  RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}"]

# ── STEP 7: FEATURE IMPORTANCE ───────────────────────────────
print("STEP 7 - Feature importance")
imps = np.mean([
    trained_models["RF_balanced_qsar3"].feature_importances_,
    trained_models["RF_extreme_qsar3"].feature_importances_,
    trained_models["XGB_conservative_qsar3"].feature_importances_,
    trained_models["XGB_aggressive_qsar3"].feature_importances_,
], axis=0)
imp_df = pd.DataFrame({"feature":ALL_FEATURE_COLS,"avg_importance":imps,
    "RF_balanced":trained_models["RF_balanced_qsar3"].feature_importances_,
    "RF_extreme":trained_models["RF_extreme_qsar3"].feature_importances_,
    "XGB_conservative":trained_models["XGB_conservative_qsar3"].feature_importances_,
    "XGB_aggressive":trained_models["XGB_aggressive_qsar3"].feature_importances_,
}).sort_values("avg_importance",ascending=False)
imp_df.to_csv(os.path.join(RESULTS_DIR,"feature_importance_qsar3.csv"),index=False)

# Group summary
groups = {"ECFP4": ECFP4_COLS, "RDKit Topo": RDKFP_COLS, "PhysChem": PHYSCHEM_COLS}
report += ["", "Feature Group Importance (avg across 4 models):"]
for g, cols in groups.items():
    v = imp_df[imp_df["feature"].isin(cols)]["avg_importance"].mean()
    line = f"  {g:<12}: {v:.6f}"
    print(line); report.append(line)

# ── STEP 8: PLOTS ─────────────────────────────────────────────
print("STEP 8 - Plots")
zone_color = {"low":"#BA7517","bulk":"#1D9E75","high":"#E24B4A"}
colors = [zone_color.get(z,"#888") for z in zone_test]

fig, axes = plt.subplots(2,3,figsize=(20,12))
fig.suptitle("qsar3 - RF×XGB Ensemble + Extended Fingerprints\n"
             "(ECFP4 + MACCS + RDKit Topo + FCFP4 + 12 PhysChem)",
             fontsize=13,fontweight="bold")

# Predicted vs Actual
ax=axes[0,0]
ax.scatter(y_test,y_pred,c=colors,alpha=0.4,s=8,linewidths=0)
mn,mx=min(y_test.min(),y_pred.min()),max(y_test.max(),y_pred.max())
ax.plot([mn,mx],[mn,mx],"k--",lw=1)
ax.axvline(ACTIVE_THRESH,color="gray",lw=0.8,ls=":")
ax.axhline(ACTIVE_THRESH,color="gray",lw=0.8,ls=":")
ax.set_xlabel("Actual pIC50"); ax.set_ylabel("Predicted pIC50")
ax.set_title(f"Predicted vs Actual (R²={r2:.3f}, RMSE={rmse:.3f})")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c,label=z) for z,c in zone_color.items()],fontsize=8)

# Residuals
ax=axes[0,1]
res=y_test-y_pred
ax.scatter(y_pred,res,c=colors,alpha=0.4,s=8,linewidths=0)
ax.axhline(0, color="k", lw=1, ls="--")
ax.set_xlabel("Predicted pIC50"); ax.set_ylabel("Residual")
ax.set_title("Residuals vs Predicted")

# ROC
ax=axes[0,2]
fpr,tpr,_=roc_curve(y_true_bin,y_pred_prob)
ax.plot(fpr,tpr,color="#534AB7",lw=2,label=f"AUC={auc:.3f}")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curve"); ax.legend(fontsize=9)

# Top-20 features
ax=axes[1,0]
top20=imp_df.head(20)
ax.barh(top20["feature"][::-1],top20["avg_importance"][::-1],color="#534AB7",edgecolor="none")
ax.set_xlabel("Avg importance (across 4 models)")
ax.set_title("Top 20 Features")
ax.tick_params(axis="y",labelsize=7)

# PhysChem importance bar chart
ax=axes[1,1]
pc_imp = imp_df[imp_df["feature"].isin(PHYSCHEM_COLS)].set_index("feature")["avg_importance"].reindex(PHYSCHEM_COLS)
ax.barh(pc_imp.index[::-1], pc_imp.values[::-1], color="#1D9E75", edgecolor="none")
ax.set_xlabel("Avg importance"); ax.set_title("Physicochemical Feature Importance")
ax.tick_params(axis="y", labelsize=8)

# Model R² comparison
ax=axes[1,2]
names=list(all_preds.keys())
r2s=[r2_score(y_test,all_preds[n]) for n in names]
ax.bar(names,r2s,color=["#9FE1CB","#5DCAA5","#AFA9EC","#7F77DD","#E24B4A"],edgecolor="none")
ax.set_ylim(0,1); ax.set_ylabel("R² on test set"); ax.set_title("Model R² Comparison")
for i,(n,v) in enumerate(zip(names,r2s)):
    ax.text(i,v+0.01,f"{v:.3f}",ha="center",fontsize=8)
plt.xticks(rotation=20,ha="right")

plt.tight_layout()
plot_path=os.path.join(RESULTS_DIR,"evaluation_plots_qsar3.png")
plt.savefig(plot_path,dpi=150,bbox_inches="tight"); plt.close()
print(f"  Saved → {plot_path}")

# ── SAVE REPORT ───────────────────────────────────────────────
report_path=os.path.join(RESULTS_DIR,"evaluation_report_qsar3.txt")
with open(report_path,"w",encoding="utf-8") as f: f.write("\n".join(report))
print(f"  Saved → {report_path}")

# ── INFERENCE SNIPPET ─────────────────────────────────────────
snippet = '''
# qsar3 inference snippet
import joblib, pickle, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import RDKFingerprint

models = {n: joblib.load(f"models/{n}.joblib") for n in
          ["RF_balanced_qsar3","RF_extreme_qsar3",
           "XGB_conservative_qsar3","XGB_aggressive_qsar3"]}
meta = joblib.load("models/meta_ridge_qsar3.joblib")
with open("models/physchem_scaler_qsar3.pkl","rb") as f:
    scaler = pickle.load(f)

PHYSCHEM_COLS = ["mw","exact_mw","logp","hbd","hba","tpsa","rot_bonds",
                 "rings","arom_rings","heavy_atoms","frac_csp3","stereo"]

def predict_pic50(smiles_list):
    import pandas as pd
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        pc = [Descriptors.MolWt(mol), Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol),
              rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
              Descriptors.TPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
              rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
              mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcFractionCSP3(mol),
              len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))]
        ecfp4 = list(map(int, AllChem.GetMorganFingerprintAsBitVect(mol,2,2048).ToBitString()))
        rdkfp = list(map(int, RDKFingerprint(mol, fpSize=2048).ToBitString()))
        rows.append(pc + ecfp4 + rdkfp)
    ALL_COLS = PHYSCHEM_COLS + [f"ecfp4_{i}" for i in range(2048)] + [f"rdkfp_{i}" for i in range(2048)]
    df = pd.DataFrame(rows, columns=ALL_COLS)
    df[PHYSCHEM_COLS] = scaler.transform(df[PHYSCHEM_COLS])
    X = df.values.astype("float32")
    stack = np.column_stack([m.predict(X) for m in models.values()])
    return meta.predict(stack)
'''
with open(os.path.join(MODEL_DIR,"inference_snippet_qsar3.py"),"w",encoding="utf-8") as f:
    f.write(snippet)

print("\n" + "="*65)
print("qsar3 complete.")
print(f"  Models  → {MODEL_DIR}/")
print(f"  Results → {RESULTS_DIR}/")
print("="*65)
