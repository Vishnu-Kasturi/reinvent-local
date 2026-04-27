"""
Script 2: RF × XGBoost Ensemble for pIC50 Regression
=======================================================
Input  : data_splits/train.csv, val.csv, test.csv (from Script 1)
Output : models/  (all 4 base models + meta-learner saved)
         results/evaluation_report.txt
         results/feature_importance.csv

Metrics:
  Regression  → R², RMSE, MAE (per-zone + global)
  Binarized   → F1, AUC-ROC  (threshold = 6.0 → active/inactive)

Requirements:
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import pickle
from time import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, f1_score, classification_report, roc_curve,
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR      = "data_splits"
MODEL_DIR     = "models"
RESULTS_DIR   = "results"
ACTIVE_THRESH = 6.0          # pIC50 threshold for F1 / AUC-ROC
RANDOM_SEED   = 42
N_JOBS        = -1           # use all CPU cores
# ─────────────────────────────────────────────

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════
print("=" * 65)
print("STEP 1 – Loading splits")
print("=" * 65)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

with open(os.path.join(DATA_DIR, "feature_cols.txt")) as f:
    FEATURE_COLS = [line.strip() for line in f if line.strip()]

print(f"  Train : {len(train_df):,}  |  Val : {len(val_df):,}  |  Test : {len(test_df):,}")
print(f"  Features : {len(FEATURE_COLS):,}")

# Drop synthetic SMOGN rows for feature matrix (they have __synthetic__ smiles)
train_df = train_df[train_df["smiles"] != "__synthetic__"].copy() \
           if "__synthetic__" in train_df.get("smiles", pd.Series()).values \
           else train_df

X_train = train_df[FEATURE_COLS].values.astype(np.float32)
y_train = train_df["pic50"].values

X_val   = val_df[FEATURE_COLS].values.astype(np.float32)
y_val   = val_df["pic50"].values

X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
y_test  = test_df["pic50"].values

zone_test = test_df.get("zone", pd.Series(["bulk"] * len(test_df))).values

# Combined train+val for final base-model fitting (after meta-learner is trained)
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

print(f"\n  pIC50 train range : [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"  pIC50 test range  : [{y_test.min():.2f}, {y_test.max():.2f}]")


# ══════════════════════════════════════════════
# STEP 2 – Define 4 base models
# ══════════════════════════════════════════════
print("\nSTEP 2 – Defining base models")

base_models = {

    # ── RF 1: balanced depth, moderate trees, strong at bulk ────────
    "RF_balanced": RandomForestRegressor(
        n_estimators   = 500,
        max_depth      = 20,
        min_samples_split = 4,
        min_samples_leaf  = 2,
        max_features   = "sqrt",
        n_jobs         = N_JOBS,
        random_state   = RANDOM_SEED,
    ),

    # ── RF 2: shallow + extra randomisation, targets extremes ───────
    "RF_extreme": RandomForestRegressor(
        n_estimators      = 600,
        max_depth         = 12,
        min_samples_split = 2,
        min_samples_leaf  = 1,
        max_features      = 0.4,        # wider random subspace
        bootstrap         = True,
        oob_score         = True,
        n_jobs            = N_JOBS,
        random_state      = RANDOM_SEED + 1,
    ),

    # ── XGB 1: conservative, low learning rate – high R² on bulk ───
    "XGB_conservative": XGBRegressor(
        n_estimators      = 1000,
        learning_rate     = 0.02,
        max_depth         = 6,
        subsample         = 0.8,
        colsample_bytree  = 0.7,
        colsample_bylevel = 0.7,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        min_child_weight  = 3,
        gamma             = 0.05,
        tree_method       = "hist",
        n_jobs            = N_JOBS,
        random_state      = RANDOM_SEED,
        early_stopping_rounds = 50,
        eval_metric       = "rmse",
        verbosity         = 0,
    ),

    # ── XGB 2: aggressive depth + L1, targets sparse FP features ───
    "XGB_aggressive": XGBRegressor(
        n_estimators      = 800,
        learning_rate     = 0.05,
        max_depth         = 8,
        subsample         = 0.7,
        colsample_bytree  = 0.5,
        colsample_bylevel = 0.6,
        reg_alpha         = 1.0,        # strong L1 for FP sparsity
        reg_lambda        = 0.5,
        min_child_weight  = 1,
        gamma             = 0.1,
        tree_method       = "hist",
        n_jobs            = N_JOBS,
        random_state      = RANDOM_SEED + 2,
        early_stopping_rounds = 50,
        eval_metric       = "rmse",
        verbosity         = 0,
    ),
}

print(f"  Models defined: {list(base_models.keys())}")


# ══════════════════════════════════════════════
# STEP 3 – Train base models on train split
#          (val used only as XGB early-stop set)
# ══════════════════════════════════════════════
print("\nSTEP 3 – Training base models")

trained_models = {}
val_preds      = {}   # out-of-fold predictions for stacking
test_preds     = {}

for name, model in base_models.items():
    t0 = time()
    print(f"\n  [{name}]")

    if "XGB" in name:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_iter = model.best_iteration
        print(f"    Best iteration : {best_iter}")
    else:
        model.fit(X_train, y_train)
        if hasattr(model, "oob_score_"):
            print(f"    OOB R²         : {model.oob_score_:.4f}")

    # Validate
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_r2   = r2_score(y_val, val_pred)
    val_rmse = mean_squared_error(y_val, val_pred) ** 0.5

    print(f"    Val R²   : {val_r2:.4f}")
    print(f"    Val RMSE : {val_rmse:.4f}")
    print(f"    Time     : {time()-t0:.1f}s")

    trained_models[name] = model
    val_preds[name]      = val_pred
    test_preds[name]     = test_pred

    # Save base model
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
    print(f"    Saved → models/{name}.joblib")


# ══════════════════════════════════════════════
# STEP 4 – Stacking meta-learner (Ridge regression)
#          trained on val-set predictions of base models
# ══════════════════════════════════════════════
print("\nSTEP 4 – Training stacking meta-learner (Ridge)")

# Stack val predictions as features
val_stack  = np.column_stack([val_preds[n]  for n in base_models])
test_stack = np.column_stack([test_preds[n] for n in base_models])

meta = Ridge(alpha=1.0)
meta.fit(val_stack, y_val)

print(f"  Meta-learner coefs : {dict(zip(base_models.keys(), meta.coef_.round(4)))}")
print(f"  Meta-learner intercept: {meta.intercept_:.4f}")

y_val_ensemble  = meta.predict(val_stack)
y_pred_ensemble = meta.predict(test_stack)

val_r2_ens = r2_score(y_val, y_val_ensemble)
print(f"  Ensemble Val R²  : {val_r2_ens:.4f}")

joblib.dump(meta, os.path.join(MODEL_DIR, "meta_ridge.joblib"))
print(f"  Saved → models/meta_ridge.joblib")


# ══════════════════════════════════════════════
# STEP 5 – Full evaluation on test set
# ══════════════════════════════════════════════
print("\nSTEP 5 – Test set evaluation")
print("-" * 65)

report_lines = [
    "ChEMBL pIC50 Ensemble Model – Evaluation Report",
    "=" * 55,
    f"Active/Inactive threshold for classification : pIC50 = {ACTIVE_THRESH}",
    "",
]

# ── Helper: compute all metrics ──────────────────────────────
def regression_metrics(y_true, y_pred, label=""):
    r2   = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

def classification_metrics(y_true, y_pred, thresh=ACTIVE_THRESH):
    y_true_bin = (y_true >= thresh).astype(int)
    y_pred_bin = (y_pred >= thresh).astype(int)
    y_pred_prob = np.clip((y_pred - (thresh - 4)) / 4, 0, 1)  # soft prob proxy
    f1  = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    try:
        auc = roc_auc_score(y_true_bin, y_pred_prob)
    except ValueError:
        auc = float("nan")
    return f1, auc, y_true_bin, y_pred_prob

# ── Global metrics (all models + ensemble) ───────────────────
print(f"\n  {'Model':<22} {'R²':>7} {'RMSE':>7} {'MAE':>7} {'F1':>7} {'AUC-ROC':>9}")
print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")

report_lines.append("Global Test Metrics")
report_lines.append("-" * 55)
report_lines.append(f"  {'Model':<22} {'R²':>7} {'RMSE':>7} {'MAE':>7} {'F1':>7} {'AUC-ROC':>9}")

all_test_preds = {}
for name in base_models:
    yp  = test_preds[name]
    r2, rmse, mae = regression_metrics(y_test, yp)
    f1, auc, _, _ = classification_metrics(y_test, yp)
    all_test_preds[name] = yp
    row = f"  {name:<22} {r2:>7.4f} {rmse:>7.4f} {mae:>7.4f} {f1:>7.4f} {auc:>9.4f}"
    print(row); report_lines.append(row)

# Ensemble
r2, rmse, mae        = regression_metrics(y_test, y_pred_ensemble)
f1, auc, y_true_bin, y_pred_prob = classification_metrics(y_test, y_pred_ensemble)
all_test_preds["Ensemble"] = y_pred_ensemble
row = f"  {'Ensemble (Ridge)':<22} {r2:>7.4f} {rmse:>7.4f} {mae:>7.4f} {f1:>7.4f} {auc:>9.4f}"
print(row); report_lines.append(row)
print(f"  {'':22} {'':7} {'':7} {'':7} {'':7} {'':9}")

# ── Per-zone metrics (ensemble only) ─────────────────────────
print(f"\n  Per-zone metrics (Ensemble):")
print(f"  {'Zone':<10} {'N':>6} {'R²':>7} {'RMSE':>7} {'MAE':>7} {'F1':>7} {'AUC-ROC':>9}")

report_lines += ["", "Per-Zone Metrics (Ensemble)", "-" * 55]
report_lines.append(f"  {'Zone':<10} {'N':>6} {'R²':>7} {'RMSE':>7} {'MAE':>7} {'F1':>7} {'AUC-ROC':>9}")

for zone in ["low", "bulk", "high", "all"]:
    if zone == "all":
        mask = np.ones(len(y_test), dtype=bool)
    else:
        mask = zone_test == zone
    if mask.sum() < 2:
        continue
    yt = y_test[mask]
    yp = y_pred_ensemble[mask]
    r2_z, rmse_z, mae_z = regression_metrics(yt, yp)
    f1_z, auc_z, _, _   = classification_metrics(yt, yp)
    row = f"  {zone:<10} {mask.sum():>6} {r2_z:>7.4f} {rmse_z:>7.4f} {mae_z:>7.4f} {f1_z:>7.4f} {auc_z:>9.4f}"
    print(row); report_lines.append(row)

# ── Classification report ─────────────────────────────────────
print(f"\n  Classification report (threshold = {ACTIVE_THRESH}):")
cr = classification_report(y_true_bin, (y_pred_ensemble >= ACTIVE_THRESH).astype(int),
                            target_names=["Inactive", "Active"])
print(cr)
report_lines += ["", f"Classification report (threshold = {ACTIVE_THRESH})", cr]


# ══════════════════════════════════════════════
# STEP 6 – Cross-validation on train+val (5-fold)
# ══════════════════════════════════════════════
print("\nSTEP 6 – 5-fold CV on train+val (Ensemble approximate: XGB_conservative)")

# For CV speed, use the best single model as a proxy
cv_model = XGBRegressor(
    n_estimators  = 500,
    learning_rate = 0.05,
    max_depth     = 6,
    subsample     = 0.8,
    colsample_bytree = 0.7,
    tree_method   = "hist",
    n_jobs        = N_JOBS,
    random_state  = RANDOM_SEED,
    verbosity     = 0,
)
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_r2 = cross_val_score(cv_model, X_trainval, y_trainval, cv=kf,
                         scoring="r2", n_jobs=N_JOBS)
cv_rmse = (-cross_val_score(cv_model, X_trainval, y_trainval, cv=kf,
                             scoring="neg_root_mean_squared_error", n_jobs=N_JOBS))

print(f"  CV R²   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"  CV RMSE : {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
report_lines += [
    "", "5-Fold CV (XGB_conservative proxy on train+val)",
    f"  R²   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}",
    f"  RMSE : {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}",
]


# ══════════════════════════════════════════════
# STEP 7 – Feature importance (RF + XGB average)
# ══════════════════════════════════════════════
print("\nSTEP 7 – Feature importance")

rf1_imp  = trained_models["RF_balanced"].feature_importances_
rf2_imp  = trained_models["RF_extreme"].feature_importances_
xgb1_imp = trained_models["XGB_conservative"].feature_importances_
xgb2_imp = trained_models["XGB_aggressive"].feature_importances_

avg_imp = (rf1_imp + rf2_imp + xgb1_imp + xgb2_imp) / 4.0

imp_df = pd.DataFrame({
    "feature"          : FEATURE_COLS,
    "RF_balanced"      : rf1_imp,
    "RF_extreme"       : rf2_imp,
    "XGB_conservative" : xgb1_imp,
    "XGB_aggressive"   : xgb2_imp,
    "avg_importance"   : avg_imp,
}).sort_values("avg_importance", ascending=False)

imp_df.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)
print(f"  Saved → results/feature_importance.csv")
print(f"\n  Top 15 features (by average importance):")
print(imp_df[["feature", "avg_importance"]].head(15).to_string(index=False))


# ══════════════════════════════════════════════
# STEP 8 – Plots
# ══════════════════════════════════════════════
print("\nSTEP 8 – Generating plots")

# Color by zone
zone_color = {"low": "#BA7517", "bulk": "#1D9E75", "high": "#E24B4A"}
colors = [zone_color.get(z, "#888") for z in zone_test]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("ChEMBL pIC50 Ensemble – Evaluation", fontsize=14, fontweight="bold")

# ── Plot 1: Predicted vs Actual (ensemble) ────────────────────
ax = axes[0, 0]
ax.scatter(y_test, y_pred_ensemble, c=colors, alpha=0.4, s=10, linewidths=0)
mn, mx = min(y_test.min(), y_pred_ensemble.min()), max(y_test.max(), y_pred_ensemble.max())
ax.plot([mn, mx], [mn, mx], "k--", lw=1, label="ideal")
ax.axvline(ACTIVE_THRESH, color="gray", lw=0.8, ls=":")
ax.axhline(ACTIVE_THRESH, color="gray", lw=0.8, ls=":")
ax.set_xlabel("Actual pIC50"); ax.set_ylabel("Predicted pIC50")
r2_all = r2_score(y_test, y_pred_ensemble)
rmse_all = mean_squared_error(y_test, y_pred_ensemble) ** 0.5
ax.set_title(f"Predicted vs Actual  (R²={r2_all:.3f}, RMSE={rmse_all:.3f})")
from matplotlib.patches import Patch
legend_elements = [Patch(color=c, label=z) for z, c in zone_color.items()]
ax.legend(handles=legend_elements, fontsize=8)

# ── Plot 2: Residuals ─────────────────────────────────────────
ax = axes[0, 1]
residuals = y_test - y_pred_ensemble
ax.scatter(y_pred_ensemble, residuals, c=colors, alpha=0.4, s=10, linewidths=0)
ax.axhline(0, color="k", lw=1, ls="--")
ax.set_xlabel("Predicted pIC50"); ax.set_ylabel("Residual (actual - predicted)")
ax.set_title("Residuals vs Predicted")

# ── Plot 3: ROC curve ─────────────────────────────────────────
ax = axes[1, 0]
fpr, tpr, _ = roc_curve(y_true_bin, y_pred_prob)
ax.plot(fpr, tpr, color="#534AB7", lw=2, label=f"Ensemble AUC-ROC = {auc:.3f}")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve  (threshold = {ACTIVE_THRESH})")
ax.legend(fontsize=9)

# ── Plot 4: Top-20 feature importances ───────────────────────
ax = axes[1, 1]
top20 = imp_df.head(20)
ax.barh(top20["feature"][::-1], top20["avg_importance"][::-1],
        color="#1D9E75", edgecolor="none")
ax.set_xlabel("Average importance")
ax.set_title("Top 20 features (avg across 4 models)")
ax.tick_params(axis="y", labelsize=7)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "evaluation_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {plot_path}")


# ── Plot 5: Per-model R² bar chart ───────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
model_names = list(base_models.keys()) + ["Ensemble"]
model_r2    = [r2_score(y_test, all_test_preds[n]) for n in model_names]
bar_colors  = ["#9FE1CB", "#5DCAA5", "#AFA9EC", "#7F77DD", "#E24B4A"]
bars = ax.bar(model_names, model_r2, color=bar_colors, edgecolor="none")
ax.set_ylim(0, 1)
ax.set_ylabel("R² on test set")
ax.set_title("Model R² comparison")
for bar, val in zip(bars, model_r2):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.4f}",
            ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
r2_bar_path = os.path.join(RESULTS_DIR, "model_r2_comparison.png")
plt.savefig(r2_bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {r2_bar_path}")


# ══════════════════════════════════════════════
# STEP 9 – Save report
# ══════════════════════════════════════════════
report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"\n  Saved → {report_path}")


# ══════════════════════════════════════════════
# STEP 10 – Inference helper (reusable)
# ══════════════════════════════════════════════
INFERENCE_SNIPPET = '''
# ── Inference snippet (paste into your notebook) ──────────────
import joblib, pickle, numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdMorganFingerprint

models = {n: joblib.load(f"models/{n}.joblib")
          for n in ["RF_balanced","RF_extreme","XGB_conservative","XGB_aggressive"]}
meta   = joblib.load("models/meta_ridge.joblib")

with open("data_splits/physchem_scaler.pkl","rb") as f:
    scaler = pickle.load(f)

with open("data_splits/feature_cols.txt") as f:
    FEATURE_COLS = [l.strip() for l in f]

def predict_pic50(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    rows = []
    for mol in mols:
        physchem = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
            Descriptors.TPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
            mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcFractionCSP3(mol),
            len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ]
        fp = rdMorganFingerprint.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        rows.append(physchem + list(map(int, fp.ToBitString())))

    import pandas as pd
    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    PHYSCHEM = ["mw","logp","hbd","hba","tpsa","rot_bonds",
                "rings","arom_rings","heavy_atoms","frac_csp3","stereo"]
    df[PHYSCHEM] = scaler.transform(df[PHYSCHEM])
    X = df[FEATURE_COLS].values.astype("float32")
    preds_stack = np.column_stack([m.predict(X) for m in models.values()])
    return meta.predict(preds_stack)
'''

with open(os.path.join(MODEL_DIR, "inference_snippet.py"), "w") as f:
    f.write(INFERENCE_SNIPPET)
print(f"  Saved → models/inference_snippet.py")

print("\n" + "=" * 65)
print("Training & evaluation complete.")
print(f"  Models    → {MODEL_DIR}/")
print(f"  Results   → {RESULTS_DIR}/")
print("=" * 65)