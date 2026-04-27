"""
Script 2: Pure Neural Network (MLP) for pIC50 Regression
=============================================================================
Input  : data_splits/train.csv, val.csv, test.csv
Output : models/NN_model.joblib
         results/evaluation_report.txt
         results/feature_importance.csv

Modifications:
  - Uses ONLY a Neural Network (MLPRegressor) for infinite geometric extrapolation.
  - Replaces tree-based feature importance with Permutation Importance.

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
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

from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, f1_score, classification_report, roc_curve,
)

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

X_train = train_df[FEATURE_COLS].values.astype(np.float32)
y_train = train_df["pic50"].values

X_val   = val_df[FEATURE_COLS].values.astype(np.float32)
y_val   = val_df["pic50"].values

X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
y_test  = test_df["pic50"].values
zone_test = test_df.get("zone", pd.Series(["bulk"] * len(test_df))).values

print(f"\n  pIC50 train range : [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"  pIC50 test range  : [{y_test.min():.2f}, {y_test.max():.2f}]")


# ══════════════════════════════════════════════
# STEP 2 & 3 – Define and Train Neural Network
# ══════════════════════════════════════════════
print("\nSTEP 2 – Training Neural Network (MLPRegressor)")
t0 = time()

# Deep architecture: 512 -> 256 -> 128 neurons
nn_model = MLPRegressor(
    hidden_layer_sizes=(512, 256, 128), 
    activation="relu",
    solver="adam",
    alpha=0.01,                        # L2 regularization to prevent overfitting
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,               # Stops if validation score stops improving
    validation_fraction=0.15,          # Uses 15% of train for early stopping check
    random_state=RANDOM_SEED
)

print("  Training in progress... (this may take a minute or two)")
nn_model.fit(X_train, y_train)

# Validate on our explicit Val set
val_pred = nn_model.predict(X_val)
val_r2   = r2_score(y_val, val_pred)
val_rmse = mean_squared_error(y_val, val_pred) ** 0.5

print(f"  Training Complete in {time()-t0:.1f}s")
print(f"  Iterations run : {nn_model.n_iter_}")
print(f"  Val R²         : {val_r2:.4f}")
print(f"  Val RMSE       : {val_rmse:.4f}")

joblib.dump(nn_model, os.path.join(MODEL_DIR, "NN_model.joblib"))
print(f"  Saved → models/NN_model.joblib")


# ══════════════════════════════════════════════
# STEP 4 – Full evaluation on test set
# ══════════════════════════════════════════════
print("\nSTEP 4 – Test set evaluation")
print("-" * 65)

y_pred = nn_model.predict(X_test)

report_lines = [
    "ChEMBL pIC50 Neural Network – Evaluation Report",
    "=" * 55,
    f"Active/Inactive threshold for classification : pIC50 = {ACTIVE_THRESH}",
    "",
]

def regression_metrics(y_true, y_p):
    r2   = r2_score(y_true, y_p)
    rmse = mean_squared_error(y_true, y_p) ** 0.5
    mae  = mean_absolute_error(y_true, y_p)
    return r2, rmse, mae

def classification_metrics(y_true, y_p, thresh=ACTIVE_THRESH):
    y_true_bin = (y_true >= thresh).astype(int)
    y_pred_bin = (y_p >= thresh).astype(int)
    y_pred_prob = np.clip((y_p - (thresh - 4)) / 4, 0, 1) 
    f1  = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    try:
        auc = roc_auc_score(y_true_bin, y_pred_prob)
    except ValueError:
        auc = float("nan")
    return f1, auc, y_true_bin, y_pred_prob

# Global metrics
r2_all, rmse_all, mae_all = regression_metrics(y_test, y_pred)
f1_all, auc_all, y_true_bin, y_pred_prob = classification_metrics(y_test, y_pred)

print(f"\n  Global Test Metrics:")
print(f"  R²      : {r2_all:.4f}")
print(f"  RMSE    : {rmse_all:.4f}")
print(f"  MAE     : {mae_all:.4f}")
print(f"  F1      : {f1_all:.4f}")
print(f"  AUC-ROC : {auc_all:.4f}")

report_lines += [
    "Global Test Metrics:",
    f"  R²      : {r2_all:.4f}",
    f"  RMSE    : {rmse_all:.4f}",
    f"  MAE     : {mae_all:.4f}",
    f"  F1      : {f1_all:.4f}",
    f"  AUC-ROC : {auc_all:.4f}",
    ""
]

# Per-zone metrics
print(f"\n  Per-zone metrics:")
print(f"  {'Zone':<10} {'N':>6} {'R²':>7} {'RMSE':>7} {'MAE':>7}")

report_lines += ["Per-Zone Metrics", "-" * 55]
report_lines.append(f"  {'Zone':<10} {'N':>6} {'R²':>7} {'RMSE':>7} {'MAE':>7}")

for zone in ["low", "bulk", "high", "all"]:
    mask = np.ones(len(y_test), dtype=bool) if zone == "all" else (zone_test == zone)
    if mask.sum() < 2: continue
    yt = y_test[mask]
    yp = y_pred[mask]
    r2_z, rmse_z, mae_z = regression_metrics(yt, yp)
    row = f"  {zone:<10} {mask.sum():>6} {r2_z:>7.4f} {rmse_z:>7.4f} {mae_z:>7.4f}"
    print(row); report_lines.append(row)

print(f"\n  Classification report (threshold = {ACTIVE_THRESH}):")
cr = classification_report(y_true_bin, (y_pred >= ACTIVE_THRESH).astype(int),
                            target_names=["Inactive", "Active"])
print(cr)
report_lines += ["", f"Classification report (threshold = {ACTIVE_THRESH})", cr]


# ══════════════════════════════════════════════
# STEP 5 – Feature importance (Permutation)
# ══════════════════════════════════════════════
print("\nSTEP 5 – Calculating Feature Importance (Permutation)")
print("  (This requires scrambling data to see what drops the R² score...)")
t0 = time()

# We run it on the validation set to see which features generalize best
result = permutation_importance(
    nn_model, X_val, y_val, n_repeats=5, random_state=RANDOM_SEED, n_jobs=N_JOBS
)

imp_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": result.importances_mean,
    "std_dev": result.importances_std
}).sort_values("importance", ascending=False)

imp_df.to_csv(os.path.join(RESULTS_DIR, "feature_importance_nn.csv"), index=False)
print(f"  Calculated in {time()-t0:.1f}s")
print(f"  Saved → results/feature_importance.csv")


# ══════════════════════════════════════════════
# STEP 6 – Plots
# ══════════════════════════════════════════════
print("\nSTEP 6 – Generating plots")

zone_color = {"low": "#BA7517", "bulk": "#1D9E75", "high": "#E24B4A"}
colors = [zone_color.get(z, "#888") for z in zone_test]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("ChEMBL pIC50 Neural Network – Evaluation", fontsize=14, fontweight="bold")

# Plot 1: Predicted vs Actual
ax = axes[0, 0]
ax.scatter(y_test, y_pred, c=colors, alpha=0.4, s=10, linewidths=0)
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
ax.plot([mn, mx], [mn, mx], "k--", lw=1, label="ideal")
ax.axvline(ACTIVE_THRESH, color="gray", lw=0.8, ls=":")
ax.axhline(ACTIVE_THRESH, color="gray", lw=0.8, ls=":")
ax.set_xlabel("Actual pIC50"); ax.set_ylabel("Predicted pIC50")
ax.set_title(f"Predicted vs Actual  (R²={r2_all:.3f}, RMSE={rmse_all:.3f})")
from matplotlib.patches import Patch
legend_elements = [Patch(color=c, label=z) for z, c in zone_color.items()]
ax.legend(handles=legend_elements, fontsize=8)

# Plot 2: Residuals
ax = axes[0, 1]
residuals = y_test - y_pred
ax.scatter(y_pred, residuals, c=colors, alpha=0.4, s=10, linewidths=0)
ax.axhline(0, color="k", lw=1, ls="--")
ax.set_xlabel("Predicted pIC50"); ax.set_ylabel("Residual (actual - predicted)")
ax.set_title("Residuals vs Predicted")

# Plot 3: ROC curve
ax = axes[1, 0]
fpr, tpr, _ = roc_curve(y_true_bin, y_pred_prob)
ax.plot(fpr, tpr, color="#534AB7", lw=2, label=f"NN AUC-ROC = {auc_all:.3f}")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve  (threshold = {ACTIVE_THRESH})")
ax.legend(fontsize=9)

# Plot 4: Top-20 feature importances (Permutation)
ax = axes[1, 1]
top20 = imp_df.head(20)
ax.barh(top20["feature"][::-1], top20["importance"][::-1], color="#534AB7", edgecolor="none")
ax.set_xlabel("Mean Decrease in R² when Shuffled")
ax.set_title("Top 20 features (Permutation Importance)")
ax.tick_params(axis="y", labelsize=7)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "evaluation_plots_nn.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {plot_path}")

# ══════════════════════════════════════════════
# STEP 7 – Save report
# ══════════════════════════════════════════════
report_path = os.path.join(RESULTS_DIR, "evaluation_report_nn.txt")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"  Saved → {report_path}")

# ══════════════════════════════════════════════
# STEP 8 – Inference helper (reusable)
# ══════════════════════════════════════════════
INFERENCE_SNIPPET = '''
# ── Inference snippet (paste into your notebook) ──────────────
import joblib, pickle, numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdMorganFingerprint

# Load the single Neural Network model
nn_model = joblib.load("models/NN_model.joblib")

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
    
    # Return the NN prediction
    return nn_model.predict(X)
'''

with open(os.path.join(MODEL_DIR, "inference_snippet_nn.py"), "w") as f:
    f.write(INFERENCE_SNIPPET)
print(f"  Saved → models/inference_snippet_nn.py")

print("\n" + "=" * 65)
print("Training & evaluation complete.")
print(f"  Model     → {MODEL_DIR}/NN_model.joblib")
print(f"  Results   → {RESULTS_DIR}/")
print("=" * 65)