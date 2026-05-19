#!/bin/bash
# run_pipeline.sh — REINVENT4 pIC50 Pipeline
# ============================================
# Usage:
#   ./run_pipeline.sh           → run JAK2 pipeline (data already in data/)
#   ./run_pipeline.sh EGFR      → fetch EGFR from ChEMBL, then run pipeline
#
# Requirements:
#   conda activate reinvent-qsar
#   pip install -r requirements.txt   (installs REINVENT4 in editable mode)
# =============================================================================
set -e

TARGET_NAME=${1:-""}

# ── Resolve repository root (always correct regardless of where you call it from)
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export REPO_ROOT

echo "========================================"
if [ -n "$TARGET_NAME" ]; then
    echo "  $TARGET_NAME pIC50 REINVENT4 PIPELINE"
else
    echo "  JAK2 pIC50 REINVENT4 PIPELINE"
fi
echo "========================================"
echo "  Repo root: $REPO_ROOT"

# ── Create output directories ──────────────────────────────────────────────────
echo ""
echo "[*] Creating output directories..."
mkdir -p "$REPO_ROOT/models"
mkdir -p "$REPO_ROOT/results"
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/tb_jak2_tl"
mkdir -p "$REPO_ROOT/tb_jak2_rl"
mkdir -p "$REPO_ROOT/data"

# ── Activate conda environment ─────────────────────────────────────────────────
echo "[*] Activating conda environment: reinvent-qsar"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate reinvent-qsar

# ── Set PYTHONPATH so REINVENT scoring plugins are importable ──────────────────
# This must point to the REINVENT4 directory so that
# reinvent_plugins.components.comp_jak2_pic50 is discoverable.
export PYTHONPATH="$REPO_ROOT/REINVENT4:${PYTHONPATH:-}"
echo "[*] PYTHONPATH → $REPO_ROOT/REINVENT4"

# ── Step 0: Dynamic target fetching (only if TARGET_NAME is provided) ──────────
if [ -n "$TARGET_NAME" ]; then
    echo ""
    echo "[*] Step 0: Fetching $TARGET_NAME bioactivity data from ChEMBL..."
    python "$REPO_ROOT/preprocess/1_prepare_data.py" "$TARGET_NAME" --min_pic50 6.0

    echo "[*] Step 0.5: Training XGBoost pIC50 model for $TARGET_NAME..."
    python "$REPO_ROOT/preprocess/2_train_xgboost.py" "$TARGET_NAME"
    # ↑ Outputs xgb_model.ubj + desc_scaler.pkl directly to data/ (no path editing needed)
fi

# ── Step 1: Transfer Learning ──────────────────────────────────────────────────
# Reads:  data/custom_train.smi + data/custom_val.smi
# Writes: models/jak2_focused.model
echo ""
echo "[*] Phase 1: Transfer Learning..."
reinvent \
    -l "$REPO_ROOT/logs/jak2_tl.log" \
    "$REPO_ROOT/REINVENT4/configs/jak2_tl.toml"

# ── Step 2: Reinforcement Learning ────────────────────────────────────────────
# Reads:  models/jak2_focused.model + data/xgb_model.ubj + data/desc_scaler.pkl
# Writes: models/jak2_rl_stage1_v2.chkpt, models/jak2_rl_stage2_v2.chkpt
#         results/focused_rl_v2_*.csv  (with JAK2pIC50 [0-1] AND JAK2pIC50_raw [4-11])
echo ""
echo "[*] Phase 2: Reinforcement Learning..."
reinvent \
    -l "$REPO_ROOT/logs/jak2_rl.log" \
    "$REPO_ROOT/REINVENT4/configs/jak2_rl_v2.toml"

# ── Step 3: Copy latest RL checkpoint as final model ──────────────────────────
echo ""
echo "[*] Phase 3: Locating latest RL checkpoint..."
python - <<'PYEOF'
import os, glob, shutil

checkpoints = glob.glob(os.path.join(os.environ["REPO_ROOT"], "models", "*.chkpt"))
if not checkpoints:
    print("[ERROR] No RL checkpoints found! Pipeline failed.")
    exit(1)

latest = max(checkpoints, key=os.path.getmtime)
dest   = os.path.join(os.environ["REPO_ROOT"], "models", "jak2_rl_final.model")
shutil.copy(latest, dest)
print(f"  Copied: {os.path.basename(latest)} → models/jak2_rl_final.model")
PYEOF

# ── Step 4: Sampling from RL model ────────────────────────────────────────────
echo ""
echo "[*] Phase 4: Sampling from RL model..."
reinvent \
    -l "$REPO_ROOT/logs/jak2_rl_sampling.log" \
    "$REPO_ROOT/REINVENT4/configs/jak2_sampling_rl.toml"

# ── Step 5: Extract top 10 unique hits ────────────────────────────────────────
echo ""
echo "[*] Phase 5: Extracting top 10 unique hits..."
python - <<'PYEOF'
import pandas as pd, glob, os

repo = os.environ["REPO_ROOT"]
csvs = sorted(glob.glob(os.path.join(repo, "results", "focused_rl*.csv")))
if not csvs:
    print("[ERROR] No RL results CSV found!")
    exit(1)

csv_path = csvs[-1]
print(f"  Reading: {csv_path}")
df = pd.read_csv(csv_path)

smiles_col = "SMILES" if "SMILES" in df.columns else df.columns[0]

# Prefer raw pIC50 column for ranking (added by comp_jak2_pic50.py)
if "JAK2pIC50_raw" in df.columns:
    rank_col = "JAK2pIC50_raw"
    print("  Ranking by JAK2pIC50_raw (actual predicted pIC50, 4-11 range)")
elif "JAK2pIC50 (raw)" in df.columns:
    rank_col = "JAK2pIC50 (raw)"
else:
    rank_col = "Score"
    print("  [WARNING] JAK2pIC50_raw column not found — ranking by Score")

top10 = (
    df[df["Score"] > 0]
    .drop_duplicates(subset=[smiles_col])
    .sort_values(by=rank_col, ascending=False)
    .head(10)
)
print(f"  Extracted {len(top10)} unique top hits")
top10.to_csv(os.path.join(repo, "results", "top_10_hits.csv"), index=False)

os.makedirs(os.path.join(repo, "data"), exist_ok=True)
top10[smiles_col].to_csv(
    os.path.join(repo, "data", "top_hits.smi"), index=False, header=False)
print("  Saved → results/top_10_hits.csv + data/top_hits.smi")
PYEOF

# ── Step 6: Mol2Mol sampling from top seeds ───────────────────────────────────
echo ""
echo "[*] Phase 6: Mol2Mol sampling from top seeds..."
reinvent \
    -l "$REPO_ROOT/logs/jak2_mol2mol.log" \
    "$REPO_ROOT/REINVENT4/configs/jak2_mol2mol.toml"

# ── Step 7: Tanimoto validation of RL output ──────────────────────────────────
echo ""
echo "[*] Phase 7: Tanimoto validation of RL output vs training data..."
python "$REPO_ROOT/validate_rl_output.py"

echo ""
echo "========================================"
if [ -n "$TARGET_NAME" ]; then
    echo "  $TARGET_NAME PIPELINE COMPLETED"
else
    echo "  JAK2 PIPELINE COMPLETED"
fi
echo "========================================"
echo "  Results:     $REPO_ROOT/results/"
echo "  Top hits:    $REPO_ROOT/results/top_10_hits.csv"
echo "  Validation:  $REPO_ROOT/results/rl_validation.csv"
echo "========================================"
