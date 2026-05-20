#!/bin/bash
# run_pipeline_run4.sh — REINVENT4 pIC50 Pipeline (Run 4 - No Tanimoto Penalty)
# =============================================================================
# Executes reinforcement learning, sampling, hit extraction, mol2mol generation,
# and final novelty validation for Run 4 (no Tanimoto penalty, starting from 
# verified optimal Transfer Learning prior Epoch 50).
#
# Requirements:
#   conda activate reinvent-qsar
# =============================================================================
set -e

# ── Resolve repository root
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export REPO_ROOT

echo "=================================================="
echo "  JAK2 pIC50 REINVENT4 PIPELINE: RUN 4 (No Tanimoto)"
echo "=================================================="
echo "  Repo root: $REPO_ROOT"

# ── Create output directories ──────────────────────────────────────────────────
echo ""
echo "[*] Creating output directories..."
mkdir -p "$REPO_ROOT/models"
mkdir -p "$REPO_ROOT/results"
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/tb_jak2_rl_run4"
mkdir -p "$REPO_ROOT/results/run4_validation"

# ── Activate conda environment ─────────────────────────────────────────────────
echo "[*] Activating conda environment: reinvent-qsar"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate reinvent-qsar

# ── Set PYTHONPATH so REINVENT scoring plugins are importable ──────────────────
export PYTHONPATH="$REPO_ROOT/REINVENT4:${PYTHONPATH:-}"
echo "[*] PYTHONPATH → $REPO_ROOT/REINVENT4"

# ── Phase 2: Reinforcement Learning (Run 4) ────────────────────────────────────
# Reads:  models/jak2_focused.model.50.chkpt + data/xgb_model.ubj + data/desc_scaler.pkl
# Writes: models/jak2_rl_stage1_run4.chkpt, models/jak2_rl_stage2_run4.chkpt
#         results/focused_rl_run4_1.csv
echo ""
if [ -f "$REPO_ROOT/results/focused_rl_run4_1.csv" ] && [ -s "$REPO_ROOT/results/focused_rl_run4_1.csv" ]; then
    echo "[*] Phase 2: Found existing RL output at results/focused_rl_run4_1.csv — skipping Reinforcement Learning."
else
    echo "[*] Phase 2: Running Reinforcement Learning (Run 4 - No Tanimoto Penalty)..."
    (
        cd "$REPO_ROOT/REINVENT4"
        reinvent \
            -l "$REPO_ROOT/logs/jak2_rl_run4.log" \
            "configs/jak2_rl_run4.toml"
    )
fi

# ── Phase 3: Copy latest RL checkpoint as final model (Run 4) ─────────────────
echo ""
if [ -f "$REPO_ROOT/models/jak2_rl_final_run4.model" ]; then
    echo "[*] Phase 3: Found existing final RL model at models/jak2_rl_final_run4.model — skipping checkpoint copy."
else
    echo "[*] Phase 3: Locating latest Run 4 RL checkpoint..."
    python - <<'PYEOF'
import os, glob, shutil

checkpoints = glob.glob(os.path.join(os.environ["REPO_ROOT"], "models", "*run4*.chkpt"))
if not checkpoints:
    # Fallback to stage2 checkpoint directly
    stage2_chkpt = os.path.join(os.environ["REPO_ROOT"], "models", "jak2_rl_stage2_run4.chkpt")
    if os.path.exists(stage2_chkpt):
        checkpoints = [stage2_chkpt]

if not checkpoints:
    print("[ERROR] No RL checkpoints found for Run 4! Pipeline failed.")
    exit(1)

latest = max(checkpoints, key=os.path.getmtime)
dest   = os.path.join(os.environ["REPO_ROOT"], "models", "jak2_rl_final_run4.model")
shutil.copy(latest, dest)
print(f"  Copied: {os.path.basename(latest)} → models/jak2_rl_final_run4.model")
PYEOF
fi

# ── Phase 4: Sampling from RL model (Run 4) ────────────────────────────────────
# Reads:  models/jak2_rl_final_run4.model
# Writes: results/jak2_rl_candidates_run4.csv
echo ""
if [ -f "$REPO_ROOT/results/jak2_rl_candidates_run4.csv" ] && [ -s "$REPO_ROOT/results/jak2_rl_candidates_run4.csv" ]; then
    echo "[*] Phase 4: Found existing sampling candidates at results/jak2_rl_candidates_run4.csv — skipping Sampling."
else
    echo "[*] Phase 4: Sampling from RL model..."
    (
        cd "$REPO_ROOT/REINVENT4"
        reinvent \
            -l "$REPO_ROOT/logs/jak2_rl_sampling_run4.log" \
            "configs/jak2_sampling_rl_run4.toml"
    )
fi

# ── Phase 5: Extract top 10 unique hits (Run 4) ────────────────────────────────
echo ""
if [ -f "$REPO_ROOT/results/top_10_hits_run4.csv" ]; then
    echo "[*] Phase 5: Found existing top 10 unique hits at results/top_10_hits_run4.csv — skipping extraction."
else
    echo "[*] Phase 5: Extracting top 10 unique hits for Run 4..."
    python - <<'PYEOF'
import pandas as pd, os

repo = os.environ["REPO_ROOT"]
csv_path = os.path.join(repo, "results", "focused_rl_run4_1.csv")

if not os.path.exists(csv_path):
    print(f"[ERROR] RL results CSV not found: {csv_path}")
    exit(1)

print(f"  Reading: {csv_path}")
df = pd.read_csv(csv_path)

smiles_col = "SMILES" if "SMILES" in df.columns else df.columns[0]

# Prefer raw pIC50 column for ranking
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
print(f"  Extracted {len(top10)} unique top hits for Run 4")
top10.to_csv(os.path.join(repo, "results", "top_10_hits_run4.csv"), index=False)

top10[smiles_col].to_csv(
    os.path.join(repo, "data", "top_hits_run4.smi"), index=False, header=False)
print("  Saved → results/top_10_hits_run4.csv + data/top_hits_run4.smi")
PYEOF
fi

# ── Phase 6: Mol2Mol sampling from top seeds (Run 4) ───────────────────────────
echo ""
if [ -f "$REPO_ROOT/results/jak2_mol2mol_candidates_run4.csv" ] && [ -s "$REPO_ROOT/results/jak2_mol2mol_candidates_run4.csv" ]; then
    echo "[*] Phase 6: Found existing Mol2Mol candidates at results/jak2_mol2mol_candidates_run4.csv — skipping Mol2Mol sampling."
else
    echo "[*] Phase 6: Mol2Mol sampling from top seeds..."
    (
        cd "$REPO_ROOT/REINVENT4"
        reinvent \
            -l "$REPO_ROOT/logs/jak2_mol2mol_run4.log" \
            "configs/jak2_mol2mol_run4.toml"
    )
fi

# ── Phase 7: Tanimoto validation of RL output (Run 4) ──────────────────────────
echo ""
echo "[*] Phase 7: Tanimoto validation of Run 4 RL output vs training data..."
python "$REPO_ROOT/validate_rl_output.py" \
    --rl_csv "$REPO_ROOT/results/focused_rl_run4_1.csv" \
    --out_dir "$REPO_ROOT/results/run4_validation"

echo ""
echo "=================================================="
echo "  JAK2 PIPELINE COMPLETED FOR RUN 4"
echo "=================================================="
echo "  Results:         $REPO_ROOT/results/"
echo "  Top hits:        $REPO_ROOT/results/top_10_hits_run4.csv"
echo "  Validation CSV:  $REPO_ROOT/results/run4_validation/rl_validation.csv"
echo "  Novelty Plot:    $REPO_ROOT/results/run4_validation/rl_tanimoto_histogram.png"
echo "=================================================="
