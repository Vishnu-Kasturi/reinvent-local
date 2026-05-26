#!/bin/bash
# Unified REINVENT4 Pipeline: JAK2 & PD1-PDL1
# Usage:
#   ./pipeline.sh --target jak2
#   ./pipeline.sh --target pd1_pdl1
#   ./pipeline.sh --target jak2 --fetch EGFR  (fetch EGFR data instead of using local)
#   ./pipeline.sh --target jak2 --step 4      (start from RL)

set -e

# --- Default Arguments ---
TARGET=""
FETCH_TARGET=""
START_STEP=0

# --- Parse Args ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --target) TARGET="$2"; shift ;;
        --fetch) FETCH_TARGET="$2"; shift ;;
        --step) START_STEP="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$TARGET" != "jak2" && "$TARGET" != "pd1_pdl1" ]]; then
    echo "[!] Error: --target must be 'jak2' or 'pd1_pdl1'"
    exit 1
fi

export REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$REPO_ROOT/REINVENT4:${PYTHONPATH:-}"

echo "=================================================="
echo " RUNNING $TARGET PIPELINE (Starting from step $START_STEP)"
echo "=================================================="

# Set target-specific variables
if [[ "$TARGET" == "jak2" ]]; then
    RAW_CSV="Preprocess/Data_jak2/data_csvs/jak2raw.csv"
    PREPROCESS_CSV="Preprocess/Data_jak2/data_csvs/jak2_preprocess_all.csv"
    SOL_CSV=""
    TL_TOML="jak2_TL.toml"
    RL_TOML="jak2_rl_toml.toml"
else
    RAW_CSV="Preprocess/Data_pd1_pdl1/pd1_pdl1_pic50_raw.csv"
    PREPROCESS_CSV="Preprocess/Data_pd1_pdl1/data_csvs/pd1_pdl1_preprocess_all.csv"
    SOL_CSV="Preprocess/Data_pd1_pdl1/pd1_pdl1_sol.csv"
    TL_TOML="pd1_pdl1_TL.toml"
    RL_TOML="pd1_pdl1_rl_toml.toml"
fi

mkdir -p "$REPO_ROOT/data"
mkdir -p "$REPO_ROOT/models"
mkdir -p "$REPO_ROOT/results"
mkdir -p "$REPO_ROOT/logs"

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate reinvent-qsar

# ---------------------------------------------------------
# STEP 0: FETCH CHEMBL DATA
# ---------------------------------------------------------
if [[ -n "$FETCH_TARGET" && $START_STEP -le 0 ]]; then
    echo "[*] Step 0: Fetching ChEMBL Data for $FETCH_TARGET"
    python Preprocess/scripts/fetch_chembl.py --target "$FETCH_TARGET" --out "data/${FETCH_TARGET}_raw.csv"
    # Overwrite RAW_CSV if we fetched fresh data
    RAW_CSV="data/${FETCH_TARGET}_raw.csv"
    PREPROCESS_CSV="$RAW_CSV"
fi

# ---------------------------------------------------------
# STEP 1: PREPARE TL SMILES
# ---------------------------------------------------------
if [[ $START_STEP -le 1 ]]; then
    echo ""
    echo "[*] Step 1: Preparing TL SMILES Data"
    python Preprocess/scripts/prepare_tl_smiles.py \
        --input_csv "$PREPROCESS_CSV" \
        --train_out "data/${TARGET}_TL_train.smi" \
        --val_out "data/${TARGET}_TL_val.smi"
fi

# ---------------------------------------------------------
# STEP 2: TRANSFER LEARNING
# ---------------------------------------------------------
if [[ $START_STEP -le 2 ]]; then
    echo ""
    echo "[*] Step 2: Running Transfer Learning"
    (
        cd "$REPO_ROOT/REINVENT4"
        reinvent -l "../logs/${TARGET}_tl.log" "configs/$TL_TOML"
    )
fi

# ---------------------------------------------------------
# STEP 3: EVALUATE TL CHECKPOINTS
# ---------------------------------------------------------
if [[ $START_STEP -le 3 ]]; then
    echo ""
    echo "[*] Step 3: Evaluating TL Checkpoints"
    python Preprocess/scripts/evaluate_tl_checkpoints.py --target "$TARGET" --raw_csv "$RAW_CSV"
fi

# ---------------------------------------------------------
# STEP 4: REINFORCEMENT LEARNING
# ---------------------------------------------------------
if [[ $START_STEP -le 4 ]]; then
    echo ""
    echo "[*] Step 4: Running Reinforcement Learning"
    
    # Read the best TL checkpoint from Step 3
    BEST_EPOCH_FILE="results/${TARGET}_best_epoch.txt"
    if [[ -f "$BEST_EPOCH_FILE" ]]; then
        BEST_CHKPT=$(cat "$BEST_EPOCH_FILE")
        echo "[*] Found best TL checkpoint: $BEST_CHKPT"
        
        # Inject the best checkpoint into the RL TOML
        ESCAPED_CHKPT=$(echo "$BEST_CHKPT" | sed 's/\//\\\//g')
        sed -i.bak "s/agent_file = .*/agent_file = \"$ESCAPED_CHKPT\"/" "REINVENT4/configs/$RL_TOML"
    else
        echo "[!] Warning: Best checkpoint file not found. Using default agent_file in TOML."
    fi
    
    (
        cd "$REPO_ROOT/REINVENT4"
        reinvent -l "../logs/${TARGET}_rl.log" "configs/$RL_TOML"
    )
fi

# ---------------------------------------------------------
# STEP 5: KDE PLOTS
# ---------------------------------------------------------
if [[ $START_STEP -le 5 ]]; then
    echo ""
    echo "[*] Step 5: Plotting KDE Distributions"
    if [[ -n "$SOL_CSV" ]]; then
        python Preprocess/scripts/plot_kde_pipeline.py --target "$TARGET" --raw_pic50 "$PREPROCESS_CSV" --raw_sol "$SOL_CSV"
    else
        python Preprocess/scripts/plot_kde_pipeline.py --target "$TARGET" --raw_pic50 "$PREPROCESS_CSV"
    fi
fi

# ---------------------------------------------------------
# STEP 6: TANIMOTO VALIDATION
# ---------------------------------------------------------
if [[ $START_STEP -le 6 ]]; then
    echo ""
    echo "[*] Step 6: Tanimoto Validation of RL Output"
    python Preprocess/scripts/validate_tanimoto_pipeline.py --target "$TARGET" --raw_csv "$RAW_CSV"
fi

echo ""
echo "[*] Pipeline complete!"
