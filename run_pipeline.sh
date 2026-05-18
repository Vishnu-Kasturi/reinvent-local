#!/bin/bash
set -e

# Support optional target argument (e.g. ./run_pipeline.sh EGFR)
TARGET_NAME=$1

echo "========================================"
if [ -n "$TARGET_NAME" ]; then
    echo "    $TARGET_NAME pIC50 REINVENT4 PIPELINE"
else
    echo "    JAK2 pIC50 REINVENT4 PIPELINE       "
fi
echo "========================================"

# Create required directories
echo "[*] Creating directories..."
mkdir -p models results logs tb_jak2_tl tb_jak2_rl

# Activate conda environment
echo "[*] Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate reinvent-qsar || conda activate reinvent4

# Export PYTHONPATH so the REINVENT scoring plugins can find reinvent_plugins/components
export PYTHONPATH="$(pwd)/REINVENT4:$PYTHONPATH"

# Dynamic Target Fetching
if [ -n "$TARGET_NAME" ]; then
    echo "[*] Target name specified: $TARGET_NAME"
    echo "[*] Step 0: Fetching desalted & standardized bioactivity data from ChEMBL..."
    
    # Run fetcher (default minimum pIC50 threshold = 6.0)
    python preprocess/fetch_chembl_target.py "$TARGET_NAME" --min_pic50 6.0
    
    echo "[*] Step 0.5: Copying desalted data to REINVENT4/custom_data/ (no TOML edits needed)..."
    mkdir -p REINVENT4/custom_data
    cp "datasets/$TARGET_NAME/train.smi" REINVENT4/custom_data/custom_train.smi
    cp "datasets/$TARGET_NAME/val.smi"   REINVENT4/custom_data/custom_val.smi
    echo "    Successfully imported $TARGET_NAME dataset into the pipeline!"
fi

# 1. Transfer Learning
echo "[*] Phase 1: Transfer Learning..."
cd REINVENT4
reinvent -l ../logs/jak2_tl.log configs/jak2_tl.toml
cd ..

# 2. Reinforcement Learning (v2 config with fixed NumRotBond and diversity filter)
echo "[*] Phase 2: Reinforcement Learning..."
cd REINVENT4
reinvent -l ../logs/jak2_rl.log configs/jak2_rl_v2.toml
cd ..

# 3. Locate and copy the latest RL checkpoint
echo "[*] Phase 3: Locating latest RL checkpoint..."
python -c "
import os, glob, shutil

checkpoints = glob.glob('models/*.chkpt')
if not checkpoints:
    print('No RL checkpoints found! Pipeline failed.')
    exit(1)

latest_checkpoint = max(checkpoints, key=os.path.getmtime)
print(f'Found latest checkpoint: {latest_checkpoint}')

shutil.copy(latest_checkpoint, 'models/jak2_rl_final.model')
print('Copied to models/jak2_rl_final.model')
"

# 4. RL Sampling
echo "[*] Phase 4: Sampling from RL model..."
cd REINVENT4
reinvent -l ../logs/jak2_rl_sampling.log configs/jak2_sampling_rl.toml
cd ..

# 5. Extract top 10 unique molecules (FIXED: deduplicate + require Score > 0)
echo "[*] Phase 5: Extracting top 10 unique hits from RL results..."
python -c "
import pandas as pd
import os

# Find the latest focused_rl CSV
import glob
csvs = sorted(glob.glob('results/focused_rl*.csv'))
if not csvs:
    print('No RL results CSV found! Pipeline failed.')
    exit(1)
csv_path = csvs[-1]
print(f'Reading: {csv_path}')

df = pd.read_csv(csv_path)

smiles_col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
pic50_col  = 'JAK2pIC50 (raw)' if 'JAK2pIC50 (raw)' in df.columns else 'Score'

# FIXED: only accepted molecules, one row per unique SMILES, ranked by pIC50
top10 = (
    df[df['Score'] > 0]                          # exclude diversity-filter rejections
    .drop_duplicates(subset=[smiles_col])         # one row per unique SMILES
    .sort_values(by=pic50_col, ascending=False)   # rank by predicted pIC50
    .head(10)
)

print(f'Extracted {len(top10)} unique top hits')
top10.to_csv('results/top_10_hits.csv', index=False)

# Write seed SMILES for Mol2Mol
os.makedirs('REINVENT4/custom_data', exist_ok=True)
top10[smiles_col].to_csv('REINVENT4/custom_data/top_hits.smi', index=False, header=False)
print('Saved top_10_hits.csv and top_hits.smi')
"

# 6. Mol2Mol Sampling
echo "[*] Phase 6: Mol2Mol sampling from top seeds..."
cd REINVENT4
reinvent -l ../logs/jak2_mol2mol.log configs/jak2_mol2mol.toml
cd ..

echo "========================================"
if [ -n "$TARGET_NAME" ]; then
    echo "    $TARGET_NAME PIPELINE COMPLETED      "
else
    echo "    JAK2 PIPELINE COMPLETED             "
fi
echo "========================================"
