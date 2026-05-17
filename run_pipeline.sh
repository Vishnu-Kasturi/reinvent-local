#!/bin/bash
set -e

echo "========================================"
echo "    JAK2 pIC50 REINVENT4 PIPELINE       "
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

# 1. Transfer Learning
echo "[*] Phase 1: Transfer Learning..."
cd REINVENT4
reinvent -l ../logs/jak2_tl.log configs/jak2_tl.toml
cd ..

# 2. Reinforcement Learning
echo "[*] Phase 2: Reinforcement Learning..."
cd REINVENT4
reinvent -l ../logs/jak2_rl.log configs/jak2_rl.toml
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

# 5. Extract top 50 molecules
echo "[*] Phase 5: Extracting top 50 SMILES from sampling results..."
python -c "
import pandas as pd

try:
    df = pd.read_csv('results/jak2_rl_candidates.csv')
    
    # Check if 'SMILES' is a column, if not, find the likely column (first col usually)
    smiles_col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
    
    # Sort by 'Score' if available, otherwise just take the first 50
    if 'Score' in df.columns:
        top50 = df.nlargest(50, 'Score')
    else:
        top50 = df.head(50)
        
    top50_smiles = top50[smiles_col].dropna().unique()[:50]
    
    with open('results/jak2_rl_top50_seeds.smi', 'w') as f:
        for smi in top50_smiles:
            f.write(f'{smi}\\n')
            
    print(f'Successfully extracted {len(top50_smiles)} top SMILES.')
except Exception as e:
    print(f'Error extracting top 50 SMILES: {e}')
    exit(1)
"

# 6. Mol2Mol Sampling
echo "[*] Phase 6: Mol2Mol sampling from top seeds..."
cd REINVENT4
reinvent -l ../logs/jak2_mol2mol.log configs/jak2_mol2mol.toml
cd ..

echo "========================================"
echo "          PIPELINE COMPLETED            "
echo "========================================"
