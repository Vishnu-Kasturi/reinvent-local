#!/bin/bash
set -e

# Resolve repository root directory
cd "$(dirname "$0")/../.."
REPO_ROOT=$(pwd)

echo "========================================================================"
echo "          JAK2 Generative Chemistry Workflow Pipeline"
echo "========================================================================"

# 1. Run Transfer Learning
echo "--> Step 1: Running Transfer Learning (TL) for 150 Epochs..."
PYTHONPATH="$REPO_ROOT/REINVENT4" conda run -n reinvent4 reinvent -l logs/jak2_tl.log REINVENT4/configs/jak2_tl.toml

# 2. Evaluate Transfer Learning Checkpoints
echo "--> Step 2: Evaluating TL Checkpoints & Generating KDE/Tanimoto plots..."
conda run -n reinvent4 python Preprocess/scripts/evaluate_jak2_tl.py

# 3. Run Reinforcement Learning (RL)
echo "--> Step 3: Running Reinforcement Learning (RL) Staged MPO..."
PYTHONPATH="$REPO_ROOT/REINVENT4" conda run -n reinvent4 reinvent -l logs/jak2_rl_run6.log REINVENT4/configs/jak2_rl_run6.toml

# 4. Sample RL Candidates
echo "--> Step 4: Sampling 5000 unique candidates from RL checkpoint..."
PYTHONPATH="$REPO_ROOT/REINVENT4" conda run -n reinvent4 reinvent -l logs/jak2_sampling_rl_run6.log REINVENT4/configs/jak2_sampling_rl_run6.toml

# 5. Evaluate RL Candidates
echo "--> Step 5: Evaluating RL candidates and extracting best MPO hits..."
conda run -n reinvent4 python Preprocess/scripts/evaluate_jak2_rl.py

echo "========================================================================"
echo "          JAK2 Pipeline Run Completed Successfully!"
echo "========================================================================"
