#!/bin/bash
#SBATCH --job-name=reinvent_mol2mol
#SBATCH --output=logs/reinvent_mol2mol_%j.log
#SBATCH --error=logs/reinvent_mol2mol_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Description: Script to run REINVENT 4 staged learning (Mol2Mol) on a DGX GPU node.
# Usage:
#   1. As a direct shell script:
#       chmod +x run_mol2mol_dgx.sh
#       ./run_mol2mol_dgx.sh
#   2. As a SLURM job submission:
#       sbatch run_mol2mol_dgx.sh

set -e

# Get the script directory (repository root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure output directories exist
mkdir -p logs results models

# --- Environment Setup ---
echo "Initializing conda..."
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    # Try other common miniconda/anaconda locations
    for path in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$path/etc/profile.d/conda.sh" ]; then
            source "$path/etc/profile.d/conda.sh"
            break
        fi
    done
fi

# Activate the conda environment
echo "Activating reinvent4 conda environment..."
conda activate reinvent4 || {
    echo "Conda activation failed. Attempting run via 'conda run'..."
    cd "$SCRIPT_DIR/REINVENT4"
    conda run -n reinvent4 python -m reinvent.Reinvent -l ../logs/pd1_pdl1_mol2mol_docking_hits.log configs/pd1_pdl1_mol2mol_docking_hits_dgx.toml
    exit 0
}

# --- Run REINVENT ---
echo "Starting REINVENT staged learning (Mol2Mol) on CUDA GPU..."
cd "$SCRIPT_DIR/REINVENT4"
python -m reinvent.Reinvent -l ../logs/pd1_pdl1_mol2mol_docking_hits.log configs/pd1_pdl1_mol2mol_docking_hits_dgx.toml

echo "REINVENT run completed successfully."
