# Unified Mol2Mol Pipeline & Post-Run Analysis Guide

This guide describes the architecture, installation requirements, and usage instructions for the unified **Mol2Mol Pipeline** and **Post-Run Analysis** tool. It is optimized to run locally on your macOS workstation as well as portably on high-performance Linux GPU clusters (such as a **Linux DGX A6000** system).

---

## 📂 Pipeline Components

The pipeline consists of the following key files:

### 1. [`run_mol2mol_pipeline.sh`](file:///Users/vishnukasturi/Intern/reinvent-local/run_mol2mol_pipeline.sh)
The main Bash orchestrator script. It:
* Automatically activates the `reinvent-qsar` Conda environment on any OS.
* Runs the REINVENT4 `mol2mol` job (staged learning or sampling mode) using the designated TOML configuration.
* Dynamically parses the TOML file at runtime to locate the correct generated results path (supporting `summary_csv_prefix` and `output_file`).
* Calls the Python analysis script with the parsed column names and target thresholds.
* Automatically checks for local Gemini directories and copies assets to IDE visual artifacts if present, otherwise gracefully skipping without error on Linux DGX environments.

### 2. [`Preprocess/scripts/run_mol2mol_analysis.py`](file:///Users/vishnukasturi/Intern/reinvent-local/Preprocess/scripts/run_mol2mol_analysis.py)
The Python analysis engine. It:
* Standardizes starting leads and generated compound SMILES to canonical RDKit format.
* **Neutralizes Formal Charges** (e.g. converting `[NH+]` or `[N@H+]` to neutral tertiary/secondary amines) to match the charged CSV compound entries against the neutralized REINVENT outputs.
* Computes Morgan Fingerprint (radius=2, 2048-bit) Tanimoto similarities between generated analogs and parent leads.
* Performs heuristic candidate matching: it filters for synthetic accessibility ($\text{SAScore} \le 4.5$) and similarity ($\text{Tanimoto} \ge 0.40$), selecting the analog that provides the highest solubility improvement while maintaining bioactivity.
* Generates three high-resolution figures:
  1. **Overall Distributions (`<run_name>_distributions.png`)**: KDE plots for pIC50, Solubility logS, and a histogram of Tanimoto similarities to parent leads.
  2. **1-to-1 Dumbbell Shifts (`<run_name>_pair_shifts.png`)**: Side-by-side shifts showing individual potency and solubility progression.
  3. **RDKit Structure Grids (`<run_name>_pairs_grid.png`)**: High-res 2D structural layout containing Parent Lead vs. Optimized Analogs side-by-side with annotations.
* Writes matched optimized pair properties to `<run_name>_top15_pairs.csv`.

### 3. Mol2Mol TOML Configurations (`REINVENT4/configs/`)
* **[`pd1_pdl1_mol2mol_sol_opt.toml`](file:///Users/vishnukasturi/Intern/reinvent-local/REINVENT4/configs/pd1_pdl1_mol2mol_sol_opt.toml)**: Runs reinforcement staged learning to optimize solubility for the top 15 starting leads.
* **[`jak2_mol2mol.toml`](file:///Users/vishnukasturi/Intern/reinvent-local/REINVENT4/configs/jak2_mol2mol.toml)**: Runs beam search sampling around JAK2 top leads.
* Additional configurations: `pd1_pdl1_mol2mol_mmp.toml`, `pd1_pdl1_mol2mol_scaffold_generic.toml`, `pd1_pdl1_mol2mol_medium_similarity.toml`.

### 4. Mol2Mol Priors (`REINVENT4/priors/`)
* Contains pre-trained baseline agents that define the chemical transition rules for Mol2Mol (e.g. `mol2mol_high_similarity.prior`, `mol2mol_mmp.prior`, `mol2mol_scaffold_generic.prior`, `mol2mol_medium_similarity.prior`).

---

## 🛠️ Linux DGX A6000 Installation & Requirements

Follow these steps to set up the runtime environment on the Linux DGX A6000:

### 1. System Dependencies
Ensure the system has CUDA drivers and compilers installed (usually pre-configured on DGX boxes).

### 2. Conda Environment Setup
Create the `reinvent-qsar` environment from the repository's configuration. Run the following in your shell:

```bash
# If using environment.yml
conda env create -f environment.yml

# Or create manually if preferred:
conda create -y -n reinvent-qsar python=3.10
conda activate reinvent-qsar

# Install PyTorch with CUDA support (A6000 DGX GPUs use CUDA 11 or 12)
conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install RDKit, Seaborn, and Scipy
conda install -y -c conda-forge rdkit
pip install pandas numpy matplotlib seaborn scipy xgboost
```

### 3. Clone and Initialize
```bash
git clone https://github.com/Vishnu-Kasturi/reinvent-local.git
cd reinvent-local

# Make the scripts executable
chmod +x run_mol2mol_pipeline.sh Preprocess/scripts/run_mol2mol_analysis.py
```

---

## 🚀 Usage Guide

The pipeline supports direct, parameter-based executions.

### 1. Running standard PD1-PDL1 Mol2Mol Run
Runs the REINVENT4 staged learning step followed by the full analysis pipeline:
```bash
./run_mol2mol_pipeline.sh --target pd1_pdl1
```

### 2. Skipping REINVENT (Just run plotting / validation on existing results)
```bash
./run_mol2mol_pipeline.sh --target pd1_pdl1 --skip_run
```

### 3. Customizing Input Leads, Configs, or Column Names on the DGX
If you modify your input CSV or change target features, specify the parameters on command line. The analysis script also features **auto-guessing** for columns if they are not exact matches:
```bash
./run_mol2mol_pipeline.sh \
    --target pd1_pdl1 \
    --config my_custom_mol2mol.toml \
    --leads_csv my_leads.csv \
    --pic50_col "pIC50_pred" \
    --sol_col "logS_pred" \
    --target_pic50 8.0 \
    --target_sol -2.5
```

---

## 🔍 Column Auto-Guessing & Verification
If column names aren't specified, the script automatically searches the results CSV for case-insensitive substrings:
* **pIC50 / Activity**: Searches for columns containing `pic50`, `activity`, `active`, or `pred_p`.
* **Solubility**: Searches for columns containing `sol`, `logs`, `solubility`, or `log_s`.
* **SAScore**: Searches for columns containing `sa`, `synthetic`, or `sascore`.

All charts, structure grids, and pairs CSV files will be saved in the designated `--output_dir` (default: `results/<run_name>_analysis/`).
