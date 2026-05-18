# REINVENT4 Drug Discovery & Stabilization Pipeline

An end-to-end production-grade generative drug design pipeline optimized for **JAK2 inhibitor generation**, featuring automated bioactivity data retrieval, robust preprocessing, stabilized reinforcement learning dynamics, and automated multi-stage generation.

---

## 🚀 Quick Start

### 1. Standard Pipeline (JAK2)
To run the default optimized JAK2 pipeline end-to-end (Transfer Learning ➔ Reinforcement Learning ➔ Sampling ➔ Mol2Mol):
```bash
./run_pipeline.sh
```

### 2. Fetch and Run for ANY Protein Target
You can run the pipeline for any protein target in ChEMBL (e.g., `EGFR`, `DRD2`, `JAK1`, `JAK3`) by passing the target name as an argument. The pipeline will automatically fetch the target component from ChEMBL, standardize the SMILES, remove salts/counterions, perform train/validation splitting, and run the pipeline:
```bash
./run_pipeline.sh EGFR
```

---

## 📂 Project Architecture

```
reinvent-local/
├── README.md                  # This documentation
├── run_pipeline.sh            # Complete end-to-end orchestration bash script
├── datasets/                  # Directory where fetched ChEMBL datasets are stored
│   └── JAK2/                  # Example fetched dataset directory
│       ├── raw_bioactivity.csv# Raw fetched records from ChEMBL API
│       ├── processed.csv      # Desalted & standardized SMILES with pIC50
│       ├── clean_smiles.smi   # REINVENT-compatible canonical SMILES
│       ├── train.smi          # 80% train split
│       └── val.smi            # 20% validation split
├── preprocess/
│   ├── fetch_chembl_target.py # Multi-target automated ChEMBL API fetcher & preprocessor
│   └── desalt_custom_data.py  # Standalone rigid salt-removal & standardization utility
└── REINVENT4/
    ├── configs/               # Staged RL and TL TOML configurations
    │   ├── jak2_tl.toml       # Transfer learning config
    │   ├── jak2_rl_v2.toml    # Optimized Stage 1 & Stage 2 RL config
    │   └── ...
    └── reinvent_plugins/
        └── components/
            └── comp_jak2_pic50.py # Robust XGBoost scoring component (with NaN & shape guards)
```

---

## 🛠️ Components & Utilities

### 1. Automated ChEMBL Bioactivity Fetcher (`fetch_chembl_target.py`)
This script queries the official ChEMBL API dynamically by protein name, resolves its UniProt ID, downloads bioactivity metrics (IC50, Ki, Kd), and applies a production-grade standardization pipeline:
- **Salt & Counterion Removal:** Strips acids, bases, and counter-ions using RDKit's built-in `SaltRemover`.
- **SMILES Normalization & Neutralization:** Normalizes resonance structures and uncharges standard functional groups.
- **REINVENT Prior Filter:** Discards molecules containing elements/tokens unsupported by the REINVENT baseline prior.
- **Drug-likeness Filtering:** Filters molecules based on molecular weight (150–800 Da), TPSA (≤ 160 Å²), and heavy atom count (≥ 10).

**Manual CLI Usage:**
```bash
python preprocess/fetch_chembl_target.py DRD2 --min_pic50 7.0 --max_mw 550
```

### 2. End-to-End Orchestrator (`run_pipeline.sh`)
The bash script automates the entire generative pipeline stages:
1. **Dynamic Target Processing:** If a target argument (like `EGFR`) is supplied, it downloads, desalts, and copies the clean datasets into `custom_data/` without modifying any configuration files.
2. **Transfer Learning (TL):** Trains/fine-tunes the REINVENT prior model on your desalted target SMILES (`custom_train.smi`) to focus generated chemistry.
3. **Staged Reinforcement Learning (RL):**
   - **Stage 1:** Focuses primarily on JAK2 potency (`JAK2pIC50`), drug-likeness (`QED`), and molecular weight bounds.
   - **Stage 2:** Multi-objective optimization introducing polar surface area (`TPSA`) and rotatable bond (`NumRotBond`) constraints.
4. **RL Checkpoint Resolution:** Auto-resolves and copies the latest/best reinforcement learning checkpoint.
5. **RL Sampling:** Generates a diverse pool of focus compounds.
6. **Top Seed Extraction (Deduplicated):** Deduplicates generated SMILES, filters out diversity filter failures (`Score > 0`), and ranks candidates by predicted pIC50.
7. **Mol2Mol Exploration:** Explores structural neighborhoods around the top-10 optimized hits to generate fine-tuned analogues.

---

## 📈 Optimization & Stabilization Fixes

We resolved several key RL training instabilities:
1. **Scaffold Exploitation & Score Collapse:** Lowered the `IdenticalMurckoScaffold` diversity filter `bucket_size` from `25` to `8`. The filter now penalizes overused scaffolds much faster, forcing the agent to explore novel chemical spaces sooner.
2. **Smooth Multi-Objective Transforms:**
   - **NumRotBond:** Fixed the previous degenerate transform (`low=high=8` cliff-edge) to a smooth reverse sigmoid (`low=2.0`, `high=10.0`, `k=20.0`). This prevents rigid molecule collapse while gracefully penalizing extremely floppy ones.
   - **TPSA:** Adjusted the scaling divisor (`coef_div` to `150.0`) to avoid potential negative boundary scale artifacts.
3. **Production XGBoost Scoring Guardrails:**
   - Pre-prediction shape checks ensuring feature dimensional consistency (exactly 4,482 features).
   - Graceful NaN/Inf fallbacks rather than pipeline-halting exception crashes.
   - Optional detailed molecule scoring traces via `export JAK2_DEBUG=1`.

---

## 📊 Live Monitoring with TensorBoard

You can monitor loss, QED, Molecular Weight, and projected pIC50 curves live:

### 1. Monitor Transfer Learning (TL)
```bash
tensorboard --logdir=tb_jak2_tl --port=6006
```
👉 Open: [http://localhost:6006](http://localhost:6006)

### 2. Monitor Reinforcement Learning (RL)
```bash
tensorboard --logdir=tb_jak2_rl --port=6007
```
👉 Open: [http://localhost:6007](http://localhost:6007)

---

## 🏆 Top Unique Gen Hits Output
Extracted deduplicated hits are saved to `results/top_10_hits_v2.csv` containing actual predicted pIC50 activities (e.g., highly potent leads in the **10.3 - 10.7 pIC50** range).
