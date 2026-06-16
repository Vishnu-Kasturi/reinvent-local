# PSearch Screening & Clustering Pipeline for ChEMBL 10K ErG Similarity

This directory contains the self-contained pipeline scripts, processed datasets, and commands to run preprocessing, conformer generation, virtual screening, and clustering on the top 10,000 ErG similarity compounds from the ChEMBL database.

---

## 1. Prerequisites & Dependencies

The pipeline requires `RDKit`, `Pandas`, and the `psearch` package (along with its dependency `pmapper`). 

To install the dependencies, use a conda environment:
```bash
# Create and activate environment
conda create -n psearch_env python=3.10 -y
conda activate psearch_env

# Install RDKit and Pandas
conda install -c conda-forge rdkit pandas -y

# Install PSearch/PMapper (or follow your system installation instructions)
pip install pmapper
pip install git+https://github.com/mti-lab/psearch.git
```

---

## 2. Execution Commands (Standard Setup)

### Step 1: Preprocess ChEMBL & ErG Similarity Filtering
Neutralizes charges, strips salts, canonicalizes SMILES, and selects the top 10,000 compounds most similar to `mol57` using **ErG** fingerprints.
```bash
python preprocess_and_filter_chembl.py -i <path_to_chembl_data.csv> -o chembl_top10000_erg.smi -s chembl_top10000_erg_scores.csv -c 8
```
*Outputs:* `chembl_top10000_erg.smi` and `chembl_top10000_erg_scores.csv`

### Step 2: 3D Conformer Generation Database
Generates 3D conformers for the selected 10,000 compounds using optimized settings (`-n 10` conformers, `-s 1` stereoisomer).
```bash
gen_db -i chembl_top10000_erg.smi -o chembl_top10000_erg.dat -n 10 -s 1 -c 8 -v
```
*Outputs:* `chembl_top10000_erg.dat` and `chembl_top10000_erg.dir/`

### Step 3: Run Virtual Screening
Screens the conformers against the 3 pharmacophore models.
```bash
python run_chembl_top10000_screening.py -d chembl_top10000_erg.dat -i chembl_top10000_erg.smi -q <path_to_models_directory> -o screening_results_top10000
```
*Outputs:* 
- `hits_psearch.t0_f6_p0_top10000.csv`
- `hits_psearch.t1_f8_p0_top10000.csv`
- `hits_psearch.t1_f8_p1_top10000.csv`

### Step 4: Clustering Hits
Runs RDKit Butina clustering (cutoff=0.40) on the virtual screening hits to group similar hits and output centroid grid images.
```bash
python cluster_top10000_hits.py
```
*Outputs:* 
- `hits_psearch.t0_f6_p0_top10000_clusters.png`
- `hits_psearch.t1_f8_p0_top10000_clusters.png`
- `hits_psearch.t1_f8_p1_top10000_clusters.png`

---

## 3. High-Performance Configuration (Scaling on a DGX)

If you are running the pipeline on a high-performance system like a **DGX**, you can significantly scale up the number of conformers, stereoisomers, and CPU threads to generate a much more robust conformer database.

### Recommended DGX Command
```bash
# 1. Run Preprocessing utilizing more cores (e.g., 64)
python preprocess_and_filter_chembl.py -i <path_to_chembl_data.csv> -c 64

# 2. Run gen_db with 50 conformers, 5 stereoisomers, and 64 cores
gen_db -i chembl_top10000_erg.smi -o chembl_top10000_erg_hd.dat -n 50 -s 5 -c 64 -v
```

### Parameter Explanations & Guidelines

#### 1. CPU Cores (`-c` / `--ncpu`)
- **What it does:** Specifies the number of parallel CPU processes to spawn.
- **DGX Recommendation:** Set this to `-c 64` or higher depending on the available CPU slots on your node. 
- **Effect:** Parallelizes the computational bottleneck (conformer embedding and force-field optimization) to finish the database creation in minutes instead of hours.

#### 2. Number of Conformers (`-n` / `--nconf`)
- **What it does:** Sets the maximum number of 3D conformers generated and optimized per stereoisomer of a compound.
- **DGX Recommendation:** Set this to `-n 50` (or up to `100`).
- **Effect:** Generating 50 conformers ensures high conformational space coverage, increasing the likelihood of capturing the bioactive conformation during 3D pharmacophore matching.
- **Disk Space note:** A 10,000 compound dataset with 50 conformers will require approximately **8-12 GB of disk space**, which fits easily on DGX storage.

#### 3. Maximum Stereoisomers (`-s` / `--nstereo`)
- **What it does:** Sets the maximum number of stereoisomers to enumerate for compounds that have chiral centers with *unspecified* stereochemistry (centers with defined stereochemistry in the input SMILES will remain unaltered).
- **DGX Recommendation:** Set this to `-s 4` or `-s 8` (e.g., `-s 5` is the default).
- **Effect:** This ensures you generate conformers for different stereoisomers of a molecule, in case the active compound is a specific enantiomer or diastereomer that was not explicitly defined in the SMILES file.
- **Computation note:** If a compound has $k$ unspecified chiral centers, it can have up to $2^k$ stereoisomers. Setting `-s 5` limits this to a maximum of 5 stereoisomers, preventing combinatorial explosion for compounds with many unspecified chiral centers.
