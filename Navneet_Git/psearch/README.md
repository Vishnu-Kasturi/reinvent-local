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

## 2. Step-by-Step Execution Instructions

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
