#!/usr/bin/env python3
"""
bimodal_tanimoto_sample.py
==========================
Samples from pdl1_ph_d_tl.model checkpoints at multiple temperatures
to find/force a bimodal Tanimoto distribution vs. filtered_smiles_tanimoto.
Tries epochs 60, 70, 80 at temperatures 1.0, 1.2, 1.3, 1.5.
Plots Tanimoto histogram for each combination.
"""

import os, subprocess, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.abspath(os.path.join(HERE, ".."))
MODELS_DIR  = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
CONFIGS_DIR = os.path.join(ROOT, "REINVENT4", "configs")
REINVENT4   = os.path.join(ROOT, "REINVENT4")
REF_CSV     = os.path.join(HERE, "Data_pd1_pdl1", "filtered_smiles_tanimoto.csv")

MODEL_STEM = "pdl1_ph_d_tl"   # run1 — most likely candidate for bimodal at mid epochs
EPOCHS      = [60, 70, 80]
TEMPS       = [1.0, 1.2, 1.3, 1.5]
N_SAMPLE    = 600

os.makedirs(RESULTS_DIR, exist_ok=True)

# Reference FPs
df_ref  = pd.read_csv(REF_CSV)
ref_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(str(s)), 2, 2048)
           for s in df_ref["SMILES"].dropna() if Chem.MolFromSmiles(str(s))]
print(f"[*] Reference FPs: {len(ref_fps)}")

run_env = os.environ.copy()
results = {}   # (epoch, temp) -> max_tans

for epoch in EPOCHS:
    chkpt = os.path.join(MODELS_DIR, f"{MODEL_STEM}.model.{epoch}.chkpt")
    if not os.path.exists(chkpt):
        print(f"[skip] {chkpt} not found"); continue
    for temp in TEMPS:
        key = (epoch, temp)
        tag = f"bimodal_e{epoch}_t{str(temp).replace('.','')}"
        out_csv   = os.path.join(RESULTS_DIR, f"{tag}.csv")
        toml_path = os.path.join(CONFIGS_DIR,  f"_{tag}.toml")

        if not os.path.exists(out_csv):
            with open(toml_path, "w") as f:
                f.write(f"""run_type = "sampling"
device   = "mps"
json_out_config = "_{tag}.json"

[parameters]
model_file       = "{chkpt}"
output_file      = "{out_csv}"
num_smiles       = {N_SAMPLE}
unique_molecules = true
sample_strategy  = "multinomial"
temperature      = {temp}
""")
            subprocess.run(
                ["conda", "run", "-n", "reinvent4", "reinvent", toml_path],
                cwd=REINVENT4, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=run_env
            )

        if not os.path.exists(out_csv):
            print(f"  [fail] {key}"); continue

        smiles  = pd.read_csv(out_csv)["SMILES"].dropna().tolist()
        gen_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
                   for s in smiles if Chem.MolFromSmiles(s)]
        max_tans = [max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps)) for fp in gen_fps]
        results[key] = max_tans
        print(f"  E{epoch} T={temp}: n={len(max_tans)} | Mean={np.mean(max_tans):.3f} | "
              f"Median={np.median(max_tans):.3f}")

# ── Plot grid: epochs × temperatures ────────────────────────────────────────
nrows, ncols = len(EPOCHS), len(TEMPS)
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), sharex=True, sharey=False)
fig.suptitle("Bimodal search: pdl1_ph_d_tl run1 | Epoch × Temperature\n"
             "vs. filtered_smiles_tanimoto reference",
             fontsize=14, weight="bold", y=1.01)

bins = np.linspace(0, 1, 40)
cmap = cm.plasma(np.linspace(0.15, 0.85, ncols))

for ri, epoch in enumerate(EPOCHS):
    for ci, temp in enumerate(TEMPS):
        ax = axes[ri][ci]
        key = (epoch, temp)
        if key not in results:
            ax.set_visible(False); continue
        vals = np.array(results[key])
        ax.hist(vals, bins=bins, color=cmap[ci], edgecolor="white", lw=0.4, alpha=0.88)
        ax.axvline(np.mean(vals), color="#2c3e50", ls="--", lw=1.5, label=f"Mean:{np.mean(vals):.2f}")
        ax.axvline(0.5, color="orange", ls=":", lw=1.2, label="T=0.5")
        ax.axvline(0.85, color="red",  ls=":", lw=1.2, label="T=0.85")
        ax.set_title(f"E{epoch} | T={temp}", fontsize=11, weight="bold")
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=7)
        ax.set_xlabel("Max Tanimoto")

plt.tight_layout()
out_png = os.path.join(RESULTS_DIR, "pdl1_ph_d_tl_bimodal_search.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[+] Bimodal search plot → {out_png}")
