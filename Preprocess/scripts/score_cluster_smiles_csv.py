#!/usr/bin/env python3
"""
Score a SMILES CSV (pIC50, Solubility, GNINA dock, Tyr pi-stacking) → top-N + Butina + PNGs.

Input: CSV with column smiles / SMILES / canonical_smiles / input_smiles
Output directory:
  scored_all.csv
  top{N}_molecules.csv
  top_clusters.csv
  top{N}_molecules.png
  top_clusters.png

Usage:
  python Preprocess/scripts/score_cluster_smiles_csv.py --input my.csv --out-dir results/run1
  python Preprocess/scripts/score_cluster_smiles_csv.py   # uses CONFIG below
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "Preprocess" / "scripts"
sys.path.insert(0, str(_REPO_ROOT / "REINVENT4"))
sys.path.insert(0, str(_SCRIPTS))

from reinvent_gnina_backend import BatchCache, GninaProlifConfig  # noqa: E402

_FP_GEN = GetMorganGenerator(radius=2, fpSize=2048)
SMILES_ALIASES = ("SMILES", "smiles", "canonical_smiles", "input_smiles")

# ── CONFIG (defaults when CLI flags omitted) ─────────────────────────────────
INPUT_CSV = _REPO_ROOT / "iict_libinvent/input.smi.csv"
OUTPUT_DIR = _REPO_ROOT / "iict_libinvent/score_cluster_out"
TOP_N = 100
TOP_CLUSTERS = 20
CLUSTER_CUTOFF = 0.4
MAX_MW = 0.0  # 0 = no filter; else drop MW > MAX_MW

RECEPTOR = "/home/genai/navneet/iict/pdl1/docking_TL_dataset/receptor.pdb"
AUTOBOX = "/home/genai/navneet/iict/pdl1/docking_TL_dataset/ref_ligand.pdb"
GNINA = "/home/genai/Documents/gnina/gnina"
DOCKING_ROOT = _REPO_ROOT / "docking_runs"

W_TYR = 0.40
W_SOL = 0.30
W_PIC50 = 0.20
W_DOCK = 0.10

GRID_COLS = 10
CLUSTER_GRID_COLS = 5
SUBIMG_SIZE = (440, 400)
LEGEND_FONT_SIZE = 14
TITLE_FONT_SIZE = 18
PNG_DPI = 250
XGB_DEVICE = "cuda:0"
# ─────────────────────────────────────────────────────────────────────────────

PIC50_MODEL = _REPO_ROOT / "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_model.ubj"
PIC50_SCALER = _REPO_ROOT / "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_scaler.pkl"
SOL_MODEL = _REPO_ROOT / "Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj"
SOL_SCALER = _REPO_ROOT / "Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl"


def _find_smiles_col(df: pd.DataFrame) -> str:
    lower = {str(c).lower(): c for c in df.columns}
    for name in SMILES_ALIASES:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise ValueError(f"No SMILES column. Columns: {list(df.columns)}")


def _load_booster(path: Path, device: str) -> xgb.Booster:
    booster = xgb.Booster({"device": device})
    booster.load_model(str(path))
    return booster


def _predict_qsar(
    smiles: list[str],
    model: xgb.Booster,
    scaler: Path,
    feature_fn,
) -> list[float]:
    features, mask = feature_fn(smiles, str(scaler))
    preds = model.predict(xgb.DMatrix(features, nthread=-1))
    return [
        round(float(preds[i]), 4) if mask[i] and np.isfinite(preds[i]) else np.nan
        for i in range(len(smiles))
    ]


def _norm(series: pd.Series) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce")
    lo, hi = v.min(), v.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.5, index=series.index, dtype=float)
    return (v - lo) / (hi - lo)


def composite_score(df: pd.DataFrame) -> pd.Series:
    total = W_TYR + W_SOL + W_PIC50 + W_DOCK
    dock_n = 1.0 - _norm(df["Docking_Score"])
    return (
        (W_TYR / total) * _norm(df["Tyrosine_PiStacking"])
        + (W_SOL / total) * _norm(df["Solubility"])
        + (W_PIC50 / total) * _norm(df["pIC50"])
        + (W_DOCK / total) * dock_n
    )


def get_fp(mol: Chem.Mol):
    return _FP_GEN.GetFingerprint(mol)


def butina_cluster(fps, cutoff: float):
    dists: list[float] = []
    n = len(fps)
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])
    return Butina.ClusterData(dists, n, cutoff, isDistData=True)


def _grid(mols, legends, cols: int):
    return Draw.MolsToGridImage(
        mols,
        molsPerRow=cols,
        subImgSize=SUBIMG_SIZE,
        legends=legends,
        legendFontSize=LEGEND_FONT_SIZE,
        returnPNG=False,
    )


def mol_grid_png(df: pd.DataFrame, outpath: Path, title: str, cols: int) -> None:
    mols, legends = [], []
    pic50_vals = pd.to_numeric(df["pIC50"], errors="coerce")
    norm = Normalize(vmin=pic50_vals.min(), vmax=pic50_vals.max())

    for _, row in df.iterrows():
        m = Chem.MolFromSmiles(str(row["SMILES"]))
        mols.append(m if m else Chem.MolFromSmiles("C"))
        legends.append(
            f"pIC50={row['pIC50']:.2f}  logS={row['Solubility']:.2f}\n"
            f"Tyr={row['Tyrosine_PiStacking']:.0f}  Dock={row['Docking_Score']:.2f}\n"
            f"MW={row['MW']:.0f}  score={row['composite']:.3f}"
        )

    rows = math.ceil(len(mols) / cols)
    img = _grid(mols, legends, cols)
    cell_h = SUBIMG_SIZE[1] / 72.0
    fig, axes = plt.subplots(
        1, 2, figsize=(max(22, cols * 2.2), rows * cell_h * 1.35 + 1.2),
        gridspec_kw={"width_ratios": [1, 0.03]},
    )
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=12)
    sm = ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=axes[1], label="pIC50")
    plt.tight_layout()
    plt.savefig(outpath, dpi=PNG_DPI, bbox_inches="tight")
    plt.close()
    print(f"saved -> {outpath}")


def cluster_grid_png(cluster_df: pd.DataFrame, outpath: Path, cols: int) -> None:
    mols, legends = [], []
    for _, row in cluster_df.iterrows():
        m = Chem.MolFromSmiles(str(row["centroid_SMILES"]))
        mols.append(m if m else Chem.MolFromSmiles("C"))
        legends.append(
            f"Cluster {int(row['cluster_id'])}  n={int(row['size'])}\n"
            f"pIC50={row['mean_pIC50']:.2f}  logS={row['mean_Solubility']:.2f}\n"
            f"Tyr={row['mean_Tyrosine_PiStacking']:.1f}  Dock={row['mean_Docking_Score']:.2f}"
        )

    rows = math.ceil(len(mols) / cols)
    img = _grid(mols, legends, cols)
    cell_h = SUBIMG_SIZE[1] / 72.0
    fig, ax = plt.subplots(figsize=(cols * 4.2, rows * cell_h * 1.4 + 1.5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Top cluster centroids", fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=PNG_DPI, bbox_inches="tight")
    plt.close()
    print(f"saved -> {outpath}")


def score_and_cluster(
    input_csv: Path,
    output_dir: Path,
    top_n: int,
    top_clusters: int,
    max_mw: float,
) -> None:
    from reinvent_plugins.components.nophyschem_features import (
        compute_features as compute_pic50_features,
    )
    from reinvent_plugins.components.pd1_pdl1_features import (
        compute_features as compute_sol_features,
    )

    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)

    df_in = pd.read_csv(input_csv, on_bad_lines="skip", engine="python")
    smi_col = _find_smiles_col(df_in)
    print(f"Loaded {len(df_in)} rows from {input_csv} (column {smi_col})")

    work = df_in.copy()
    work["SMILES"] = work[smi_col].astype(str).str.strip()
    work = work[work["SMILES"].str.len() > 0]
    work = work[work["SMILES"].str.lower() != "nan"]
    work["mol"] = work["SMILES"].apply(Chem.MolFromSmiles)
    work = work[work["mol"].notna()].copy()
    work["MW"] = work["mol"].apply(Descriptors.MolWt)
    if max_mw > 0:
        before = len(work)
        work = work[work["MW"] <= max_mw].copy()
        print(f"  MW <= {max_mw}: {len(work)} / {before}")

    work = work.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)
    if work.empty:
        raise SystemExit("No valid SMILES after filtering.")

    smiles_list = work["SMILES"].tolist()
    print(f"Scoring {len(smiles_list)} unique molecules...")

    pic50_model = _load_booster(PIC50_MODEL, XGB_DEVICE)
    sol_model = _load_booster(SOL_MODEL, XGB_DEVICE)
    work["pIC50"] = _predict_qsar(smiles_list, pic50_model, PIC50_SCALER, compute_pic50_features)
    work["Solubility"] = _predict_qsar(smiles_list, sol_model, SOL_SCALER, compute_sol_features)

    dock_cfg = GninaProlifConfig(
        receptor_path=RECEPTOR,
        autobox_ligand=AUTOBOX,
        gnina_executable=GNINA,
        output_root=str(DOCKING_ROOT),
        cnn_scoring="none",
        keep_outputs=True,
        tyr_residue=56,
    )
    dock_cfg.validate()

    print("Docking + Tyr (GNINA + ProLIF)...")
    dock_results = BatchCache.get_or_run(smiles_list, dock_cfg)
    work["Docking_Score"] = [
        float(r.affinity) if r.docking_ok and np.isfinite(r.affinity) else np.nan
        for r in dock_results
    ]
    work["Tyrosine_PiStacking"] = [
        float(r.tyr_pi_stacking_count) if r.docking_ok else np.nan for r in dock_results
    ]
    work["docking_ok"] = [bool(r.docking_ok) for r in dock_results]

    scored = work.drop(columns=["mol"])
    metric_cols = ["pIC50", "Solubility", "Docking_Score", "Tyrosine_PiStacking"]
    before = len(scored)
    scored = scored.dropna(subset=metric_cols).reset_index(drop=True)
    if before - len(scored):
        print(f"  dropped {before - len(scored)} rows with missing scores")

    os.makedirs(output_dir, exist_ok=True)
    all_path = output_dir / "scored_all.csv"
    scored.to_csv(all_path, index=False)
    print(f"saved -> {all_path}")

    scored["composite"] = composite_score(scored)
    n_keep = min(top_n, len(scored))
    top = scored.nlargest(n_keep, "composite").reset_index(drop=True)

    top_path = output_dir / f"top{n_keep}_molecules.csv"
    top.to_csv(top_path, index=False)
    print(f"saved -> {top_path}")

    top_v = top.reset_index(drop=True)
    fps = []
    for smi in top_v["SMILES"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fps.append(get_fp(mol))

    if len(fps) <= 1:
        clusters = (tuple(range(len(fps))),) if fps else ()
    else:
        clusters = butina_cluster(fps, CLUSTER_CUTOFF)
    mol_cluster: dict[int, int] = {}
    for cid, cluster in enumerate(clusters):
        for idx in cluster:
            mol_cluster[idx] = cid
    top_v["cluster_id"] = [mol_cluster.get(i, -1) for i in range(len(top_v))]

    rows = []
    for cid, cluster in enumerate(clusters):
        sub = top_v.iloc[list(cluster)]
        centroid_idx = cluster[0]
        rows.append(
            {
                "cluster_id": cid,
                "size": len(cluster),
                "centroid_SMILES": top_v.iloc[centroid_idx]["SMILES"],
                "mean_pIC50": sub["pIC50"].mean(),
                "mean_Solubility": sub["Solubility"].mean(),
                "mean_Tyrosine_PiStacking": sub["Tyrosine_PiStacking"].mean(),
                "mean_Docking_Score": sub["Docking_Score"].mean(),
                "mean_composite": sub["composite"].mean(),
            }
        )

    cluster_df = (
        pd.DataFrame(rows).sort_values("mean_composite", ascending=False).reset_index(drop=True)
    )
    cluster_path = output_dir / "top_clusters.csv"
    cluster_df.to_csv(cluster_path, index=False)
    print(f"saved -> {cluster_path} ({len(cluster_df)} clusters)")

    mol_grid_png(
        top_v,
        output_dir / f"top{n_keep}_molecules.png",
        title=f"Top {n_keep} molecules (Tyr|logS|pIC50|Dock)",
        cols=GRID_COLS,
    )
    n_cl_png = min(top_clusters, len(cluster_df))
    cluster_grid_png(
        cluster_df.head(n_cl_png),
        output_dir / "top_clusters.png",
        cols=CLUSTER_GRID_COLS,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Score SMILES CSV + top-N Butina + PNGs")
    p.add_argument("--input", type=Path, default=None, help="Input CSV")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument("--top-clusters", type=int, default=TOP_CLUSTERS)
    p.add_argument("--max-mw", type=float, default=MAX_MW)
    args = p.parse_args()

    inp = (args.input or INPUT_CSV).expanduser().resolve()
    if not inp.is_absolute():
        inp = (_REPO_ROOT / inp).resolve()
    out = (args.out_dir or OUTPUT_DIR).expanduser().resolve()
    if not out.is_absolute():
        out = (_REPO_ROOT / out).resolve()

    score_and_cluster(inp, out, args.top_n, args.top_clusters, args.max_mw)


if __name__ == "__main__":
    main()
