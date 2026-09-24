#!/usr/bin/env python3
"""
Mol2Mol balanced top-N + Butina clustering (v2).

Same workflow as the standalone Tyr/dock/sol/pIC50 script, plus:
  - Keep only molecules with MW <= MAX_MW (default 600)
  - Higher-clarity PNGs (larger cells, legend font, DPI)

Required CSV columns:
  SMILES, pIC50, Solubility, Docking_Score, Tyrosine_PiStacking

Outputs:
  top{N}_balanced.csv, top_clusters.csv, top{N}_molecules.png, top_clusters.png

Edit CONFIG below, then:
  python Preprocess/scripts/select_balanced_mol2mol_v2.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FP_GEN = GetMorganGenerator(radius=2, fpSize=2048)

# ── paths (relative to repo root unless absolute) ───────────────────────────
INPUT_PATH = "iict_libinvent/mol2mol_sh1/r1_rd_top.csv"
OUTPUT_DIR = "iict_libinvent/mol2mol_rdfilter/r1_v2"

# ── selection / clustering ───────────────────────────────────────────────────
TOP_N = 500
MAX_MW = 600.0
CLUSTER_CUTOFF = 0.4
TOP_CLUSTERS = 20

# ── composite weights (need not sum to 1; normalized by sum of weights) ───
W_TYR = 0.40
W_SOL = 0.30
W_PIC50 = 0.20
W_DOCK = 0.10

# ── PNG clarity ──────────────────────────────────────────────────────────────
GRID_COLS = 10
CLUSTER_GRID_COLS = 5
SUBIMG_SIZE = (440, 400)
LEGEND_FONT_SIZE = 14
TITLE_FONT_SIZE = 18
PNG_DPI = 250
# ─────────────────────────────────────────────────────────────────────────────


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve()


def composite_score(df: pd.DataFrame) -> pd.Series:
    pic50_n = (df["pIC50"] - df["pIC50"].min()) / (
        df["pIC50"].max() - df["pIC50"].min() + 1e-9
    )
    sol_n = (df["Solubility"] - df["Solubility"].min()) / (
        df["Solubility"].max() - df["Solubility"].min() + 1e-9
    )
    tyr_n = (df["Tyrosine_PiStacking"] - df["Tyrosine_PiStacking"].min()) / (
        df["Tyrosine_PiStacking"].max() - df["Tyrosine_PiStacking"].min() + 1e-9
    )
    dock_n = 1.0 - (df["Docking_Score"] - df["Docking_Score"].min()) / (
        df["Docking_Score"].max() - df["Docking_Score"].min() + 1e-9
    )
    total_w = W_TYR + W_SOL + W_PIC50 + W_DOCK
    return (
        (W_TYR / total_w) * tyr_n
        + (W_SOL / total_w) * sol_n
        + (W_PIC50 / total_w) * pic50_n
        + (W_DOCK / total_w) * dock_n
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


def _mols_grid_image(mols, legends, cols: int):
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
    pic50_vals = df["pIC50"].values
    norm = Normalize(vmin=pic50_vals.min(), vmax=pic50_vals.max())

    for _, row in df.iterrows():
        m = Chem.MolFromSmiles(str(row["SMILES"]))
        mols.append(m if m else Chem.MolFromSmiles("C"))
        mw = row.get("MW", float("nan"))
        legends.append(
            f"MW={mw:.0f}  pIC50={row['pIC50']:.2f}\n"
            f"logS={row['Solubility']:.2f}  Tyr={row['Tyrosine_PiStacking']:.2f}\n"
            f"Dock={row['Docking_Score']:.2f}"
        )

    rows = math.ceil(len(mols) / cols)
    img = _mols_grid_image(mols, legends, cols)
    cell_h = SUBIMG_SIZE[1] / 72.0
    fig_h = rows * cell_h * 1.35 + 1.2
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(22, cols * 2.2), fig_h),
        gridspec_kw={"width_ratios": [1, 0.03]},
    )
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=12)
    sm = ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=axes[1], label="pIC50")
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("pIC50", fontsize=13)
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
            f"Tyr={row['mean_Tyrosine_PiStacking']:.2f}  pIC50={row['mean_pIC50']:.2f}\n"
            f"logS={row['mean_Solubility']:.2f}  Dock={row['mean_Docking_Score']:.2f}"
        )

    rows = math.ceil(len(mols) / cols)
    img = _mols_grid_image(mols, legends, cols)
    cell_h = SUBIMG_SIZE[1] / 72.0
    fig, ax = plt.subplots(figsize=(cols * 4.2, rows * cell_h * 1.4 + 1.5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        "Top Clusters — Centroid Structures",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=PNG_DPI, bbox_inches="tight")
    plt.close()
    print(f"saved -> {outpath}")


def main() -> None:
    output_dir = _resolve(OUTPUT_DIR)
    input_path = _resolve(INPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)

    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    df = pd.read_csv(input_path, on_bad_lines="skip", engine="python")
    required = {"SMILES", "Solubility", "pIC50", "Docking_Score", "Tyrosine_PiStacking"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}")

    print(f"loaded {len(df)} molecules from {input_path}")

    before = len(df)
    df = df.dropna(
        subset=["pIC50", "Solubility", "Docking_Score", "Tyrosine_PiStacking"]
    ).reset_index(drop=True)
    if before - len(df):
        print(f"  skipped {before - len(df)} rows missing core metrics")

    df["mol"] = df["SMILES"].astype(str).apply(Chem.MolFromSmiles)
    bad = int(df["mol"].isna().sum())
    if bad:
        print(f"  dropped {bad} unparseable SMILES")
    df = df[df["mol"].notna()].copy().reset_index(drop=True)

    df["MW"] = df["mol"].apply(Descriptors.MolWt)
    before_mw = len(df)
    df = df[df["MW"] <= MAX_MW].copy().reset_index(drop=True)
    print(f"  MW <= {MAX_MW:.0f}: kept {len(df)} / {before_mw} molecules")

    if df.empty:
        raise SystemExit(f"No molecules left after MW <= {MAX_MW} filter.")

    df["composite"] = composite_score(df)
    n_keep = min(TOP_N, len(df))
    df_top = df.nlargest(n_keep, "composite").reset_index(drop=True)
    df_top.index = df_top.index + 1

    top_csv = output_dir / f"top{n_keep}_balanced.csv"
    df_top.drop(columns=["mol"]).to_csv(top_csv, index_label="rank")
    print(f"saved -> {top_csv}")

    fps = [get_fp(m) for m in df_top["mol"]]
    clusters = butina_cluster(fps, CLUSTER_CUTOFF)

    mol_cluster: dict[int, int] = {}
    for cid, cluster in enumerate(clusters):
        for idx in cluster:
            mol_cluster[idx] = cid
    df_top["cluster_id"] = [mol_cluster.get(i, -1) for i in range(len(df_top))]

    rows = []
    for cid, cluster in enumerate(clusters):
        sub = df_top.iloc[list(cluster)]
        centroid_idx = cluster[0]
        rows.append(
            {
                "cluster_id": cid,
                "size": len(cluster),
                "centroid_SMILES": df_top.iloc[centroid_idx]["SMILES"],
                "mean_Tyrosine_PiStacking": sub["Tyrosine_PiStacking"].mean(),
                "mean_Solubility": sub["Solubility"].mean(),
                "mean_pIC50": sub["pIC50"].mean(),
                "mean_Docking_Score": sub["Docking_Score"].mean(),
                "mean_MW": sub["MW"].mean(),
                "mean_composite": sub["composite"].mean(),
            }
        )

    cluster_df = (
        pd.DataFrame(rows)
        .sort_values("mean_Tyrosine_PiStacking", ascending=False)
        .reset_index(drop=True)
    )
    cluster_csv = output_dir / "top_clusters.csv"
    cluster_df.to_csv(cluster_csv, index=False)
    print(f"saved -> {cluster_csv}  ({len(cluster_df)} clusters)")

    mol_grid_png(
        df_top,
        output_dir / f"top{n_keep}_molecules.png",
        title=f"Top {n_keep} Balanced (MW≤{int(MAX_MW)}) — Tyr | logS | pIC50 | Dock",
        cols=GRID_COLS,
    )
    cluster_grid_png(
        cluster_df.head(TOP_CLUSTERS),
        output_dir / "top_clusters.png",
        cols=CLUSTER_GRID_COLS,
    )

    print(f"\n── top {n_keep} molecules (MW <= {MAX_MW}) ──")
    print(
        df_top[
            ["SMILES", "MW", "Tyrosine_PiStacking", "Solubility", "pIC50", "Docking_Score", "composite"]
        ]
        .head(10)
        .to_string()
    )
    print("\n── top clusters (by mean Tyrosine Pi-Stacking) ──")
    print(
        cluster_df[
            [
                "cluster_id",
                "size",
                "mean_Tyrosine_PiStacking",
                "mean_Solubility",
                "mean_pIC50",
                "mean_Docking_Score",
                "mean_MW",
            ]
        ]
        .head(10)
        .to_string()
    )


if __name__ == "__main__":
    main()
