#!/usr/bin/env python3
"""
Mol2Mol balanced top-N selection + Butina clustering.

From a Mol2Mol REINVENT RL summary CSV (enriched with enrich_mol2mol_csv.py or raw):
  1. Select top N molecules by weighted composite score
  2. Cluster them by ECFP4 Butina clustering
  3. Output:
       - top{N}_balanced.csv   (TOML scoring cols + pIC50/sol only)
       - top_clusters.csv
       - top{N}_molecules.png
       - top_clusters.png

For reinvent-local-main pipeline (NOT RBDD).

Edit CONFIG below, then from repo root:
  python Preprocess/scripts/select_balanced_mol2mol.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FP_GEN = GetMorganGenerator(radius=2, fpSize=2048)

# ── paths ────────────────────────────────────────────────────────────────────
INPUT_PATH = "iict_libinvent/mol2mol_scaffold_hop_1_enriched.csv"
OUTPUT_DIR = "iict_libinvent/mol2mol_scaffold_hop_1_analysis"
SMILES_COL = "SMILES"

# ── tuneable knobs ───────────────────────────────────────────────────────────
TOP_N = 100
CLUSTER_CUTOFF = 0.4
TOP_CLUSTERS = 20
GRID_COLS = 10
CLUSTER_GRID_COLS = 5
COLOR_BY = "pIC50"

# ── module toggles ───────────────────────────────────────────────────────────
RUN_CLUSTERING = True
WRITE_TOP_CSV = True
WRITE_CLUSTER_CSV = True
WRITE_MOLECULE_PNG = True
WRITE_CLUSTER_PNG = True

# ── composite score: enable + weight per property ────────────────────────────
USE_SCAFFOLD = True
W_SCAFFOLD = 0.15

USE_TYR = True
W_TYR = 0.20

USE_SOL = True
W_SOL = 0.15

USE_PIC50 = True
W_PIC50 = 0.20

USE_DOCK = True
W_DOCK = 0.20

USE_CSP3 = False
W_CSP3 = 0.05

USE_ROT = False
W_ROT = 0.05

USE_AROMATIC = False
W_AROMATIC = 0.05

USE_MULTIRING = False
W_MULTIRING = 0.05
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from mol2mol_rl_columns import MOL2MOL_OUTPUT_COLUMNS as OUTPUT_COLUMNS  # noqa: E402

_METRICS = (
    ("scaffold_hop", USE_SCAFFOLD, W_SCAFFOLD, "ScaffoldHop",
     ["ScaffoldHop (raw)"], True),
    ("tyrosine", USE_TYR, W_TYR, "TyrInteractionReward",
     ["tyr_pi_stacking (TyrInteractionReward)", "TyrInteractionCount_raw",
      "TyrInteractionCount_raw (raw)"], True),
    ("solubility", USE_SOL, W_SOL, "Solubility",
     ["PD1PDL1Sol", "PD1PDL1Sol_raw", "logS"], True),
    ("pic50", USE_PIC50, W_PIC50, "pIC50",
     ["PD1PDL1pIC50", "PD1PDL1pIC50_raw"], True),
    ("docking", USE_DOCK, W_DOCK, "DockingReward",
     ["DockingAffinity_raw", "DockingAffinity_raw (raw)"], True),
    ("csp3", USE_CSP3, W_CSP3, "LowCsp3",
     ["LowCsp3 (raw)"], True),
    ("rotbonds", USE_ROT, W_ROT, "LowRotBonds",
     ["LowRotBonds (raw)"], True),
    ("aromatic", USE_AROMATIC, W_AROMATIC, "AromaticRings_2_4",
     ["AromaticRings_2_4 (raw)"], True),
    ("multiring", USE_MULTIRING, W_MULTIRING, "MultiRing",
     ["MultiRing (raw)"], True),
)


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve()


def _find_column(df: pd.DataFrame, primary: str, aliases: list[str]) -> str | None:
    candidates = [primary] + list(aliases)
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    for col in df.columns:
        for name in candidates:
            if name.lower() in col.lower():
                return col
    return None


def _active_metrics() -> list[tuple]:
    active = [m for m in _METRICS if m[1]]
    if not active:
        raise SystemExit("No metrics enabled — set at least one USE_* = True in CONFIG.")
    if sum(m[2] for m in active) <= 0:
        raise SystemExit("Enabled metric weights must sum to > 0.")
    return active


def _normalize(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    lo, hi = vals.min(), vals.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.5, index=series.index, dtype=float)
    return (vals - lo) / (hi - lo)


def composite_score(df: pd.DataFrame, active: list[tuple], col_map: dict[str, str]) -> pd.Series:
    score = pd.Series(0.0, index=df.index, dtype=float)
    total_w = sum(m[2] for m in active)
    for name, _, weight, _, _, higher_better in active:
        normed = _normalize(df[col_map[name]])
        if not higher_better:
            normed = 1.0 - normed
        score = score + (weight / total_w) * normed
    return score


def get_fp(mol: Chem.Mol):
    return _FP_GEN.GetFingerprint(mol)


def butina_cluster(fps, cutoff: float):
    dists: list[float] = []
    n = len(fps)
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])
    return Butina.ClusterData(dists, n, cutoff, isDistData=True)


def mol_grid_png(df, outpath, title, color_col, active, col_map, cols):
    mols, legends = [], []
    color_vals = pd.to_numeric(df[color_col], errors="coerce")
    norm = Normalize(vmin=color_vals.min(), vmax=color_vals.max())

    for _, row in df.iterrows():
        m = Chem.MolFromSmiles(str(row["SMILES"]))
        mols.append(m if m else Chem.MolFromSmiles("C"))
        parts = []
        for name, _, _, _, _, _ in active:
            val = row[col_map[name]]
            parts.append(f"{name}={val:.2f}" if pd.notna(val) else f"{name}=NA")
        legends.append("\n".join(parts))

    rows = math.ceil(len(mols) / cols)
    img = Draw.MolsToGridImage(
        mols, molsPerRow=cols, subImgSize=(300, 270), legends=legends, returnPNG=False,
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(18, rows * 2.8 + 1.5), gridspec_kw={"width_ratios": [1, 0.02]}
    )
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title(title, fontsize=14, fontweight="bold", pad=10)
    sm = ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=axes[1], label=color_col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved -> {outpath}")


def cluster_grid_png(cluster_df, outpath, active, cols):
    mols, legends = [], []
    for _, row in cluster_df.iterrows():
        m = Chem.MolFromSmiles(str(row["centroid_SMILES"]))
        mols.append(m if m else Chem.MolFromSmiles("C"))
        parts = [f"Cluster {int(row['cluster_id'])} (n={int(row['size'])})"]
        for name, _, _, _, _, _ in active:
            parts.append(f"{name}={row[f'mean_{name}']:.2f}")
        legends.append("\n".join(parts))

    rows = math.ceil(len(mols) / cols)
    img = Draw.MolsToGridImage(
        mols, molsPerRow=cols, subImgSize=(320, 290), legends=legends, returnPNG=False,
    )
    fig, ax = plt.subplots(figsize=(cols * 3.5, rows * 3.2 + 1))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Top Clusters — Centroid Structures", fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved -> {outpath}")


def _build_output_frame(df_top: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    out_cols = [c for c in OUTPUT_COLUMNS if c in source.columns]
    if "SMILES" not in out_cols:
        out_cols.insert(0, "SMILES")
    return source.loc[df_top.index, out_cols].copy()


def main():
    active = _active_metrics()
    input_path = _resolve(INPUT_PATH)
    output_dir = _resolve(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    df = pd.read_csv(input_path, on_bad_lines="skip", engine="python")
    print(f"loaded {len(df)} molecules from {input_path}")

    smi_col = _find_column(df, SMILES_COL, ["smiles", "canonical_smiles"])
    if not smi_col:
        raise SystemExit(f"SMILES column not found. Columns: {list(df.columns)}")

    col_map: dict[str, str] = {}
    for name, _, _, primary, aliases, _ in active:
        col = _find_column(df, primary, aliases)
        if not col:
            raise SystemExit(
                f"Enabled metric '{name}' column not found (tried {primary} / {aliases}). "
                f"Columns: {list(df.columns)}"
            )
        col_map[name] = col

    work = df.copy().reset_index(drop=True)
    work["SMILES"] = df[smi_col].astype(str).values
    for name, _, _, _, _, _ in active:
        work[col_map[name]] = pd.to_numeric(df[col_map[name]], errors="coerce").values

    metric_cols = [col_map[m[0]] for m in active]
    before = len(work)
    work = work.dropna(subset=metric_cols).reset_index(drop=True)
    if before - len(work):
        print(f"  skipped {before - len(work)} rows missing enabled metric data")

    work["mol"] = work["SMILES"].apply(Chem.MolFromSmiles)
    bad = work["mol"].isna().sum()
    if bad:
        print(f"  dropped {bad} unparseable SMILES")
    work = work[work["mol"].notna()].copy().reset_index(drop=True)

    if work.empty:
        raise SystemExit("No valid molecules left after filtering.")

    work["composite"] = composite_score(work, active, col_map)
    df_top = work.nlargest(min(TOP_N, len(work)), "composite").copy()

    cluster_df = pd.DataFrame()
    if RUN_CLUSTERING and len(df_top) > 1:
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
            row = {
                "cluster_id": cid,
                "size": len(cluster),
                "centroid_SMILES": df_top.iloc[centroid_idx]["SMILES"],
                "mean_composite": sub["composite"].mean(),
            }
            for name, _, _, _, _, _ in active:
                row[f"mean_{name}"] = sub[col_map[name]].mean()
            rows.append(row)

        sort_key = f"mean_{active[0][0]}"
        cluster_df = (
            pd.DataFrame(rows).sort_values(sort_key, ascending=False).reset_index(drop=True)
        )

        if WRITE_CLUSTER_CSV:
            cluster_csv = output_dir / "top_clusters.csv"
            cluster_df.to_csv(cluster_csv, index=False)
            print(f"saved -> {cluster_csv}  ({len(cluster_df)} clusters)")

    if WRITE_TOP_CSV:
        top_csv = output_dir / f"top{TOP_N}_balanced.csv"
        out_df = _build_output_frame(df_top, work)
        out_df.to_csv(top_csv, index_label="rank")
        print(f"saved -> {top_csv}")

    color_col = _find_column(df_top, COLOR_BY, [])
    if not color_col or color_col not in df_top.columns:
        color_col = col_map[active[0][0]]

    enabled = ", ".join(m[0] for m in active)
    if WRITE_MOLECULE_PNG:
        mol_grid_png(
            df_top,
            output_dir / f"top{TOP_N}_molecules.png",
            title=f"Top {TOP_N} Balanced Mol2Mol Molecules ({enabled})",
            color_col=color_col,
            active=active,
            col_map=col_map,
            cols=GRID_COLS,
        )

    if RUN_CLUSTERING and WRITE_CLUSTER_PNG and not cluster_df.empty:
        cluster_grid_png(
            cluster_df.head(TOP_CLUSTERS),
            output_dir / "top_clusters.png",
            active=active,
            cols=CLUSTER_GRID_COLS,
        )

    print(f"\n── top {TOP_N} molecules (enabled: {enabled}) ──")
    print(df_top[["SMILES"] + metric_cols + ["composite"]].head(10).to_string())
    if not cluster_df.empty:
        mean_cols = ["cluster_id", "size"] + [f"mean_{m[0]}" for m in active]
        print(f"\n── top clusters (sorted by {mean_cols[2]}) ──")
        print(cluster_df[mean_cols].head(10).to_string())


if __name__ == "__main__":
    main()
