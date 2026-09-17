#!/usr/bin/env python3
"""
LinkInvent balanced top-N selection + Butina clustering.

From a LinkInvent / REINVENT results CSV (enriched or raw RL summary):
  1. Select top N molecules by weighted composite score (per-metric toggles in TOML)
  2. Cluster them by ECFP4 Butina clustering
  3. Output (each toggled in TOML):
       - top{N}_balanced.csv
       - top_clusters.csv
       - top{N}_molecules.png
       - top_clusters.png

For reinvent-local-main pipeline (NOT RBDD).

Run from repo root:
  python Preprocess/scripts/select_balanced_linkinvent.py
  python Preprocess/scripts/select_balanced_linkinvent.py --config Preprocess/configs/linkinvent_balanced_analysis.toml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

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

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

RDLogger.DisableLog("rdApp.*")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "Preprocess/configs/linkinvent_balanced_analysis.toml"
_FP_GEN = GetMorganGenerator(radius=2, fpSize=2048)


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


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


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Config not found: {path}")
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _active_components(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    comps = cfg.get("composite", [])
    active = [c for c in comps if c.get("enabled", False)]
    if not active:
        raise SystemExit("No composite components enabled in TOML (set enabled = true on at least one [[composite]] row).")
    total_w = sum(float(c.get("weight", 0.0)) for c in active)
    if total_w <= 0:
        raise SystemExit("Enabled composite weights must sum to > 0.")
    return active


def _normalize(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    lo, hi = vals.min(), vals.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.5, index=series.index, dtype=float)
    return (vals - lo) / (hi - lo)


def composite_score(df: pd.DataFrame, active: list[dict[str, Any]], col_map: dict[str, str]) -> pd.Series:
    score = pd.Series(0.0, index=df.index, dtype=float)
    total_w = sum(float(c.get("weight", 0.0)) for c in active)

    for comp in active:
        name = comp["name"]
        col = col_map[name]
        w = float(comp.get("weight", 0.0)) / total_w
        normed = _normalize(df[col])
        if not comp.get("higher_is_better", True):
            normed = 1.0 - normed
        score = score + w * normed

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


def _legend_line(label: str, value: float) -> str:
    if pd.isna(value):
        return f"{label}=NA"
    return f"{label}={value:.2f}" if abs(value) < 100 else f"{label}={value:.1f}"


def mol_grid_png(
    df: pd.DataFrame,
    outpath: Path,
    title: str,
    color_col: str,
    active: list[dict[str, Any]],
    col_map: dict[str, str],
    cols: int,
) -> None:
    mols: list[Chem.Mol] = []
    legends: list[str] = []

    color_vals = pd.to_numeric(df[color_col], errors="coerce")
    norm = Normalize(vmin=color_vals.min(), vmax=color_vals.max())
    cmap = plt.cm.RdYlGn

    for _, row in df.iterrows():
        m = Chem.MolFromSmiles(str(row["SMILES"]))
        mols.append(m if m else Chem.MolFromSmiles("C"))
        parts = [_legend_line(comp["name"], row[col_map[comp["name"]]]) for comp in active]
        legends.append("\n".join(parts))

    rows = math.ceil(len(mols) / cols)
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=cols,
        subImgSize=(300, 270),
        legends=legends,
        returnPNG=False,
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(18, rows * 2.8 + 1.5),
        gridspec_kw={"width_ratios": [1, 0.02]},
    )
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title(title, fontsize=14, fontweight="bold", pad=10)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=axes[1], label=color_col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved -> {outpath}")


def cluster_grid_png(cluster_df: pd.DataFrame, outpath: Path, active: list[dict[str, Any]], cols: int) -> None:
    mols: list[Chem.Mol] = []
    legends: list[str] = []

    for _, row in cluster_df.iterrows():
        m = Chem.MolFromSmiles(str(row["centroid_SMILES"]))
        mols.append(m if m else Chem.MolFromSmiles("C"))
        parts = [f"Cluster {int(row['cluster_id'])} (n={int(row['size'])})"]
        for comp in active:
            mean_col = f"mean_{comp['name']}"
            if mean_col in row.index:
                parts.append(_legend_line(comp["name"], row[mean_col]))
        legends.append("\n".join(parts))

    rows = math.ceil(len(mols) / cols)
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=cols,
        subImgSize=(320, 290),
        legends=legends,
        returnPNG=False,
    )

    fig, ax = plt.subplots(figsize=(cols * 3.5, rows * 3.2 + 1))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Top Clusters — Centroid Structures", fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved -> {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LinkInvent balanced selection + Butina clustering")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="TOML config path (default: Preprocess/configs/linkinvent_balanced_analysis.toml)",
    )
    args = parser.parse_args()

    cfg_path = _resolve(args.config, _REPO_ROOT)
    cfg = load_config(cfg_path)

    paths = cfg.get("paths", {})
    selection = cfg.get("selection", {})
    clustering = cfg.get("clustering", {})
    outputs = cfg.get("outputs", {})
    plot_cfg = cfg.get("plot", {})

    input_csv = _resolve(paths.get("input_csv", ""), _REPO_ROOT)
    output_dir = _resolve(paths.get("output_dir", "results/linkinvent_analysis"), _REPO_ROOT)
    smiles_col = paths.get("smiles_col", "SMILES")

    top_n = int(selection.get("top_n", 100))
    cluster_enabled = bool(clustering.get("enabled", True))
    cluster_cutoff = float(clustering.get("cutoff", 0.4))
    top_clusters = int(clustering.get("top_clusters", 20))

    grid_cols = int(plot_cfg.get("grid_cols", 10))
    cluster_grid_cols = int(plot_cfg.get("cluster_grid_cols", 5))
    color_by = plot_cfg.get("color_by", "pIC50")

    active = _active_components(cfg)

    if not input_csv.is_file():
        raise SystemExit(f"Input not found: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    print(f"loaded {len(df)} rows from {input_csv}")

    smi_col = _find_column(df, smiles_col, ["smiles", "canonical_smiles"])
    if not smi_col:
        raise SystemExit(f"SMILES column not found. Columns: {list(df.columns)}")

    col_map: dict[str, str] = {}
    for comp in active:
        col = _find_column(df, comp.get("column", ""), comp.get("column_aliases", []))
        if not col:
            raise SystemExit(
                f"Enabled component '{comp['name']}' column not found "
                f"(tried {comp.get('column')} / {comp.get('column_aliases', [])}). "
                f"Columns: {list(df.columns)}"
            )
        col_map[comp["name"]] = col

    work = pd.DataFrame()
    work["SMILES"] = df[smi_col].astype(str)
    for comp in active:
        work[col_map[comp["name"]]] = pd.to_numeric(df[col_map[comp["name"]]], errors="coerce")

    metric_cols = [col_map[c["name"]] for c in active]
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
        raise SystemExit("No valid molecules left after filtering. Check SMILES and enabled metric columns.")

    work["composite"] = composite_score(work, active, col_map)
    df_top = work.nlargest(min(top_n, len(work)), "composite").reset_index(drop=True)
    df_top.index = df_top.index + 1

    enabled_names = ", ".join(c["name"] for c in active)
    top_csv_name = f"top{top_n}_balanced.csv"

    if outputs.get("write_top_csv", True):
        top_csv = output_dir / top_csv_name
        df_top.drop(columns=["mol"]).to_csv(top_csv, index_label="rank")
        print(f"saved -> {top_csv}")

    cluster_df = pd.DataFrame()
    if cluster_enabled and len(df_top) > 1:
        fps = [get_fp(m) for m in df_top["mol"]]
        clusters = butina_cluster(fps, cluster_cutoff)

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
            for comp in active:
                row[f"mean_{comp['name']}"] = sub[col_map[comp["name"]]].mean()
            rows.append(row)

        sort_key = f"mean_{active[0]['name']}"
        cluster_df = pd.DataFrame(rows).sort_values(sort_key, ascending=False).reset_index(drop=True)

        if outputs.get("write_cluster_csv", True):
            cluster_csv = output_dir / "top_clusters.csv"
            cluster_df.to_csv(cluster_csv, index=False)
            print(f"saved -> {cluster_csv}  ({len(cluster_df)} clusters)")

    color_col = col_map.get(color_by) or _find_column(df_top, color_by, [])
    if not color_col:
        color_col = col_map[active[0]["name"]]

    title = f"Top {top_n} Balanced Molecules ({enabled_names})"
    if outputs.get("write_molecule_png", True):
        mol_grid_png(
            df_top,
            output_dir / f"top{top_n}_molecules.png",
            title=title,
            color_col=color_col,
            active=active,
            col_map=col_map,
            cols=grid_cols,
        )

    if cluster_enabled and outputs.get("write_cluster_png", True) and not cluster_df.empty:
        cluster_grid_png(
            cluster_df.head(top_clusters),
            output_dir / "top_clusters.png",
            active=active,
            cols=cluster_grid_cols,
        )

    print(f"\n── config: {cfg_path} ──")
    print(f"── enabled components: {enabled_names} ──")
    preview_cols = ["SMILES"] + metric_cols + ["composite"]
    print(df_top[preview_cols].head(10).to_string())
    if not cluster_df.empty:
        mean_cols = ["cluster_id", "size"] + [f"mean_{c['name']}" for c in active]
        print(f"\n── top clusters (sorted by {mean_cols[2]}) ──")
        print(cluster_df[mean_cols].head(10).to_string())


if __name__ == "__main__":
    main()
