#!/usr/bin/env python3
"""
RD-filters CSV → pIC50 + Solubility → balanced top-N → Butina clusters + PNGs.

Input: output from rd_filters / apply_rd_filters (e.g. *_passed.csv or flagged CSV)
  Required: SMILES
  Typical extras: rd_filter_pass, property_pass, alert_set_score, MW, LogP, ...

Outputs (under OUTPUT_DIR):
  enriched.csv          — all kept rows + pIC50, Solubility, composite
  top{N}_balanced.csv
  top_clusters.csv
  top{N}_molecules.png
  top_clusters.png

  python Preprocess/scripts/enrich_cluster_rd_filter_csv.py
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
import xgboost as xgb
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FP_GEN = GetMorganGenerator(radius=2, fpSize=2048)

# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_CSV = _REPO_ROOT / "iict_libinvent/rd_filtered_compounds.csv"
OUTPUT_DIR = _REPO_ROOT / "iict_libinvent/rd_filter_clustered"
SMILES_COL = "SMILES"

ONLY_RD_PASS = True
ONLY_PROPERTY_PASS = True
MAX_MW = 600.0

TOP_N = 100
CLUSTER_CUTOFF = 0.4
TOP_CLUSTERS = 20

W_PIC50 = 0.45
W_SOL = 0.35
W_ALERT = 0.20

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

PASS_THROUGH_COLS = [
    "rd_filter_pass",
    "property_pass",
    "alert_set_score",
    "alert_sets_passed",
    "alert_sets_failed",
    "filters_passed",
    "filters_failed",
    "MW",
    "LogP",
    "HBD",
    "HBA",
    "TPSA",
    "Rot",
]


def _resolve_smiles_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for c in df.columns:
        if c.lower() == preferred.lower():
            return c
    raise ValueError(f"SMILES column not found. Columns: {list(df.columns)}")


def _norm(series: pd.Series) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce")
    lo, hi = v.min(), v.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.5, index=series.index, dtype=float)
    return (v - lo) / (hi - lo)


def composite_score(df: pd.DataFrame) -> pd.Series:
    total = W_PIC50 + W_SOL + W_ALERT
    score = (W_PIC50 / total) * _norm(df["pIC50"])
    score = score + (W_SOL / total) * _norm(df["Solubility"])
    if "alert_set_score" in df.columns:
        score = score + (W_ALERT / total) * _norm(df["alert_set_score"])
    return score


def _load_booster(path: Path, device: str) -> xgb.Booster:
    booster = xgb.Booster({"device": device})
    booster.load_model(str(path))
    return booster


def _predict(smiles: list[str], model: xgb.Booster, scaler: Path, feature_fn) -> list[float]:
    features, mask = feature_fn(smiles, str(scaler))
    preds = model.predict(xgb.DMatrix(features, nthread=-1))
    return [
        round(float(preds[i]), 4) if mask[i] and np.isfinite(preds[i]) else np.nan
        for i in range(len(smiles))
    ]


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
        mw = row.get("MW", row.get("mw_calc", float("nan")))
        alert = row.get("alert_set_score", float("nan"))
        legends.append(
            f"MW={mw:.0f}  pIC50={row['pIC50']:.2f}\n"
            f"logS={row['Solubility']:.2f}  alert={alert:.2f}\n"
            f"LogP={row.get('LogP', float('nan')):.2f}  TPSA={row.get('TPSA', float('nan')):.0f}"
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
    cbar = plt.colorbar(sm, cax=axes[1], label="pIC50")
    cbar.ax.tick_params(labelsize=12)
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
            f"alert={row.get('mean_alert_set_score', float('nan')):.2f}"
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


def main() -> None:
    sys.path.insert(0, str(_REPO_ROOT / "REINVENT4"))
    from reinvent_plugins.components.nophyschem_features import (
        compute_features as compute_pic50_features,
    )
    from reinvent_plugins.components.pd1_pdl1_features import (
        compute_features as compute_sol_features,
    )

    if not INPUT_CSV.is_file():
        raise SystemExit(f"Input not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, on_bad_lines="skip", engine="python")
    smi_col = _resolve_smiles_col(df, SMILES_COL)
    print(f"loaded {len(df)} rows from {INPUT_CSV}")

    work = df.copy()
    work["SMILES"] = work[smi_col].astype(str)

    if ONLY_RD_PASS and "rd_filter_pass" in work.columns:
        work = work[work["rd_filter_pass"].astype(str).str.lower().isin(("true", "1", "yes"))]

    if ONLY_PROPERTY_PASS and "property_pass" in work.columns:
        work = work[work["property_pass"].astype(str).str.lower().isin(("true", "1", "yes"))]

    work["mol"] = work["SMILES"].apply(Chem.MolFromSmiles)
    work = work[work["mol"].notna()].copy()
    work["mw_calc"] = work["mol"].apply(Descriptors.MolWt)
    if "MW" not in work.columns:
        work["MW"] = work["mw_calc"]
    before = len(work)
    work = work[work["mw_calc"] <= MAX_MW].copy()
    print(f"  MW <= {MAX_MW}: {len(work)} / {before}")

    if work.empty:
        raise SystemExit("No rows left after filters.")

    smiles_list = work["SMILES"].tolist()
    pic50_model = _load_booster(PIC50_MODEL, XGB_DEVICE)
    sol_model = _load_booster(SOL_MODEL, XGB_DEVICE)
    work["pIC50"] = _predict(smiles_list, pic50_model, PIC50_SCALER, compute_pic50_features)
    work["Solubility"] = _predict(smiles_list, sol_model, SOL_SCALER, compute_sol_features)
    work = work.dropna(subset=["pIC50", "Solubility"]).copy()
    work["composite"] = composite_score(work)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    enriched_cols = ["SMILES", "pIC50", "Solubility", "composite", "mw_calc"]
    for c in PASS_THROUGH_COLS:
        if c in work.columns:
            enriched_cols.append(c)
    enriched_path = OUTPUT_DIR / "enriched.csv"
    work.drop(columns=["mol"]).loc[:, enriched_cols].to_csv(enriched_path, index=False)
    print(f"saved -> {enriched_path}")

    n_keep = min(TOP_N, len(work))
    df_top = work.nlargest(n_keep, "composite").reset_index(drop=True)

    top_path = OUTPUT_DIR / f"top{n_keep}_balanced.csv"
    df_top.drop(columns=["mol"]).to_csv(top_path, index_label="rank")
    print(f"saved -> {top_path}")

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
            "mean_pIC50": sub["pIC50"].mean(),
            "mean_Solubility": sub["Solubility"].mean(),
            "mean_composite": sub["composite"].mean(),
            "mean_MW": sub["MW"].mean(),
        }
        if "alert_set_score" in sub.columns:
            row["mean_alert_set_score"] = sub["alert_set_score"].mean()
        rows.append(row)

    cluster_df = (
        pd.DataFrame(rows).sort_values("mean_composite", ascending=False).reset_index(drop=True)
    )
    cluster_path = OUTPUT_DIR / "top_clusters.csv"
    cluster_df.to_csv(cluster_path, index=False)
    print(f"saved -> {cluster_path} ({len(cluster_df)} clusters)")

    mol_grid_png(
        df_top,
        OUTPUT_DIR / f"top{n_keep}_molecules.png",
        title=f"Top {n_keep} (MW≤{int(MAX_MW)}, RD pass) — pIC50 | logS | alerts",
        cols=GRID_COLS,
    )
    cluster_grid_png(
        cluster_df.head(TOP_CLUSTERS),
        OUTPUT_DIR / "top_clusters.png",
        cols=CLUSTER_GRID_COLS,
    )


if __name__ == "__main__":
    main()
