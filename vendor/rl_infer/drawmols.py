#!/usr/bin/env python3
"""
Scan vis_rl / RL docking output folders and build analysis tables + plots.

Docking score is read from mol*_log.txt (pose-1 line). No docking_scorer module.

Expected layout (any depth under --root):
  vis_cpt1_files/RL_iter_0/batch_000001/mol0_log.txt
  vis_cpt1_files/RL_iter_0/batch_000001/mol0_out.sdf

Usage:
  conda activate reinvent_qsar
  cd CVAE_RL_finetuning
  python drawmols.py vis_cpt1_files/ --out-dir results/run1/

Outputs (in --out-dir, default <root>/analysis/):
  all_molecules.csv
  all_molecules_deduped.csv       (butina_cluster, is_butina_centroid)
  butina_cluster_summary.csv      (centroid = first member in each RDKit cluster)
  butina_cluster_centroids.png    (2D structures, one per cluster)
  butina_clusters/cluster_*.sdf   (one SDF per cluster)
  top100_balanced.csv
  butina_clusters.png             (scatter colored by cluster id)
  top10_molecules.png
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from pic50_scorer import readModel as readPic50Model, calculateScore as calculatePic50
from sol_scorer import readModel as readSolModel, calculateScore as calculateSolubility

# smina / vina log: first data row is pose rank 1
_FIRST_POSE_RE = re.compile(r"^\s*1\s+(-?\d+(?:\.\d+)?)")

# Composite ranking weights (pic50 + sol + docking).
WEIGHT_PIC50 = 5.0
WEIGHT_SOL = 5.0
WEIGHT_DOCK = 2.0

TOP_BALANCED_N = 100
TOP_MOLS_N = 10
DIVERSITY_MAX_TANIMOTO = 0.85  # skip if too similar to an already picked mol
BUTINA_CUTOFF = 0.4  # Tanimoto distance (1 - sim); 0.4 => cluster if sim >= 0.6


def _parse_rl_iter(path: Path) -> Optional[int]:
    for part in path.parts:
        m = re.match(r"RL_iter_(\d+)", part, re.I)
        if m:
            return int(m.group(1))
    return None


def _parse_batch(path: Path) -> Optional[int]:
    for part in path.parts:
        m = re.match(r"batch_(\d+)", part, re.I)
        if m:
            return int(m.group(1))
    return None


def _mol_id_from_stem(stem: str) -> Optional[int]:
    m = re.match(r"mol(\d+)", stem, re.I)
    return int(m.group(1)) if m else None


def _smiles_from_sdf(sdf_path: Path) -> Optional[str]:
    if not sdf_path.is_file():
        return None
    try:
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    except OSError:
        return None
    for mol in suppl:
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    return None


def _canonical_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _parse_docking_score(log_path: Path) -> float:
    """Read mol*_log.txt and return pose-1 affinity (kcal/mol)."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = _FIRST_POSE_RE.match(line.strip())
        if m:
            return float(m.group(1))
    # gnina fallback
    m = re.search(r"Affinity:\s*([-\d.]+)", text)
    if m:
        return float(m.group(1))
    return float("nan")


def _find_pose_sdf(log_path: Path) -> Optional[Path]:
    stem = log_path.name.replace("_log.txt", "")
    parent = log_path.parent
    for name in (f"{stem}_out.sdf", f"{stem}.sdf", "mol0_out.sdf"):
        p = parent / name
        if p.is_file():
            return p
    for p in parent.glob(f"{stem}*.sdf"):
        return p
    return None


def collect_records(root: Path) -> List[Dict]:
    records: List[Dict] = []
    seen_logs = set()

    for log_path in sorted(root.rglob("mol*_log.txt")):
        log_path = log_path.resolve()
        if log_path in seen_logs:
            continue
        seen_logs.add(log_path)

        stem = log_path.stem.replace("_log", "")
        mol_num = _mol_id_from_stem(stem)
        pose_sdf = _find_pose_sdf(log_path)

        smiles = None
        if pose_sdf is not None:
            smiles = _smiles_from_sdf(pose_sdf)

        if not smiles:
            continue

        docking = _parse_docking_score(log_path)
        pic50 = calculatePic50(smiles)
        sol = calculateSolubility(smiles)

        records.append({
            "rl_iter": _parse_rl_iter(log_path),
            "batch": _parse_batch(log_path),
            "mol_name": stem,
            "mol_id": mol_num,
            "SMILES": smiles,
            "canonical_smiles": _canonical_smiles(smiles) or smiles,
            "pIC50": pic50 if math.isfinite(pic50) else np.nan,
            "Solubility": sol if math.isfinite(sol) else np.nan,
            "Docking_Score": docking if math.isfinite(docking) else np.nan,
            "pose_sdf": str(pose_sdf) if pose_sdf else "",
            "log_file": str(log_path),
            "source_dir": str(log_path.parent),
        })

    return records


def _minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    vals = series.astype(float)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.5, index=series.index)
    norm = (vals - lo) / (hi - lo)
    return 1.0 - norm if invert else norm


def add_composite(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pic50_n = _minmax(out["pIC50"])
    sol_n = _minmax(out["Solubility"])
    dock_n = _minmax(out["Docking_Score"], invert=True)  # more negative = better
    out["composite"] = (
        WEIGHT_PIC50 * pic50_n + WEIGHT_SOL * sol_n + WEIGHT_DOCK * dock_n
    ) / (WEIGHT_PIC50 + WEIGHT_SOL + WEIGHT_DOCK)
    out["combined_pic50_sol"] = out["pIC50"].astype(float) + out["Solubility"].astype(float)
    return out


def dedupe_best_composite(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values("composite", ascending=False)
    return work.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)


def _morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _MORGAN_GEN.GetFingerprint(mol)


def butina_cluster_smiles(
    smiles_list: List[str],
    cutoff: float = BUTINA_CUTOFF,
) -> Tuple[List[int], List[bool]]:
    """
    RDKit Butina clustering on Morgan fingerprints (ECFP-like, radius=2).

    Returns (cluster_id per input SMILES, is_centroid per input SMILES).
    In RDKit Butina output the first member of each cluster tuple is the centroid.
    """
    fps = []
    valid_idx: List[int] = []
    for i, smi in enumerate(smiles_list):
        fp = _morgan_fp(smi)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)

    cluster_ids = [-1] * len(smiles_list)
    is_centroid = [False] * len(smiles_list)
    n_fps = len(fps)
    if n_fps == 0:
        return cluster_ids, is_centroid
    if n_fps == 1:
        cluster_ids[valid_idx[0]] = 0
        is_centroid[valid_idx[0]] = True
        return cluster_ids, is_centroid

    # Flat lower-triangle distance matrix (required when isDistData=True).
    dists: List[float] = []
    for i in range(1, n_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(dists, n_fps, cutoff, isDistData=True)
    fp_to_cluster: Dict[int, int] = {}
    centroid_fp_idxs = set()
    for cid, members in enumerate(clusters):
        centroid_fp_idxs.add(members[0])
        for m in members:
            fp_to_cluster[m] = cid

    for fp_i, orig_i in enumerate(valid_idx):
        cluster_ids[orig_i] = fp_to_cluster[fp_i]
        is_centroid[orig_i] = fp_i in centroid_fp_idxs
    return cluster_ids, is_centroid


def add_butina_clusters(df: pd.DataFrame, cutoff: float = BUTINA_CUTOFF) -> pd.DataFrame:
    out = df.copy()
    cids, centroids = butina_cluster_smiles(out["canonical_smiles"].tolist(), cutoff=cutoff)
    out["butina_cluster"] = cids
    out["is_butina_centroid"] = centroids
    return out


def butina_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per Butina cluster: size + RDKit centroid molecule."""
    rows = []
    for cid, grp in df.groupby("butina_cluster", sort=True):
        if cid < 0:
            continue
        centroids = grp[grp["is_butina_centroid"]]
        rep = centroids.iloc[0] if len(centroids) else grp.iloc[0]
        rows.append({
            "butina_cluster": int(cid),
            "size": len(grp),
            "centroid_mol_name": rep.get("mol_name", ""),
            "centroid_smiles": rep["canonical_smiles"],
            "centroid_composite": float(rep["composite"]),
            "centroid_pIC50": float(rep["pIC50"]),
            "centroid_Solubility": float(rep["Solubility"]),
            "centroid_Docking_Score": float(rep["Docking_Score"]),
        })
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


def write_butina_cluster_sdfs(df: pd.DataFrame, out_dir: Path) -> int:
    """Write one SDF per Butina cluster (RDKit-style grouped output)."""
    cluster_dir = out_dir / "butina_clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for cid, grp in df.groupby("butina_cluster", sort=True):
        if cid < 0:
            continue
        mols = []
        for _, row in grp.iterrows():
            mol = Chem.MolFromSmiles(row["canonical_smiles"])
            if mol is None:
                continue
            mol.SetProp("_Name", str(row.get("mol_name", "")))
            mol.SetProp("butina_cluster", str(int(cid)))
            mol.SetProp("is_butina_centroid", "1" if row.get("is_butina_centroid") else "0")
            mol.SetProp("composite", f"{float(row['composite']):.4f}")
            mol.SetProp("pIC50", f"{float(row['pIC50']):.3f}")
            mol.SetProp("Solubility", f"{float(row['Solubility']):.3f}")
            mol.SetProp("Docking_Score", f"{float(row['Docking_Score']):.3f}")
            mols.append(mol)
        if not mols:
            continue
        out_path = cluster_dir / f"cluster_{int(cid):04d}_n{len(mols)}.sdf"
        writer = Chem.SDWriter(str(out_path))
        for mol in mols:
            writer.write(mol)
        writer.close()
        n_written += 1
    return n_written


def plot_butina_centroids(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    out_png: Path,
    max_clusters: int = 20,
) -> None:
    """Grid of 2D structures: one RDKit Butina centroid per cluster."""
    top = summary.head(max_clusters)
    mols = []
    legends = []
    for _, row in top.iterrows():
        cid = int(row["butina_cluster"])
        mol = Chem.MolFromSmiles(row["centroid_smiles"])
        if mol is None:
            continue
        AllChem.Compute2DCoords(mol)
        mols.append(mol)
        legends.append(
            f"C{cid} (n={int(row['size'])})\n"
            f"pIC50={row['centroid_pIC50']:.2f} logS={row['centroid_Solubility']:.2f}\n"
            f"Dock={row['centroid_Docking_Score']:.2f}"
        )
    if not mols:
        return
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(5, len(mols)),
        subImgSize=(350, 350),
        legends=legends,
    )
    img.save(str(out_png))


def select_balanced(
    df: pd.DataFrame,
    n: int = TOP_BALANCED_N,
    max_tanimoto: float = DIVERSITY_MAX_TANIMOTO,
) -> pd.DataFrame:
    """Greedy diversity: high composite, max Tanimoto to picked set <= threshold."""
    work = df.sort_values("composite", ascending=False).reset_index(drop=True)
    picked: List[int] = []
    fps: List = []

    for idx, row in work.iterrows():
        if len(picked) >= n:
            break
        fp = _morgan_fp(row["canonical_smiles"])
        if fp is None:
            continue
        if not picked:
            picked.append(idx)
            fps.append(fp)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        if max(sims) <= max_tanimoto:
            picked.append(idx)
            fps.append(fp)

    out = work.iloc[picked].copy()
    out["Rank"] = range(1, len(out) + 1)
    max_tans = []
    for i, idx in enumerate(picked):
        fp = fps[i]
        if i == 0:
            max_tans.append(0.0)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps[:i])
        max_tans.append(float(max(sims)) if sims else 0.0)
    out["max_tanimoto_to_prior"] = max_tans
    return out.reset_index(drop=True)


def plot_butina_clusters(
    df: pd.DataFrame,
    out_png: Path,
    title: str = "Butina clusters (pIC50 vs Solubility)",
) -> None:
    work = df[df["butina_cluster"] >= 0].copy()
    if work.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    x = work["pIC50"].astype(float)
    y = work["Solubility"].astype(float)
    clusters = work["butina_cluster"].astype(int)
    n_clusters = clusters.nunique()
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 1))
    sc = ax.scatter(
        x,
        y,
        c=clusters,
        cmap=cmap,
        alpha=0.8,
        s=40,
        edgecolors="k",
        linewidths=0.3,
    )
    cb = fig.colorbar(sc, ax=ax, ticks=sorted(clusters.unique()))
    cb.set_label("Butina cluster id")
    ax.set_xlabel("pIC50 (predicted)")
    ax.set_ylabel("Solubility logS (predicted)")
    ax.set_title(f"{title} (n={n_clusters} clusters)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_top_molecules(df: pd.DataFrame, out_png: Path, n: int = TOP_MOLS_N) -> None:
    top = df.sort_values("composite", ascending=False).head(n)
    mols = []
    legends = []
    for i, row in top.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is None:
            continue
        AllChem.Compute2DCoords(mol)
        mols.append(mol)
        legends.append(
            f"#{len(mols)} {row.get('mol_name', '')}\n"
            f"pIC50={row['pIC50']:.2f} logS={row['Solubility']:.2f}\n"
            f"Dock={row['Docking_Score']:.2f} comp={row['composite']:.3f}"
        )
    if not mols:
        return
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(5, len(mols)),
        subImgSize=(400, 400),
        legends=legends,
    )
    img.save(str(out_png))


def main():
    p = argparse.ArgumentParser(description="Analyze vis_rl docking output folders")
    p.add_argument("root", help="Root folder, e.g. vis_cpt1_files/")
    p.add_argument("--out-dir", default=None, help="Output directory (default: <root>/analysis)")
    p.add_argument("--top-n", type=int, default=TOP_BALANCED_N)
    p.add_argument("--diversity", type=float, default=DIVERSITY_MAX_TANIMOTO)
    p.add_argument(
        "--butina-cutoff",
        type=float,
        default=BUTINA_CUTOFF,
        help="Butina Tanimoto distance cutoff (default 0.4 => sim>=0.6)",
    )
    p.add_argument(
        "--max-centroid-plots",
        type=int,
        default=20,
        help="Max Butina cluster centroids to draw (largest clusters first)",
    )
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading QSAR models...")
    readPic50Model()
    readSolModel()

    print(f"Scanning {root} ...")
    records = collect_records(root)
    if not records:
        print("No mol*_log.txt + pose SDF pairs found.")
        sys.exit(1)

    df = pd.DataFrame(records)
    df = add_composite(df)
    all_path = out_dir / "all_molecules.csv"
    df.to_csv(all_path, index=False)
    print(f"[+] {len(df)} rows -> {all_path}")

    deduped = dedupe_best_composite(df)
    deduped = add_butina_clusters(deduped, cutoff=args.butina_cutoff)
    dedup_path = out_dir / "all_molecules_deduped.csv"
    deduped.to_csv(dedup_path, index=False)
    n_butina = deduped["butina_cluster"].nunique()
    print(f"[+] {len(deduped)} unique SMILES, {n_butina} Butina clusters -> {dedup_path}")

    summary = butina_cluster_summary(deduped)
    summary_path = out_dir / "butina_cluster_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[+] Butina cluster summary -> {summary_path}")

    n_sdfs = write_butina_cluster_sdfs(deduped, out_dir)
    print(f"[+] {n_sdfs} Butina cluster SDFs -> {out_dir / 'butina_clusters'}/")

    centroid_png = out_dir / "butina_cluster_centroids.png"
    plot_butina_centroids(deduped, summary, centroid_png, max_clusters=args.max_centroid_plots)
    print(f"[+] Butina centroid structures -> {centroid_png}")

    balanced = select_balanced(deduped, n=args.top_n, max_tanimoto=args.diversity)
    bal_path = out_dir / f"top{args.top_n}_balanced.csv"
    balanced.to_csv(bal_path, index=False)
    print(f"[+] top {len(balanced)} balanced -> {bal_path}")

    cluster_png = out_dir / "butina_clusters.png"
    plot_butina_clusters(
        deduped,
        cluster_png,
        title=f"All unique mols (n={len(deduped)})",
    )
    print(f"[+] Butina cluster plot -> {cluster_png}")

    if len(balanced) > 0:
        bal_cluster_png = out_dir / f"top{args.top_n}_butina_clusters.png"
        plot_butina_clusters(balanced, bal_cluster_png, title=f"Top {len(balanced)} balanced")

    top10_png = out_dir / "top10_molecules.png"
    plot_top_molecules(deduped, top10_png, n=TOP_MOLS_N)
    print(f"[+] top 10 structures -> {top10_png}")

    print("Done.")


if __name__ == "__main__":
    main()
