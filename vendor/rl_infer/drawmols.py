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
  top100_balanced.csv
  top_clusters.png
  top10_molecules.png
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw

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
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


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


def plot_clusters(df: pd.DataFrame, out_png: Path, title: str = "pIC50 vs Solubility") -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    x = df["pIC50"].astype(float)
    y = df["Solubility"].astype(float)
    c = df["Docking_Score"].astype(float)
    sc = ax.scatter(x, y, c=c, cmap="viridis_r", alpha=0.75, s=40, edgecolors="k", linewidths=0.3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Docking score (kcal/mol)")
    ax.set_xlabel("pIC50 (predicted)")
    ax.set_ylabel("Solubility logS (predicted)")
    ax.set_title(title)
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
    dedup_path = out_dir / "all_molecules_deduped.csv"
    deduped.to_csv(dedup_path, index=False)
    print(f"[+] {len(deduped)} unique SMILES -> {dedup_path}")

    balanced = select_balanced(deduped, n=args.top_n, max_tanimoto=args.diversity)
    bal_path = out_dir / f"top{args.top_n}_balanced.csv"
    balanced.to_csv(bal_path, index=False)
    print(f"[+] top {len(balanced)} balanced -> {bal_path}")

    cluster_png = out_dir / "top_clusters.png"
    plot_clusters(
        deduped,
        cluster_png,
        title=f"All unique mols (n={len(deduped)}) — color = docking score",
    )
    print(f"[+] cluster plot -> {cluster_png}")

    if len(balanced) > 0:
        bal_cluster_png = out_dir / f"top{args.top_n}_clusters.png"
        plot_clusters(balanced, bal_cluster_png, title=f"Top {len(balanced)} balanced")

    top10_png = out_dir / "top10_molecules.png"
    plot_top_molecules(deduped, top10_png, n=TOP_MOLS_N)
    print(f"[+] top 10 structures -> {top10_png}")

    print("Done.")


if __name__ == "__main__":
    main()
