#!/usr/bin/env python3
"""
Compare checkpoint-generated SMILES to a training set.

Reads per-checkpoint CSVs (SMILES column), computes:
  - mean max Tanimoto similarity to training set (bar chart)
  - KDE of predicted pIC50 and solubility (generated vs training)

Usage:
  python plot_checkpoint_analysis.py SAMPLES_DIR TRAIN_CSV --out-dir results/compare/

SAMPLES_DIR: directory with checkpoint CSVs from sample_checkpoints.py
TRAIN_CSV:     training CSV with a smiles column (only SMILES is used)
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
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from pic50_scorer import readModel as readPic50Model, calculateScore as calculatePic50
from sol_scorer import readModel as readSolModel, calculateScore as calculateSolubility

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
SMILES_COL_NAMES = ("smiles", "canonical_smiles", "input_smiles", "SMILES")


def _find_smiles_col(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for name in SMILES_COL_NAMES:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise SystemExit(f"No SMILES column found. Columns: {list(df.columns)}")


def _canonical_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _MORGAN_GEN.GetFingerprint(mol)


def _load_unique_smiles(csv_path: Path) -> List[str]:
    df = pd.read_csv(csv_path)
    col = _find_smiles_col(df)
    seen = set()
    out: List[str] = []
    for raw in df[col].astype(str):
        if not raw or raw.lower() == "nan":
            continue
        smi = _canonical_smiles(raw)
        if smi and smi not in seen:
            seen.add(smi)
            out.append(smi)
    return out


def _fps_for_smiles(smiles_list: List[str]) -> Tuple[List, List[str]]:
    fps = []
    valid = []
    for smi in smiles_list:
        fp = _morgan_fp(smi)
        if fp is not None:
            fps.append(fp)
            valid.append(smi)
    return fps, valid


def _mean_max_tanimoto(gen_fps: List, train_fps: List) -> float:
    if not gen_fps or not train_fps:
        return float("nan")
    max_sims = []
    for fp in gen_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        max_sims.append(max(sims) if sims else 0.0)
    return float(np.mean(max_sims))


def _checkpoint_label(path: Path) -> str:
    name = path.stem
    m = re.search(r"(?:RL_)?(?:Weights_)?epoch[_-]?(\d+)", name, re.I)
    if m:
        return f"epoch_{m.group(1)}"
    return name


def _score_properties(smiles_list: List[str]) -> pd.DataFrame:
    rows = []
    for smi in smiles_list:
        pic50 = calculatePic50(smi)
        sol = calculateSolubility(smi)
        rows.append({
            "SMILES": smi,
            "pIC50": pic50 if math.isfinite(pic50) else np.nan,
            "Solubility": sol if math.isfinite(sol) else np.nan,
        })
    return pd.DataFrame(rows)


def _plot_tanimoto_bar(summary: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(summary) * 0.45), 6))
    x = np.arange(len(summary))
    ax.bar(x, summary["mean_max_tanimoto"], color="steelblue", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["checkpoint"], rotation=45, ha="right")
    ax.set_ylabel("Mean max Tanimoto to training set")
    ax.set_xlabel("Checkpoint")
    ax.set_title("Generated vs training set similarity (Morgan radius=2)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_kde(
    train_df: pd.DataFrame,
    generated: Dict[str, pd.DataFrame],
    value_col: str,
    ylabel: str,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    train_vals = train_df[value_col].dropna().astype(float)
    if len(train_vals):
        sns.kdeplot(train_vals, ax=ax, label="training", color="black", linewidth=2.5)

    cmap = plt.cm.get_cmap("tab10", max(len(generated), 1))
    for i, (label, df) in enumerate(generated.items()):
        vals = df[value_col].dropna().astype(float)
        if len(vals) < 2:
            continue
        sns.kdeplot(vals, ax=ax, label=label, color=cmap(i), linewidth=1.5, alpha=0.85)

    ax.set_xlabel(ylabel)
    ax.set_ylabel("Density")
    ax.set_title(f"{ylabel} distribution: training vs checkpoints")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Tanimoto + pIC50/solubility KDE for checkpoint samples")
    p.add_argument("samples_dir", help="Directory with per-checkpoint CSV files")
    p.add_argument("train_csv", help="Training CSV (smiles column only is used)")
    p.add_argument("--out-dir", default=None, help="Output directory (default: <samples_dir>/analysis)")
    p.add_argument("--pattern", default="*.csv", help="Glob for sample CSVs (default: *.csv)")
    args = p.parse_args()

    samples_dir = Path(args.samples_dir).expanduser().resolve()
    train_csv = Path(args.train_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else samples_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_csvs = sorted(samples_dir.glob(args.pattern))
    sample_csvs = [p for p in sample_csvs if p.name != "checkpoint_manifest.txt"]
    if not sample_csvs:
        raise SystemExit(f"No CSV files matching {args.pattern!r} in {samples_dir}")

    print("Loading QSAR models...")
    readPic50Model()
    readSolModel()

    print(f"Loading training SMILES from {train_csv}")
    train_smiles = _load_unique_smiles(train_csv)
    if not train_smiles:
        raise SystemExit(f"No valid SMILES in {train_csv}")
    train_fps, train_smiles = _fps_for_smiles(train_smiles)
    print(f"  {len(train_smiles)} unique valid training molecules")

    train_props = _score_properties(train_smiles)
    train_props.to_csv(out_dir / "training_scored.csv", index=False)

    summary_rows = []
    scored_generated: Dict[str, pd.DataFrame] = {}

    for csv_path in sample_csvs:
        label = _checkpoint_label(csv_path)
        gen_smiles = _load_unique_smiles(csv_path)
        gen_fps, gen_smiles = _fps_for_smiles(gen_smiles)
        mean_max = _mean_max_tanimoto(gen_fps, train_fps)
        props = _score_properties(gen_smiles)
        props.to_csv(out_dir / f"{csv_path.stem}_scored.csv", index=False)
        scored_generated[label] = props
        summary_rows.append({
            "checkpoint": label,
            "source_csv": csv_path.name,
            "n_smiles": len(gen_smiles),
            "mean_max_tanimoto": mean_max,
            "mean_pIC50": float(props["pIC50"].mean()),
            "mean_Solubility": float(props["Solubility"].mean()),
        })
        print(f"  {label}: n={len(gen_smiles)} mean_max_tanimoto={mean_max:.3f}")

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "checkpoint_similarity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[+] summary -> {summary_path}")

    tanimoto_png = out_dir / "tanimoto_similarity_bar.png"
    _plot_tanimoto_bar(summary, tanimoto_png)
    print(f"[+] bar plot -> {tanimoto_png}")

    pic50_png = out_dir / "pic50_kde.png"
    _plot_kde(train_props, scored_generated, "pIC50", "pIC50 (predicted)", pic50_png)
    print(f"[+] pIC50 KDE -> {pic50_png}")

    sol_png = out_dir / "solubility_kde.png"
    _plot_kde(train_props, scored_generated, "Solubility", "Solubility logS (predicted)", sol_png)
    print(f"[+] solubility KDE -> {sol_png}")

    print("Done.")


if __name__ == "__main__":
    main()
