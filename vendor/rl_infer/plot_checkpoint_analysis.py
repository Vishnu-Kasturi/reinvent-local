#!/usr/bin/env python3
"""
Plot checkpoint sample CSVs vs a training set (plots only by default).

Does NOT sample or generate new SMILES — reads existing per-checkpoint CSVs from
``sample_checkpoints.py`` (or any folder of sample CSVs), computes metrics in memory,
and writes PNGs.

Uses pIC50 / Solubility from each CSV when those columns exist; otherwise predicts
with the local QSAR models.

Usage:
  python plot.py SAMPLES_DIR TRAIN_CSV --out-dir results/compare/
  python plot_checkpoint_analysis.py SAMPLES_DIR TRAIN_CSV

Optional:
  --write-csv   also write checkpoint_similarity_summary.csv and *_scored.csv
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
PIC50_COL_ALIASES = ("pIC50", "pic50", "PD1PDL1pIC50", "PD1PDL1pIC50 (raw)")
SOL_COL_ALIASES = ("Solubility", "solubility", "logS", "PD1PDL1Sol", "PD1PDL1Sol (raw)")
SKIP_CSV_NAMES = frozenset({"checkpoint_similarity_summary.csv", "training_scored.csv"})


def _find_smiles_col(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for name in SMILES_COL_NAMES:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise SystemExit(f"No SMILES column found. Columns: {list(df.columns)}")


def _find_optional_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for name in aliases:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


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


def _load_molecule_table(csv_path: Path) -> pd.DataFrame:
    """One row per unique canonical SMILES; keep scores from CSV when present."""
    df = pd.read_csv(csv_path)
    smi_col = _find_smiles_col(df)
    pic50_col = _find_optional_col(df, PIC50_COL_ALIASES)
    sol_col = _find_optional_col(df, SOL_COL_ALIASES)

    rows: list[dict] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        raw = row[smi_col]
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        smi = _canonical_smiles(str(raw))
        if not smi or smi in seen:
            continue
        seen.add(smi)
        rec: dict = {"SMILES": smi}
        rec["pIC50"] = (
            pd.to_numeric(row[pic50_col], errors="coerce") if pic50_col else np.nan
        )
        rec["Solubility"] = (
            pd.to_numeric(row[sol_col], errors="coerce") if sol_col else np.nan
        )
        rows.append(rec)
    return pd.DataFrame(rows)


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
    vals = _max_tanimoto_to_training(gen_fps, train_fps)
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def _max_tanimoto_to_training(mol_fps: List, train_fps: List) -> np.ndarray:
    if not mol_fps or not train_fps:
        return np.array([], dtype=float)
    out: List[float] = []
    for fp in mol_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        out.append(max(sims) if sims else 0.0)
    return np.array(out, dtype=float)


def _max_tanimoto_within_training(train_fps: List) -> np.ndarray:
    n = len(train_fps)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([0.0], dtype=float)
    out: List[float] = []
    for i, fp in enumerate(train_fps):
        sims = list(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        sims[i] = -1.0
        out.append(max(sims))
    return np.array(out, dtype=float)


def _checkpoint_label(path: Path) -> str:
    name = path.stem
    m = re.search(r"(?:RL_)?(?:Weights_)?epoch[_-]?(\d+)", name, re.I)
    if m:
        return f"epoch_{m.group(1)}"
    return name


def _qsar_models_loaded() -> bool:
    return getattr(_fill_missing_qsar, "_models_ready", False)


def _ensure_qsar_models() -> None:
    if not _qsar_models_loaded():
        print("Loading QSAR models (pIC50 / Solubility missing in some inputs)...")
        readPic50Model()
        readSolModel()
        _fill_missing_qsar._models_ready = True  # type: ignore[attr-defined]


def _fill_missing_qsar(props: pd.DataFrame) -> pd.DataFrame:
    out = props.copy()
    missing_pic50 = out["pIC50"].isna()
    missing_sol = out["Solubility"].isna()
    if not missing_pic50.any() and not missing_sol.any():
        return out
    _ensure_qsar_models()
    for idx in out.index:
        smi = out.at[idx, "SMILES"]
        if missing_pic50.loc[idx]:
            v = calculatePic50(smi)
            out.at[idx, "pIC50"] = v if math.isfinite(v) else np.nan
        if missing_sol.loc[idx]:
            v = calculateSolubility(smi)
            out.at[idx, "Solubility"] = v if math.isfinite(v) else np.nan
    return out


_fill_missing_qsar._models_ready = False  # type: ignore[attr-defined]


def _list_sample_csvs(samples_dir: Path, pattern: str, out_dir: Path) -> List[Path]:
    paths = sorted(samples_dir.glob(pattern))
    filtered: List[Path] = []
    for p in paths:
        if p.name in SKIP_CSV_NAMES or p.stem.endswith("_scored"):
            continue
        if p.resolve().parent == out_dir.resolve():
            continue
        filtered.append(p)
    return filtered


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
    xlim: Optional[Tuple[float, float]] = None,
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
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot checkpoint sample CSVs vs training (reads existing samples; PNGs only by default)"
    )
    p.add_argument("samples_dir", help="Directory with per-checkpoint sample CSV files")
    p.add_argument("train_csv", help="Training CSV (SMILES + optional pIC50/Solubility columns)")
    p.add_argument("--out-dir", default=None, help="Directory for PNGs (default: <samples_dir>/analysis)")
    p.add_argument("--pattern", default="*.csv", help="Glob for sample CSVs in samples_dir (default: *.csv)")
    p.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write training_scored.csv, per-checkpoint *_scored.csv, and summary CSV",
    )
    args = p.parse_args()

    samples_dir = Path(args.samples_dir).expanduser().resolve()
    train_csv = Path(args.train_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else samples_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_csvs = _list_sample_csvs(samples_dir, args.pattern, out_dir)
    if not sample_csvs:
        raise SystemExit(
            f"No sample CSV files matching {args.pattern!r} in {samples_dir} "
            f"(skipped *_scored.csv and prior analysis outputs)"
        )

    print(f"Using {len(sample_csvs)} sample CSV(s) from {samples_dir}")

    print(f"Loading training set from {train_csv}")
    train_props = _load_molecule_table(train_csv)
    if train_props.empty:
        raise SystemExit(f"No valid SMILES in {train_csv}")
    train_props = _fill_missing_qsar(train_props)
    train_smiles = train_props["SMILES"].tolist()
    train_fps, train_smiles = _fps_for_smiles(train_smiles)
    train_props = train_props.set_index("SMILES").loc[train_smiles].reset_index()
    train_props["max_tanimoto"] = _max_tanimoto_within_training(train_fps)
    print(f"  {len(train_smiles)} unique valid training molecules")

    if args.write_csv:
        train_props.to_csv(out_dir / "training_scored.csv", index=False)

    summary_rows = []
    scored_generated: Dict[str, pd.DataFrame] = {}

    for csv_path in sample_csvs:
        label = _checkpoint_label(csv_path)
        props = _load_molecule_table(csv_path)
        if props.empty:
            print(f"  {label}: skip (no valid SMILES in {csv_path.name})")
            continue
        props = _fill_missing_qsar(props)
        gen_smiles = props["SMILES"].tolist()
        gen_fps, gen_smiles = _fps_for_smiles(gen_smiles)
        props = props.set_index("SMILES").loc[gen_smiles].reset_index()
        props["max_tanimoto"] = _max_tanimoto_to_training(gen_fps, train_fps)
        mean_max = float(props["max_tanimoto"].mean()) if len(props) else float("nan")

        if args.write_csv:
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
        print(f"  {label}: n={len(gen_smiles)} from {csv_path.name} mean_max_tanimoto={mean_max:.3f}")

    if not scored_generated:
        raise SystemExit("No checkpoint data to plot.")

    summary = pd.DataFrame(summary_rows)
    if args.write_csv:
        summary_path = out_dir / "checkpoint_similarity_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[+] summary -> {summary_path}")

    tanimoto_png = out_dir / "tanimoto_similarity_bar.png"
    _plot_tanimoto_bar(summary, tanimoto_png)
    print(f"[+] bar plot -> {tanimoto_png}")

    tanimoto_kde_png = out_dir / "tanimoto_kde.png"
    _plot_kde(
        train_props,
        scored_generated,
        "max_tanimoto",
        "Max Tanimoto to training set (Morgan r=2)",
        tanimoto_kde_png,
        xlim=(0.0, 1.0),
    )
    print(f"[+] Tanimoto KDE -> {tanimoto_kde_png}")

    pic50_png = out_dir / "pic50_kde.png"
    _plot_kde(train_props, scored_generated, "pIC50", "pIC50", pic50_png)
    print(f"[+] pIC50 KDE -> {pic50_png}")

    sol_png = out_dir / "solubility_kde.png"
    _plot_kde(train_props, scored_generated, "Solubility", "Solubility logS", sol_png)
    print(f"[+] solubility KDE -> {sol_png}")

    print("Done.")


if __name__ == "__main__":
    main()
