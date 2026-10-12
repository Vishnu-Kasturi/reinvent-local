#!/usr/bin/env python3
"""
From an enriched RL / Butina CSV, pick the single best molecule per cluster id.

Expects columns:
  butina_cluster (or cluster_id)
  composite or combined_pic50_sol (ranking)
  pIC50, Solubility, Docking_Score
  Tyrosine column optional (Tyrosine_PiStacking, tyr_pi_stacking, etc.)
  SMILES or canonical_smiles

Usage:
  python Preprocess/scripts/select_one_per_butina_cluster.py --input run.csv --out-dir picks/
  python Preprocess/scripts/select_one_per_butina_cluster.py --clusters 0,5,3,23
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

RDLogger.DisableLog("rdApp.*")

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_CSV = _REPO_ROOT / "iict_libinvent/enriched_butina.csv"
OUTPUT_DIR = _REPO_ROOT / "iict_libinvent/four_cluster_picks"
CLUSTER_IDS = [0, 5, 3, 23]
RANK_COLUMN = "composite"  # fallback: combined_pic50_sol
SMILES_PREF = ("canonical_smiles", "SMILES", "smiles")
TYR_ALIASES = (
    "Tyrosine_PiStacking",
    "tyr_pi_stacking",
    "Tyrosine_PiStacking_count",
    "TyrInteractionCount_raw",
)
SUBIMG_SIZE = (480, 420)
LEGEND_FONT_SIZE = 13
TITLE_FONT_SIZE = 16
PNG_DPI = 250
GRID_COLS = 2
# ─────────────────────────────────────────────────────────────────────────────


def _resolve(path: Path) -> Path:
    p = path.expanduser()
    return p.resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve()


def _find_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _smiles_series(df: pd.DataFrame) -> pd.Series:
    for name in SMILES_PREF:
        col = _find_col(df, (name,))
        if col:
            return df[col].astype(str)
    raise ValueError(f"No SMILES column. Columns: {list(df.columns)}")


def _rank_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    alt = _find_col(df, ("combined_pic50_sol", "composite", "score"))
    if alt:
        return alt
    raise ValueError(
        f"No rank column (tried {preferred}, combined_pic50_sol). Columns: {list(df.columns)}"
    )


def _cluster_col(df: pd.DataFrame) -> str:
    col = _find_col(df, ("butina_cluster", "cluster_id", "Butina_cluster"))
    if col:
        return col
    raise ValueError("No butina_cluster / cluster_id column.")


def _tyr_col(df: pd.DataFrame) -> str | None:
    return _find_col(df, TYR_ALIASES)


def pick_one_per_cluster(
    df: pd.DataFrame,
    cluster_ids: list[int],
    rank_col: str,
    cluster_col: str,
) -> pd.DataFrame:
    work = df.copy()
    work["_rank"] = pd.to_numeric(work[rank_col], errors="coerce")
    work[cluster_col] = pd.to_numeric(work[cluster_col], errors="coerce").astype("Int64")

    rows = []
    for cid in cluster_ids:
        sub = work[work[cluster_col] == cid].dropna(subset=["_rank"])
        if sub.empty:
            print(f"  WARNING: no rows for cluster {cid}")
            continue
        best = sub.nlargest(1, "_rank").iloc[0]
        rows.append(best)
        print(
            f"  cluster {cid}: {best.get('mol_name', best.get('mol_id', '?'))} "
            f"rank={best['_rank']:.4f}"
        )

    if not rows:
        raise SystemExit("No molecules selected — check cluster ids and CSV.")

    out = pd.DataFrame(rows).drop(columns=["_rank"], errors="ignore")
    return out.reset_index(drop=True)


def write_png(
    picks: pd.DataFrame,
    smiles_col: str,
    cluster_col: str,
    outpath: Path,
    tyr_col: str | None,
) -> None:
    mols, legends = [], []
    pic50 = pd.to_numeric(picks["pIC50"], errors="coerce")
    norm = Normalize(vmin=pic50.min(), vmax=pic50.max())

    for _, row in picks.iterrows():
        smi = str(row[smiles_col])
        m = Chem.MolFromSmiles(smi)
        mols.append(m if m else Chem.MolFromSmiles("C"))
        cid = int(row[cluster_col])
        tyr_txt = "n/a"
        if tyr_col and tyr_col in row.index and pd.notna(row[tyr_col]):
            tyr_txt = f"{float(row[tyr_col]):.0f}"
        legends.append(
            f"cluster {cid}  {row.get('mol_name', '')}\n"
            f"pIC50={float(row['pIC50']):.2f}  logS={float(row['Solubility']):.2f}\n"
            f"Dock={float(row['Docking_Score']):.2f}  Tyr={tyr_txt}"
        )

    cols = min(GRID_COLS, len(mols))
    rows = math.ceil(len(mols) / cols)
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=cols,
        subImgSize=SUBIMG_SIZE,
        legends=legends,
        legendFontSize=LEGEND_FONT_SIZE,
        returnPNG=False,
    )
    cell_h = SUBIMG_SIZE[1] / 72.0
    fig, axes = plt.subplots(
        1, 2,
        figsize=(cols * 5.5, rows * cell_h * 1.5 + 1),
        gridspec_kw={"width_ratios": [1, 0.03]},
    )
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Best molecule per Butina cluster", fontsize=TITLE_FONT_SIZE, fontweight="bold")
    sm = ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=axes[1], label="pIC50")
    plt.tight_layout()
    plt.savefig(outpath, dpi=PNG_DPI, bbox_inches="tight")
    plt.close()
    print(f"saved -> {outpath}")


def main() -> None:
    p = argparse.ArgumentParser(description="Pick best mol per Butina cluster id")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--clusters",
        type=str,
        default=",".join(str(c) for c in CLUSTER_IDS),
        help="Comma-separated butina_cluster ids (default: 0,5,3,23)",
    )
    p.add_argument("--rank-by", type=str, default=RANK_COLUMN)
    args = p.parse_args()

    inp = _resolve(args.input or INPUT_CSV)
    out_dir = _resolve(args.out_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    cluster_ids = [int(x.strip()) for x in args.clusters.split(",") if x.strip()]

    df = pd.read_csv(inp, on_bad_lines="skip", engine="python")
    print(f"Loaded {len(df)} rows from {inp}")

    cluster_col = _cluster_col(df)
    rank_col = _rank_col(df, args.rank_by)
    tyr_col = _tyr_col(df)
    if tyr_col:
        print(f"Tyrosine column: {tyr_col}")
    else:
        print("Tyrosine column not found — PNG will show Tyr=n/a")

    picks = pick_one_per_cluster(df, cluster_ids, rank_col, cluster_col)
    picks["SMILES_out"] = _smiles_series(picks).values

    n = len(picks)
    csv_path = out_dir / f"best_per_cluster_{n}.csv"
    picks.to_csv(csv_path, index=False)
    print(f"saved -> {csv_path}")

    write_png(picks, picks["SMILES_out"], out_dir / f"best_per_cluster_{n}.png", tyr_col)


if __name__ == "__main__":
    main()
