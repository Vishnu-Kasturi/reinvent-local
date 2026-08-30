#!/usr/bin/env python3
"""
export_top5_run5.py — Top-5 CSV exports + RDKit structure grids for iict_libinvent runs.

Creates four ranked top-5 tables and matching 5-molecule PNG grids per run folder:
  1. top5_sol_pic50_dock.csv      + top5_sol_pic50_dock.png      (composite: Sol + pIC50 + TYR + dock)
  2. top5_asp_interaction.csv     + top5_asp_interaction.png     (ASP122 typed TOTAL > 0, then composite)
  3. top5_docking.csv             + top5_docking.png             (best GNINA affinity)
  4. top5_solubility.csv          + top5_solubility.png          (best solubility)

Run:
    conda activate reinvent_qsar
    python Preprocess/scripts/export_top5_run5.py                  # run5–run8
    python Preprocess/scripts/export_top5_run5.py run6 run7 run8   # specific runs
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = _PROJECT_ROOT / "iict_libinvent"

# Default run folders (override via CLI: python export_top5_run5.py run6 run7)
DEFAULT_RUNS = ["run5", "run6", "run7", "run8"]

TOP_N = 5

WEIGHT_SOL = 5.0
WEIGHT_PIC50 = 5.0
WEIGHT_TYR = 3.0
WEIGHT_DOCK = 2.0

# Grid image settings (5 molecules in one PNG)
MOLS_PER_ROW = 5
MOL_IMG_SIZE = (480, 360)   # RDKit structure render size (pixels)
FIG_DPI = 200               # output PNG resolution
LEGEND_FONTSIZE = 11        # matplotlib text below each structure
TITLE_FONTSIZE = 14

OUTPUT_COLUMNS = [
    "rank",
    "molID",
    "SMILES",
    "pIC50",
    "Solubility",
    "Docking_Score",
    "Tyrosine_PiStacking",
    "TYR56_Interactions",
    "ASP122_HBond",
    "ASP122_SaltBridge",
    "ASP122_Anionic",
    "ASP122_CationPi",
    "ASP122_PiCation",
    "ASP122_Interactions",
    "ASP122_AllContacts",
    "pose_sdf",
    "composite",
]

EXPORTS: list[tuple[str, str, str]] = [
    ("top5_sol_pic50_dock", "Composite (Sol + pIC50 + TYR + dock)", "composite"),
    ("top5_asp_interaction", "ASP122 interaction (TOTAL > 0)", "asp"),
    ("top5_docking", "Best docking score", "docking"),
    ("top5_solubility", "Best solubility", "solubility"),
]


def _minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    vals = series.astype(float)
    lo, hi = vals.min(), vals.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.5, index=series.index)
    norm = (vals - lo) / (hi - lo)
    return 1.0 - norm if invert else norm


def _weighted_mean(scores: list[float], weights: list[float]) -> float:
    valid = [(s, w) for s, w in zip(scores, weights) if np.isfinite(s)]
    if not valid:
        return float("nan")
    s_arr = np.array([s for s, _ in valid], dtype=np.float64)
    w_arr = np.array([w for _, w in valid], dtype=np.float64)
    return float(np.average(s_arr, weights=w_arr))


def parse_prolif_summary(summary_path: Path) -> tuple[list[int], dict[int, dict], dict[int, str]]:
    """Parse asp122_prolif_summary.txt; return ASP-positive molIDs, ASP counts, pose paths."""
    text = summary_path.read_text(encoding="utf-8")
    asp_positive: list[int] = []
    asp_data: dict[int, dict] = {}
    poses: dict[int, str] = {}

    for block in text.split("========================================================================"):
        m = re.search(r"^molID\s+(\d+)", block, re.M)
        if not m:
            continue
        mol = int(m.group(1))

        pose_m = re.search(r"^Pose SDF:\s*(.+)$", block, re.M)
        if pose_m:
            pose = pose_m.group(1).strip().replace("\\", "/")
            run_m = re.search(r"run\d+/", pose)
            if run_m:
                pose = pose[run_m.start() :]
            poses[mol] = pose

        if "ASP122 interaction counts:" not in block:
            continue

        section = block.split("ASP122 interaction counts:")[1].split("Other ProLIF")[0]
        total_m = re.search(r"TOTAL:\s+(\d+)", section)
        total = int(total_m.group(1)) if total_m else 0

        def grab(name: str) -> int:
            mm = re.search(rf"{name}:\s+(\d+)", section)
            return int(mm.group(1)) if mm else 0

        other = (
            block.split("Other ProLIF contacts at ASP122:")[1].split("Detailed interactions:")[0]
            if "Other ProLIF" in block
            else ""
        )
        all_m = re.search(r"All contacts:\s+(\d+)", other)
        asp_data[mol] = {
            "ASP122_HBond": grab("HBond"),
            "ASP122_SaltBridge": grab("SaltBridge"),
            "ASP122_Anionic": grab("Anionic"),
            "ASP122_CationPi": grab("CationPi"),
            "ASP122_PiCation": grab("PiCation"),
            "ASP122_Interactions": total,
            "ASP122_AllContacts": int(all_m.group(1)) if all_m else 0,
        }
        if 0 <= mol <= 99 and total > 0:
            asp_positive.append(mol)

    return asp_positive, asp_data, poses


def add_composite(work: pd.DataFrame) -> pd.DataFrame:
    work = work.copy()
    pic50_n = _minmax(work["pIC50"])
    sol_n = _minmax(work["Solubility"])
    tyr_n = _minmax(work["Tyrosine_PiStacking"].astype(float))
    dock_n = _minmax(work["Docking_Score"], invert=True)
    work["composite"] = [
        _weighted_mean([s, p, t, d], [WEIGHT_SOL, WEIGHT_PIC50, WEIGHT_TYR, WEIGHT_DOCK])
        for s, p, t, d in zip(sol_n, pic50_n, tyr_n, dock_n)
    ]
    return work


def format_output(df: pd.DataFrame, asp_data: dict[int, dict], poses: dict[int, str]) -> pd.DataFrame:
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        mol = int(row["molID"])
        a = asp_data.get(mol, {})
        tyr = int(row["Tyrosine_PiStacking"])
        rows.append(
            {
                "rank": i,
                "molID": mol,
                "SMILES": row["SMILES"],
                "pIC50": row["pIC50"],
                "Solubility": row["Solubility"],
                "Docking_Score": row["Docking_Score"],
                "Tyrosine_PiStacking": tyr,
                "TYR56_Interactions": tyr,
                "ASP122_HBond": a.get("ASP122_HBond", 0),
                "ASP122_SaltBridge": a.get("ASP122_SaltBridge", 0),
                "ASP122_Anionic": a.get("ASP122_Anionic", 0),
                "ASP122_CationPi": a.get("ASP122_CationPi", 0),
                "ASP122_PiCation": a.get("ASP122_PiCation", 0),
                "ASP122_Interactions": a.get("ASP122_Interactions", 0),
                "ASP122_AllContacts": a.get("ASP122_AllContacts", 0),
                "pose_sdf": poses.get(mol, ""),
                "composite": row.get("composite", float("nan")),
            }
        )
    return pd.DataFrame(rows)[OUTPUT_COLUMNS]


def rank_composite(df: pd.DataFrame) -> pd.DataFrame:
    return add_composite(df).sort_values(
        ["composite", "Solubility", "pIC50", "Docking_Score"],
        ascending=[False, False, False, True],
    )


def rank_docking(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["composite"] = float("nan")
    return out.sort_values(
        ["Docking_Score", "pIC50", "Solubility"],
        ascending=[True, False, False],
    )


def rank_solubility(df: pd.DataFrame) -> pd.DataFrame:
    out = add_composite(df)
    return out.sort_values(
        ["Solubility", "pIC50", "Docking_Score"],
        ascending=[False, False, True],
    )


def rank_asp(df: pd.DataFrame, asp_positive: list[int]) -> pd.DataFrame:
    subset = df[df["molID"].isin(asp_positive)]
    return rank_composite(subset)


RANKERS: dict[str, Callable[[pd.DataFrame, list[int]], pd.DataFrame]] = {
    "composite": lambda df, _: rank_composite(df),
    "asp": lambda df, asp: rank_asp(df, asp),
    "docking": lambda df, _: rank_docking(df),
    "solubility": lambda df, _: rank_solubility(df),
}


def _mol_draw_options() -> Draw.rdMolDraw2D.MolDrawOptions:
    opts = Draw.rdMolDraw2D.MolDrawOptions()
    opts.bondLineWidth = 2.2
    opts.padding = 0.12
    opts.additionalAtomLabelPadding = 0.06
    return opts


def _mol_to_array(mol: Chem.Mol) -> np.ndarray:
    """Render molecule to a numpy RGB array at high resolution."""
    img = Draw.MolToImage(mol, size=MOL_IMG_SIZE, options=_mol_draw_options())
    return np.asarray(img)


def make_legend(row: pd.Series) -> str:
    comp = row.get("composite", float("nan"))
    comp_s = f"{comp:.2f}" if np.isfinite(comp) else "n/a"
    return (
        f"mol{int(row['molID'])}  |  rank {int(row['rank'])}\n"
        f"pIC50: {row['pIC50']:.2f}    Sol: {row['Solubility']:.2f}\n"
        f"Dock: {row['Docking_Score']:.2f}    TYR: {int(row['TYR56_Interactions'])}\n"
        f"ASP: {int(row['ASP122_Interactions'])}    comp: {comp_s}"
    )


def save_molecule_grid(df: pd.DataFrame, png_path: Path, title: str) -> None:
    entries: list[tuple[np.ndarray, str]] = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["SMILES"]))
        if mol is None:
            print(f"  WARNING: invalid SMILES for mol{row['molID']}, skipping in grid")
            continue
        entries.append((_mol_to_array(mol), make_legend(row)))

    if not entries:
        print(f"  WARNING: no valid molecules for {png_path.name}")
        return

    n = len(entries)
    cols = min(MOLS_PER_ROW, n)
    # Wide figure: structure row + legend row; high DPI for crisp text
    fig_w = 3.8 * cols
    fig_h = 5.2
    fig, axes = plt.subplots(
        2,
        cols,
        figsize=(fig_w, fig_h),
        dpi=FIG_DPI,
        gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.35, "wspace": 0.25},
    )
    if cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight="bold", y=0.98)

    for i, (arr, legend) in enumerate(entries):
        ax_mol = axes[0, i]
        ax_mol.imshow(arr)
        ax_mol.axis("off")

        ax_txt = axes[1, i]
        ax_txt.axis("off")
        ax_txt.text(
            0.5,
            0.5,
            legend,
            transform=ax_txt.transAxes,
            ha="center",
            va="center",
            fontsize=LEGEND_FONTSIZE,
            fontfamily="sans-serif",
            linespacing=1.45,
            wrap=True,
        )

    for j in range(n, cols):
        axes[0, j].axis("off")
        axes[1, j].axis("off")

    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  PNG -> {png_path}")


def export_run(run_dir: Path) -> None:
    input_csv = run_dir / "top100_balanced.csv"
    prolif_summary = run_dir / "asp122_prolif_summary.txt"

    if not input_csv.is_file():
        print(f"SKIP {run_dir.name}: missing {input_csv.name}")
        return
    if not prolif_summary.is_file():
        print(f"SKIP {run_dir.name}: missing {prolif_summary.name}")
        return

    df = pd.read_csv(input_csv)
    df["molID"] = range(len(df))
    asp_positive, asp_data, poses = parse_prolif_summary(prolif_summary)

    print(f"=== {run_dir.name} ===")
    print(f"Input:   {input_csv}")
    print(f"Output:  {run_dir}")
    print(f"ASP122+ molIDs (TOTAL>0): {asp_positive}")
    print()

    for stem, label, rank_key in EXPORTS:
        ranked = RANKERS[rank_key](df, asp_positive).head(TOP_N)
        out_df = format_output(ranked, asp_data, poses)

        csv_path = run_dir / f"{stem}.csv"
        png_path = run_dir / f"{stem}.png"
        panel_title = f"{run_dir.name} — {label}"

        out_df.to_csv(csv_path, index=False)
        print(f"[{label}]")
        print(f"  CSV -> {csv_path}")
        save_molecule_grid(out_df, png_path, panel_title)
        print()


def export_all(runs: list[str] | None = None) -> None:
    run_names = runs if runs else DEFAULT_RUNS
    for name in run_names:
        run_dir = REPO_ROOT / name
        if not run_dir.is_dir():
            print(f"SKIP {name}: folder not found at {run_dir}")
            continue
        export_run(run_dir)
    print("Done.")


if __name__ == "__main__":
    cli_runs = [a for a in sys.argv[1:] if not a.startswith("-")]
    export_all(cli_runs or None)
