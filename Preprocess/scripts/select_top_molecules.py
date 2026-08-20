#!/usr/bin/env python3
"""
select_top_molecules.py — Filter, score, and rank REINVENT RL CSV hits.

Pipeline:
  1. Keep molecules with exactly N TYR56 pi-pi interactions (default: 2)
  2. Compute ASP122 interactions via ProLIF (filter to ASP122+ by default)
  3. Rank by weighted geometric-mean composite (Sol 5, pIC50 4, Tyr 3, Dock 2)

Output columns:
  rank, SMILES, pIC50, Solubility, Docking_Score, Tyrosine_PiStacking,
  ASP122_Interaction, composite

Usage:
    python Preprocess/scripts/select_top_molecules.py results/libinvent_results_5_1.csv \\
        -o top_hits.csv --top 50 --run-prolif \\
        --receptor /path/receptor.pdb --docking-runs docking_runs

    # Re-dock SMILES missing from docking_runs (slow)
    python Preprocess/scripts/select_top_molecules.py input.csv -o out.csv --run-prolif \\
        --receptor /path/receptor.pdb --autobox /path/ref_ligand.pdb --gnina gnina
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reinvent_gnina_backend import (  # noqa: E402
    GninaProlifConfig,
    _smiles_hash,
    affinity_to_reward,
    analyze_asp_interactions,
    canonicalize,
    run_molecule_pipeline,
    tyr_count_to_reward,
)
from prolif_compat import residue_ids  # noqa: E402


# RL reward weights (match iict_new_reward.toml)
WEIGHT_SOL = 5.0
WEIGHT_PIC50 = 4.0
WEIGHT_TYR = 3.0
WEIGHT_DOCK = 2.0

# Raw-value normalization bounds (from PD1-PDL1 scoring components)
PIC50_MIN, PIC50_MAX = 4.01, 11.0
SOL_MIN, SOL_MAX = -13.17, 2.14

COLUMN_ALIASES = {
    "smiles": ["SMILES", "smiles"],
    "tyr_count": [
        "TyrInteractionCount_raw (raw)",
        "TyrInteractionCount_raw",
        "tyr_pi_stacking (TyrInteractionReward)",
    ],
    "tyr_reward": ["TyrInteractionReward (raw)", "TyrInteractionReward"],
    "pic50": ["PD1PDL1pIC50 (raw)", "PD1PDL1pIC50_raw", "PD1PDL1pIC50"],
    "pic50_reward": ["PD1PDL1pIC50"],
    "sol": ["PD1PDL1Sol (raw)", "PD1PDL1Sol_raw", "PD1PDL1Sol"],
    "sol_reward": ["PD1PDL1Sol"],
    "docking": ["DockingAffinity_raw (raw)", "DockingAffinity_raw", "DockingAffinity_raw"],
    "dock_reward": ["DockingReward (raw)", "DockingReward"],
    "asp122": ["ASP122_interaction", "Asp122Interaction", "ASP122 (raw)", "ASP122"],
    "score": ["Score", "score", "total_score"],
}


def _find_column(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in cols:
            return cols[alias.lower()]
    for col in df.columns:
        for alias in aliases:
            if alias.lower() in col.lower():
                return col
    return None


def _to_float(val, default=float("nan")) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _to_bool_yes(val) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    if s in ("0", "0.0", "false", "no", "n", ""):
        return False
    try:
        return float(s) > 0
    except ValueError:
        return s in ("1", "1.0", "true", "yes", "y")


def _normalize_raw(val: float, lo: float, hi: float) -> float:
    if not np.isfinite(val):
        return float("nan")
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


def _weighted_geometric_mean(scores: list[float], weights: list[float]) -> float:
    valid = [(s, w) for s, w in zip(scores, weights) if np.isfinite(s)]
    if not valid:
        return float("nan")
    s_arr = np.array([max(s, 1e-8) for s, _ in valid], dtype=np.float64)
    w_arr = np.array([w for _, w in valid], dtype=np.float64)
    w_sum = w_arr.sum()
    if w_sum <= 0:
        return float("nan")
    return float(np.prod(s_arr ** (w_arr / w_sum)))


def find_docked_sdf(smiles: str, docking_runs: str) -> Optional[str]:
    can = canonicalize(smiles)
    if not can:
        return None
    h = _smiles_hash(can)
    pattern = os.path.join(docking_runs, "**", f"mol_*_{h}", "mol0_out.sdf")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def compute_asp122_for_smiles(
    smiles: str,
    receptor_pdb: str,
    docking_runs: str,
    asp_resid: int = 122,
    config: Optional[GninaProlifConfig] = None,
) -> tuple[int, str]:
    """Return (asp_interaction_count, path_to_sdf_used)."""
    sdf = find_docked_sdf(smiles, docking_runs)
    if sdf and os.path.isfile(sdf):
        count, _ = analyze_asp_interactions(receptor_pdb, sdf, smiles, asp_residue=asp_resid)
        return count, sdf

    if config is None:
        return 0, ""

    res = run_molecule_pipeline(smiles, 0, 0, config)
    if not res.docking_ok or not os.path.isfile(res.out_sdf):
        return 0, ""
    count, _ = analyze_asp_interactions(
        receptor_pdb, res.out_sdf, smiles, asp_residue=asp_resid
    )
    return count, res.out_sdf


def _reward_or_normalize(
    raw_val: float,
    reward_col_val: float,
    raw_lo: float,
    raw_hi: float,
    tier_fn=None,
) -> float:
    """Prefer transformed RL reward column; else normalize raw or apply tier fn."""
    if np.isfinite(reward_col_val):
        return float(reward_col_val)
    if tier_fn is not None:
        try:
            return float(tier_fn(int(raw_val) if np.isfinite(raw_val) else 0))
        except (TypeError, ValueError):
            pass
    if tier_fn is not None and np.isfinite(raw_val):
        return float(tier_fn(raw_val))
    return _normalize_raw(raw_val, raw_lo, raw_hi)


def compute_composite_row(
    pic50_raw: float,
    sol_raw: float,
    dock_raw: float,
    tyr_raw: float,
    pic50_reward: float = float("nan"),
    sol_reward: float = float("nan"),
    dock_reward: float = float("nan"),
    tyr_reward: float = float("nan"),
) -> float:
    sol_s = _reward_or_normalize(sol_raw, sol_reward, SOL_MIN, SOL_MAX)
    pic50_s = _reward_or_normalize(pic50_raw, pic50_reward, PIC50_MIN, PIC50_MAX)
    tyr_s = _reward_or_normalize(tyr_raw, tyr_reward, 0, 2, tier_fn=tyr_count_to_reward)
    dock_s = (
        dock_reward
        if np.isfinite(dock_reward)
        else affinity_to_reward(dock_raw)
    )
    return _weighted_geometric_mean(
        [sol_s, pic50_s, tyr_s, dock_s],
        [WEIGHT_SOL, WEIGHT_PIC50, WEIGHT_TYR, WEIGHT_DOCK],
    )


def format_output(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename columns for final CSV."""
    out = pd.DataFrame({
        "rank": range(1, len(df) + 1),
        "SMILES": df["_smiles"],
        "pIC50": df["_pic50"],
        "Solubility": df["_sol"],
        "Docking_Score": df["_dock"],
        "Tyrosine_PiStacking": df["_tyr"].astype("Int64"),
        "ASP122_Interaction": df["_asp122_count"].astype("Int64"),
        "composite": df["_composite"],
    })
    return out


def select_top(
    df: pd.DataFrame,
    tyr_target: int = 2,
    require_asp122: bool = True,
    top_n: int = 50,
    receptor_pdb: Optional[str] = None,
    docking_runs: Optional[str] = None,
    gnina_config: Optional[GninaProlifConfig] = None,
    asp_resid: int = 122,
    compute_asp: bool = False,
) -> pd.DataFrame:
    col_smiles = _find_column(df, COLUMN_ALIASES["smiles"])
    col_tyr = _find_column(df, COLUMN_ALIASES["tyr_count"])
    col_tyr_r = _find_column(df, COLUMN_ALIASES["tyr_reward"])
    col_pic50 = _find_column(df, COLUMN_ALIASES["pic50"])
    col_pic50_r = _find_column(df, COLUMN_ALIASES["pic50_reward"])
    col_sol = _find_column(df, COLUMN_ALIASES["sol"])
    col_sol_r = _find_column(df, COLUMN_ALIASES["sol_reward"])
    col_dock = _find_column(df, COLUMN_ALIASES["docking"])
    col_dock_r = _find_column(df, COLUMN_ALIASES["dock_reward"])
    col_asp = _find_column(df, COLUMN_ALIASES["asp122"])

    if not col_smiles:
        raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")

    work = df.copy()
    print(f"Input rows: {len(work)}")

    # Step 1: TYR pi-pi count == tyr_target
    if col_tyr:
        work["_tyr"] = work[col_tyr].apply(_to_float).round().astype(int)
        before = len(work)
        work = work[work["_tyr"] == tyr_target]
        print(f"After TYR56 pi-pi == {tyr_target}: {len(work)} (removed {before - len(work)})")
    else:
        print("WARNING: No TyrInteractionCount column — skipping TYR filter")
        work["_tyr"] = 0

    if work.empty:
        return work

    # Step 2: ASP122 interactions
    if compute_asp and receptor_pdb and (docking_runs or gnina_config):
        print(f"Computing ASP{asp_resid} via ProLIF (unique SMILES)...")
        unique_smiles = work[col_smiles].dropna().unique()
        asp_cache: dict[str, int] = {}
        for i, smi in enumerate(unique_smiles, 1):
            asp_cache[smi], _ = compute_asp122_for_smiles(
                smi,
                receptor_pdb,
                docking_runs or gnina_config.output_root,
                asp_resid=asp_resid,
                config=gnina_config,
            )
            if i % 10 == 0 or i == len(unique_smiles):
                n_pos = sum(1 for v in asp_cache.values() if v > 0)
                print(f"  ProLIF ASP{asp_resid}: {i}/{len(unique_smiles)} ({n_pos} with interactions)")
        work["_asp122_count"] = work[col_smiles].map(asp_cache).fillna(0).astype(int)
        before = len(work)
        if require_asp122:
            work = work[work["_asp122_count"] > 0]
        print(f"After ASP{asp_resid} (ProLIF): {len(work)} (removed {before - len(work)})")
    elif col_asp:
        work["_asp122_count"] = work[col_asp].apply(
            lambda v: int(_to_float(v, 0)) if _to_float(v, 0) == int(_to_float(v, 0)) else int(_to_bool_yes(v))
        )
        before = len(work)
        if require_asp122:
            work = work[work["_asp122_count"] > 0]
        print(f"After ASP{asp_resid} (from CSV): {len(work)} (removed {before - len(work)})")
    else:
        print("WARNING: ASP122 not computed — pass --run-prolif --receptor or add ASP122 column")
        work["_asp122_count"] = 0
        if require_asp122:
            work = work.iloc[0:0]

    if work.empty:
        return work

    # Step 3: composite score and rank
    work["_smiles"] = work[col_smiles]
    work["_pic50"] = work[col_pic50].apply(_to_float) if col_pic50 else float("nan")
    work["_sol"] = work[col_sol].apply(_to_float) if col_sol else float("nan")
    work["_dock"] = work[col_dock].apply(_to_float) if col_dock else float("nan")
    work["_pic50_r"] = work[col_pic50_r].apply(_to_float) if col_pic50_r else float("nan")
    work["_sol_r"] = work[col_sol_r].apply(_to_float) if col_sol_r else float("nan")
    work["_dock_r"] = work[col_dock_r].apply(_to_float) if col_dock_r else float("nan")
    work["_tyr_r"] = work[col_tyr_r].apply(_to_float) if col_tyr_r else float("nan")

    work["_composite"] = work.apply(
        lambda row: compute_composite_row(
            row["_pic50"],
            row["_sol"],
            row["_dock"],
            row["_tyr"],
            pic50_reward=row["_pic50_r"],
            sol_reward=row["_sol_r"],
            dock_reward=row["_dock_r"],
            tyr_reward=row["_tyr_r"],
        ),
        axis=1,
    )

    work = work.sort_values(
        by=["_composite", "_pic50", "_sol", "_dock"],
        ascending=[False, False, False, True],
        na_position="last",
    )

    if top_n > 0:
        work = work.head(top_n)

    return format_output(work)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select top RL molecules from CSV")
    parser.add_argument("input_csv", help="REINVENT results CSV")
    parser.add_argument("-o", "--output", default="top_molecules.csv", help="Output CSV path")
    parser.add_argument("--top", type=int, default=50, help="Number of top molecules to keep")
    parser.add_argument("--tyr-count", type=int, default=2, help="Required TYR56 pi-pi count (default: 2)")
    parser.add_argument("--asp-residue", type=int, default=122, help="ASP residue number (default: 122)")
    parser.add_argument("--no-asp-required", action="store_true", help="Do not require ASP122 interaction")
    parser.add_argument(
        "--run-prolif",
        action="store_true",
        help="Compute ASP122 via ProLIF (required if ASP122 not in CSV)",
    )
    parser.add_argument("--receptor", default=os.environ.get("RECEPTOR_PDB", ""), help="Receptor PDB for ProLIF")
    parser.add_argument("--docking-runs", default="docking_runs", help="Folder with mol0_out.sdf from RL")
    parser.add_argument("--autobox", default="", help="Ref ligand for GNINA (only if re-docking needed)")
    parser.add_argument("--gnina", default="gnina", help="GNINA executable")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    print(f"Columns: {list(df.columns)}\n")
    print(f"ASP{args.asp_residue} ProLIF residues: {residue_ids('ASP', args.asp_residue)}\n")

    gnina_config = None
    if args.run_prolif:
        if not args.receptor:
            sys.exit("ERROR: --receptor required for --run-prolif")
        if args.autobox:
            gnina_config = GninaProlifConfig(
                receptor_path=args.receptor,
                autobox_ligand=args.autobox,
                gnina_executable=args.gnina,
                output_root=args.docking_runs,
                keep_outputs=True,
            )

    result = select_top(
        df,
        tyr_target=args.tyr_count,
        require_asp122=not args.no_asp_required,
        top_n=args.top,
        receptor_pdb=args.receptor or None,
        docking_runs=args.docking_runs if args.run_prolif else None,
        gnina_config=gnina_config,
        asp_resid=args.asp_residue,
        compute_asp=args.run_prolif,
    )

    if result.empty:
        print("\nNo molecules passed filters.")
        sys.exit(1)

    result.to_csv(args.output, index=False)
    print(f"\nSaved {len(result)} molecules → {args.output}")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
