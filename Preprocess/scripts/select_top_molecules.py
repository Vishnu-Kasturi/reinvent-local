#!/usr/bin/env python3
"""
select_top_molecules.py — Post-RL molecule selection (not a REINVENT reward).

Filters RL results CSV, computes ASP122 via ProLIF, ranks top hits.

Run (no CLI args needed — edit CONFIG below):
    conda activate reinvent_qsar
    cd ~/Vishnu/psearch-master/reinvent-local-main
    python Preprocess/scripts/select_top_molecules.py

Output columns:
    rank, SMILES, pIC50, Solubility, Docking_Score, Tyrosine_PiStacking,
    ASP122_Interaction, composite
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — edit these paths/settings only
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_CSV = REPO_ROOT / "results" / "libinvent_results_5_1.csv"
OUTPUT_CSV = REPO_ROOT / "top_hits.csv"

TOP_N = 50
TYR_TARGET = 2          # require exactly this many TYR56 pi-pi stacks
ASP_RESIDUE = 122
REQUIRE_ASP122 = True   # keep only molecules with ASP122 interaction
RUN_PROLIF = True         # compute ASP122 from docked poses

RECEPTOR_PDB = "/home/genai/navneet/iict/pdl1/docking_TL_dataset/receptor.pdb"
AUTOBOX_LIGAND = "/home/genai/navneet/iict/pdl1/docking_TL_dataset/ref_ligand.pdb"
GNINA_EXECUTABLE = "/home/genai/Documents/gnina/gnina"
DOCKING_RUNS = REPO_ROOT / "docking_runs"

# Selection ranking weights (higher = more important in composite score)
WEIGHT_SOL = 5.0
WEIGHT_PIC50 = 4.0
WEIGHT_TYR = 3.0
WEIGHT_DOCK = 2.0

# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reinvent_gnina_backend import (  # noqa: E402
    GninaProlifConfig,
    _smiles_hash,
    analyze_asp_interactions,
    canonicalize,
    run_molecule_pipeline,
)
from prolif_compat import residue_ids  # noqa: E402

COLUMN_ALIASES = {
    "smiles": ["SMILES", "smiles"],
    "tyr_count": [
        "TyrInteractionCount_raw (raw)",
        "TyrInteractionCount_raw",
        "tyr_pi_stacking (TyrInteractionReward)",
    ],
    "pic50": ["PD1PDL1pIC50 (raw)", "PD1PDL1pIC50_raw", "PD1PDL1pIC50"],
    "sol": ["PD1PDL1Sol (raw)", "PD1PDL1Sol_raw", "PD1PDL1Sol"],
    "docking": ["DockingAffinity_raw (raw)", "DockingAffinity_raw"],
    "asp122": ["ASP122_interaction", "Asp122Interaction", "ASP122 (raw)", "ASP122"],
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


def _minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    """Normalize column to [0, 1] within the filtered set."""
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


def format_output(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "rank": range(1, len(df) + 1),
        "SMILES": df["_smiles"],
        "pIC50": df["_pic50"],
        "Solubility": df["_sol"],
        "Docking_Score": df["_dock"],
        "Tyrosine_PiStacking": df["_tyr"].astype("Int64"),
        "ASP122_Interaction": df["_asp122_count"].astype("Int64"),
        "composite": df["_composite"],
    })


def select_top(
    df: pd.DataFrame,
    tyr_target: int = TYR_TARGET,
    require_asp122: bool = REQUIRE_ASP122,
    top_n: int = TOP_N,
    receptor_pdb: Optional[str] = None,
    docking_runs: Optional[str] = None,
    gnina_config: Optional[GninaProlifConfig] = None,
    asp_resid: int = ASP_RESIDUE,
    compute_asp: bool = RUN_PROLIF,
) -> pd.DataFrame:
    col_smiles = _find_column(df, COLUMN_ALIASES["smiles"])
    col_tyr = _find_column(df, COLUMN_ALIASES["tyr_count"])
    col_pic50 = _find_column(df, COLUMN_ALIASES["pic50"])
    col_sol = _find_column(df, COLUMN_ALIASES["sol"])
    col_dock = _find_column(df, COLUMN_ALIASES["docking"])
    col_asp = _find_column(df, COLUMN_ALIASES["asp122"])

    if not col_smiles:
        raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")

    work = df.copy()
    print(f"Input rows: {len(work)}")

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

    if compute_asp and receptor_pdb and (docking_runs or gnina_config):
        print(f"Computing ASP{asp_resid} via ProLIF (unique SMILES)...")
        unique_smiles = work[col_smiles].dropna().unique()
        asp_cache: dict[str, int] = {}
        for i, smi in enumerate(unique_smiles, 1):
            asp_cache[smi], _ = compute_asp122_for_smiles(
                smi,
                receptor_pdb,
                str(docking_runs or gnina_config.output_root),
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
        print("WARNING: ASP122 not computed — set RUN_PROLIF=True and RECEPTOR_PDB in CONFIG")
        work["_asp122_count"] = 0
        if require_asp122:
            work = work.iloc[0:0]

    if work.empty:
        return work

    work["_smiles"] = work[col_smiles]
    work["_pic50"] = work[col_pic50].apply(_to_float) if col_pic50 else float("nan")
    work["_sol"] = work[col_sol].apply(_to_float) if col_sol else float("nan")
    work["_dock"] = work[col_dock].apply(_to_float) if col_dock else float("nan")

    # Composite for ranking only — normalized raw values within filtered set
    pic50_n = _minmax(work["_pic50"])
    sol_n = _minmax(work["_sol"])
    tyr_n = _minmax(work["_tyr"].astype(float))
    dock_n = _minmax(work["_dock"], invert=True)  # more negative = better

    work["_composite"] = [
        _weighted_mean([s, p, t, d], [WEIGHT_SOL, WEIGHT_PIC50, WEIGHT_TYR, WEIGHT_DOCK])
        for s, p, t, d in zip(sol_n, pic50_n, tyr_n, dock_n)
    ]

    work = work.sort_values(
        by=["_composite", "_pic50", "_sol", "_dock"],
        ascending=[False, False, False, True],
        na_position="last",
    )

    if top_n > 0:
        work = work.head(top_n)

    return format_output(work)


def main() -> None:
    input_csv = Path(INPUT_CSV)
    output_csv = Path(OUTPUT_CSV)
    docking_runs = str(DOCKING_RUNS)

    if not input_csv.is_file():
        sys.exit(f"ERROR: input CSV not found: {input_csv}")

    print("=== select_top_molecules ===")
    print(f"Input:        {input_csv}")
    print(f"Output:       {output_csv}")
    print(f"Receptor:     {RECEPTOR_PDB}")
    print(f"Docking runs: {docking_runs}")
    print(f"Top N:        {TOP_N}")
    print(f"TYR target:   {TYR_TARGET}")
    print(f"ASP residue:  {ASP_RESIDUE}  {residue_ids('ASP', ASP_RESIDUE)}")
    print()

    gnina_config = None
    if RUN_PROLIF and AUTOBOX_LIGAND:
        gnina_config = GninaProlifConfig(
            receptor_path=RECEPTOR_PDB,
            autobox_ligand=AUTOBOX_LIGAND,
            gnina_executable=GNINA_EXECUTABLE,
            output_root=docking_runs,
            keep_outputs=True,
        )

    df = pd.read_csv(input_csv)
    print(f"Columns: {list(df.columns)}\n")

    result = select_top(
        df,
        tyr_target=TYR_TARGET,
        require_asp122=REQUIRE_ASP122,
        top_n=TOP_N,
        receptor_pdb=RECEPTOR_PDB,
        docking_runs=docking_runs if RUN_PROLIF else None,
        gnina_config=gnina_config,
        asp_resid=ASP_RESIDUE,
        compute_asp=RUN_PROLIF,
    )

    if result.empty:
        print("\nNo molecules passed filters.")
        sys.exit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"\nSaved {len(result)} molecules → {output_csv}")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
