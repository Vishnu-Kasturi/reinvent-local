#!/usr/bin/env python3
"""
select_top_molecules.py — Post-RL molecule selection (not a REINVENT reward).

Filters RL results CSV, computes ASP122 via ProLIF, ranks top hits.

Run (no CLI args needed — edit CONFIG below):
    conda activate reinvent_qsar
    cd ~/Vishnu/psearch-master/reinvent-local-main
    python Preprocess/scripts/select_top_molecules.py

Note: REPO_ROOT = reinvent-local-main/iict_libinvent (auto-resolved from script path).

Output columns:
    rank, SMILES, pIC50, Solubility, Docking_Score, Tyrosine_PiStacking,
    ASP122_Interaction, composite
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem

# ---------------------------------------------------------------------------
# CONFIG — edit these paths/settings only
# Paths resolve from this script's location (safe to run from any cwd).
# Layout: reinvent-local-main/iict_libinvent/run5/...
# ---------------------------------------------------------------------------

def _P(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


# reinvent-local-main/  (auto-detected: parent of Preprocess/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = _PROJECT_ROOT / "iict_libinvent"

INPUT_CSV = REPO_ROOT / "run5" / "top100_balanced.csv"
OUTPUT_CSV = REPO_ROOT / "run5" / "top_hits.csv"
DOCKING_DIR = REPO_ROOT / "run5" / "docking"   # existing GNINA poses (mol0_out.sdf, etc.)

TOP_N = 50
TYR_MIN = 2             # minimum TYR56 pi-pi stacks (keep if count >= TYR_MIN)
TYR_MAX = None          # optional max (e.g. 2 for exact match only); None = no limit
ASP_RESIDUE = 122
REQUIRE_ASP122 = True   # keep only molecules with ASP122 interaction
RUN_PROLIF = True       # compute ASP122 from existing docked poses

# Docking behaviour — use poses already in DOCKING_DIR; do NOT re-run GNINA
USE_EXISTING_DOCKING_ONLY = True   # True = never dock again, only read mol*_out.sdf
REDock_IF_MISSING = False          # True = GNINA only when pose not found (slow)

RECEPTOR_PDB = "/home/genai/navneet/iict/pdl1/docking_TL_dataset/receptor.pdb"
AUTOBOX_LIGAND = "/home/genai/navneet/iict/pdl1/docking_TL_dataset/ref_ligand.pdb"
GNINA_EXECUTABLE = "/home/genai/Documents/gnina/gnina"

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
)
if REDock_IF_MISSING:
    from reinvent_gnina_backend import run_molecule_pipeline  # noqa: E402
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


def _canonical_smiles_from_sdf(sdf_path: str) -> Optional[str]:
    """Best-effort canonical SMILES from first valid pose in an SDF."""
    try:
        for mol in Chem.SDMolSupplier(sdf_path, removeHs=False):
            if mol is None:
                continue
            for prop in ("SMILES", "_SMILES", "smiles"):
                if mol.HasProp(prop):
                    can = canonicalize(mol.GetProp(prop))
                    if can:
                        return can
            return canonicalize(Chem.MolToSmiles(mol))
    except Exception:
        return None
    return None


class DockPoseCache:
    """
    Index existing GNINA outputs under DOCKING_DIR.

    Finds poses like:
      docking/batch_*/mol_0000_<hash>/mol0_out.sdf
      docking/<any_subdir>/mol0_out.sdf, mol1_out.sdf, ...
    Prefers mol0_out.sdf (best pose) over mol1, mol2, ...
    """

    def __init__(self, docking_dir: str):
        self.docking_dir = docking_dir
        self._by_hash: dict[str, str] = {}
        self._by_smiles: dict[str, str] = {}
        self._built = False

    def build(self) -> None:
        if self._built or not os.path.isdir(self.docking_dir):
            self._built = True
            return

        candidates: list[tuple[int, str, str]] = []  # (pose_rank, path, hash_or_empty)

        for pattern in ("**/mol*_out.sdf", "**/*_out.sdf"):
            for sdf_path in glob.glob(os.path.join(self.docking_dir, pattern), recursive=True):
                if not os.path.isfile(sdf_path):
                    continue
                base = os.path.basename(sdf_path)
                pose_rank = 0
                m_pose = re.match(r"mol(\d+)_out\.sdf$", base, re.I)
                if m_pose:
                    pose_rank = int(m_pose.group(1))

                parent = os.path.basename(os.path.dirname(sdf_path))
                h = ""
                m_hash = re.search(r"([a-f0-9]{8})$", parent)
                if m_hash:
                    h = m_hash.group(1)

                candidates.append((pose_rank, sdf_path, h))

        # Lower pose_rank = better (mol0 before mol1)
        candidates.sort(key=lambda x: (x[2], x[0], x[1]))

        for pose_rank, sdf_path, h in candidates:
            if h and h not in self._by_hash:
                self._by_hash[h] = sdf_path
            can = _canonical_smiles_from_sdf(sdf_path)
            if can and can not in self._by_smiles:
                self._by_smiles[can] = sdf_path

        self._built = True
        print(
            f"Dock pose index: {len(self._by_hash)} by hash, "
            f"{len(self._by_smiles)} by SMILES under {self.docking_dir}"
        )

    def find(self, smiles: str) -> Optional[str]:
        self.build()
        can = canonicalize(smiles)
        if not can:
            return None
        h = _smiles_hash(can)
        if h in self._by_hash:
            return self._by_hash[h]
        if can in self._by_smiles:
            return self._by_smiles[can]
        return None


_POSE_CACHE: Optional[DockPoseCache] = None


def get_pose_cache(docking_dir: str) -> DockPoseCache:
    global _POSE_CACHE
    if _POSE_CACHE is None or _POSE_CACHE.docking_dir != docking_dir:
        _POSE_CACHE = DockPoseCache(docking_dir)
    return _POSE_CACHE


def find_docked_sdf(smiles: str, docking_dir: str) -> Optional[str]:
    return get_pose_cache(docking_dir).find(smiles)


def compute_asp122_for_smiles(
    smiles: str,
    receptor_pdb: str,
    docking_dir: str,
    asp_resid: int = 122,
    config: Optional[GninaProlifConfig] = None,
    allow_redock: bool = False,
) -> tuple[int, str]:
    sdf = find_docked_sdf(smiles, docking_dir)
    if sdf and os.path.isfile(sdf):
        count, _ = analyze_asp_interactions(receptor_pdb, sdf, smiles, asp_residue=asp_resid)
        return count, sdf

    if not allow_redock or config is None:
        return 0, ""

    from reinvent_gnina_backend import run_molecule_pipeline

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
    tyr_min: int = TYR_MIN,
    tyr_max: Optional[int] = TYR_MAX,
    require_asp122: bool = REQUIRE_ASP122,
    top_n: int = TOP_N,
    receptor_pdb: Optional[str] = None,
    docking_dir: Optional[str] = None,
    gnina_config: Optional[GninaProlifConfig] = None,
    asp_resid: int = ASP_RESIDUE,
    compute_asp: bool = RUN_PROLIF,
    allow_redock: bool = False,
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
        work = work[work["_tyr"] >= tyr_min]
        if tyr_max is not None:
            work = work[work["_tyr"] <= tyr_max]
        if tyr_max is not None and tyr_max == tyr_min:
            label = f"== {tyr_min}"
        elif tyr_max is not None:
            label = f"{tyr_min}–{tyr_max}"
        else:
            label = f">= {tyr_min}"
        print(f"After TYR56 pi-pi {label}: {len(work)} (removed {before - len(work)})")
    else:
        print("WARNING: No TyrInteractionCount column — skipping TYR filter")
        work["_tyr"] = 0

    if work.empty:
        return work

    if compute_asp and receptor_pdb and docking_dir:
        get_pose_cache(docking_dir).build()
        print(f"Computing ASP{asp_resid} via ProLIF on existing poses (no GNINA)...")
        unique_smiles = work[col_smiles].dropna().unique()
        asp_cache: dict[str, int] = {}
        sdf_cache: dict[str, str] = {}
        missing: list[str] = []
        for i, smi in enumerate(unique_smiles, 1):
            asp_cache[smi], sdf_cache[smi] = compute_asp122_for_smiles(
                smi,
                receptor_pdb,
                docking_dir,
                asp_resid=asp_resid,
                config=gnina_config,
                allow_redock=allow_redock,
            )
            if not sdf_cache[smi]:
                missing.append(smi)
            if i % 10 == 0 or i == len(unique_smiles):
                n_pos = sum(1 for v in asp_cache.values() if v > 0)
                print(f"  ProLIF ASP{asp_resid}: {i}/{len(unique_smiles)} ({n_pos} with interactions)")
        if missing:
            print(f"  WARNING: {len(missing)} SMILES had no pose in {docking_dir}")
            if not allow_redock:
                print("  (set REDock_IF_MISSING=True to GNINA missing poses)")
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
    input_csv = _P(INPUT_CSV)
    output_csv = _P(OUTPUT_CSV)
    docking_dir = str(_P(DOCKING_DIR))
    allow_redock = REDock_IF_MISSING and not USE_EXISTING_DOCKING_ONLY

    if not input_csv.is_file():
        sys.exit(
            f"ERROR: input CSV not found:\n  {input_csv}\n\n"
            f"Expected layout:\n"
            f"  {REPO_ROOT}/run5/top100_balanced.csv\n\n"
            f"REPO_ROOT resolves to: {REPO_ROOT}\n"
            f"Place your CSV there, or edit REPO_ROOT / INPUT_CSV in CONFIG."
        )

    if not os.path.isdir(docking_dir):
        sys.exit(f"ERROR: docking folder not found:\n  {docking_dir}")

    print("=== select_top_molecules ===")
    print(f"Input:        {input_csv}")
    print(f"Output:       {output_csv}")
    print(f"Receptor:     {RECEPTOR_PDB}")
    print(f"Docking dir:  {docking_dir}")
    print(f"Use existing: {USE_EXISTING_DOCKING_ONLY}  (re-dock missing: {allow_redock})")
    print(f"Top N:        {TOP_N}")
    if TYR_MAX is not None and TYR_MAX == TYR_MIN:
        tyr_label = f"== {TYR_MIN}"
    elif TYR_MAX is not None:
        tyr_label = f"{TYR_MIN}–{TYR_MAX}"
    else:
        tyr_label = f">= {TYR_MIN}"
    print(f"TYR filter:   {tyr_label}")
    print(f"ASP residue:  {ASP_RESIDUE}  {residue_ids('ASP', ASP_RESIDUE)}")
    print()

    gnina_config = None
    if allow_redock and AUTOBOX_LIGAND:
        gnina_config = GninaProlifConfig(
            receptor_path=RECEPTOR_PDB,
            autobox_ligand=AUTOBOX_LIGAND,
            gnina_executable=GNINA_EXECUTABLE,
            output_root=docking_dir,
            keep_outputs=True,
        )

    df = pd.read_csv(input_csv)
    print(f"Columns: {list(df.columns)}\n")

    result = select_top(
        df,
        tyr_min=TYR_MIN,
        tyr_max=TYR_MAX,
        require_asp122=REQUIRE_ASP122,
        top_n=TOP_N,
        receptor_pdb=RECEPTOR_PDB,
        docking_dir=docking_dir if RUN_PROLIF else None,
        gnina_config=gnina_config,
        asp_resid=ASP_RESIDUE,
        compute_asp=RUN_PROLIF,
        allow_redock=allow_redock,
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
