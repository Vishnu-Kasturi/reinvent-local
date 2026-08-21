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
from rdkit.Chem import inchi

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
DOCKING_DIR = REPO_ROOT / "run5" / "docking"          # existing GNINA poses
NEW_DOCKING_DIR = REPO_ROOT / "run5" / "new_docking"  # missing poses docked here

TOP_N = 50
# TYR filter — integer pi-pi stack count at TYR56 (2 ligand rings stacking → 2).
# Tyrosine_PiStacking in LibInvent CSV = TyrInteractionCount_raw from RL.
# Reward tier is separate: 0→0.0, 1 stack→0.5, >=2 stacks→1.0
APPLY_TYR_FILTER = True
TYR_MIN = 2          # keep molecules with >= 2 pi-pi ring stacks at TYR56
TYR_MAX = None
ASP_RESIDUE = 122
REQUIRE_ASP122 = True   # keep molecules with >=1 ProLIF contact at ASP122
RUN_PROLIF = True

# If pose not in DOCKING_DIR → GNINA dock into NEW_DOCKING_DIR (never overwrites original)
REDock_IF_MISSING = True
VERBOSE = True          # print per-molecule pose lookup / docking status

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
    _format_asp_breakdown,
    _smiles_hash,
    analyze_asp_interactions,
    canonicalize,
    run_molecule_pipeline,
)
from prolif_compat import residue_ids  # noqa: E402

COLUMN_ALIASES = {
    "smiles": ["SMILES", "smiles"],
    # RL TYR56 pi-pi stack count (integer: N rings stacking → N)
    "tyr_count": [
        "Tyrosine_PiStacking",
        "tyrosine_pistacking",
        "TyrInteractionCount_raw (raw)",
        "TyrInteractionCount_raw",
    ],
    "pic50": ["pIC50", "PD1PDL1pIC50 (raw)", "PD1PDL1pIC50_raw", "PD1PDL1pIC50"],
    "sol": ["Solubility", "PD1PDL1Sol (raw)", "PD1PDL1Sol_raw", "PD1PDL1Sol"],
    "docking": ["Docking_Score", "DockingAffinity_raw (raw)", "DockingAffinity_raw"],
    "asp122": ["ASP122_Interaction", "ASP122_interaction", "Asp122Interaction", "ASP122 (raw)", "ASP122"],
}


def _tyr_filter_label(tyr_min: int = TYR_MIN, tyr_max: Optional[int] = TYR_MAX) -> str:
    """Human-readable TYR filter description (uses module CONFIG defaults)."""
    if tyr_max is not None and tyr_max == tyr_min:
        return f"== {tyr_min}"
    if tyr_max is not None:
        return f"{tyr_min}–{tyr_max}"
    return f">= {tyr_min}"


def _print_series_stats(name: str, series: pd.Series) -> None:
    vals = series.dropna()
    if vals.empty:
        print(f"  {name}: (all NaN)")
        return
    vc = vals.value_counts().sort_index()
    summary = ", ".join(f"{k}:{v}" for k, v in vc.head(8).items())
    if len(vc) > 8:
        summary += f", ... ({len(vc)} unique)"
    print(f"  {name}: min={vals.min()}, max={vals.max()}, counts=[{summary}]")


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


def _inchikey_from_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def _mol_keys(mol) -> list[str]:
    """Lookup keys for a pose molecule (canonical SMILES + InChIKey)."""
    keys: list[str] = []
    if mol is None:
        return keys
    try:
        m = Chem.RemoveHs(mol)
        can = Chem.MolToSmiles(m, canonical=True)
        if can:
            keys.append(can)
        ik = inchi.MolToInchiKey(m)
        if ik:
            keys.append(f"inchikey:{ik}")
    except Exception:
        pass
    return keys


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
            keys = _mol_keys(mol)
            return keys[0] if keys else None
    except Exception:
        return None
    return None


def _pose_label(sdf_path: str) -> str:
    parent = os.path.basename(os.path.dirname(sdf_path))
    return f"{parent}/{os.path.basename(sdf_path)}"


def is_valid_sdf(sdf_path: str) -> bool:
    """True if SDF exists and contains at least one readable molecule."""
    if not os.path.isfile(sdf_path):
        return False
    if os.path.getsize(sdf_path) < 20:
        return False
    try:
        for mol in Chem.SDMolSupplier(sdf_path, removeHs=False):
            if mol is not None:
                return True
        return False
    except OSError:
        return False


class DockPoseCache:
    """
    Index GNINA outputs under one or more folders (searched in order).

    Finds poses like:
      docking/batch_*/mol_0000_<hash>/mol0_out.sdf
      new_docking/.../mol0_out.sdf, mol1_out.sdf, ...
    Prefers mol0_out.sdf (best pose) over mol1, mol2, ...
    """

    def __init__(self, docking_dirs: list[str]):
        self.docking_dirs = [os.path.abspath(d) for d in docking_dirs if d]
        self._by_hash: dict[str, str] = {}
        self._by_smiles: dict[str, str] = {}
        self._by_inchikey: dict[str, str] = {}
        self._sdf_source: dict[str, str] = {}  # sdf_path -> folder label
        self._built = False

    def _folder_label(self, sdf_path: str, docking_dir: str) -> str:
        base = os.path.basename(os.path.abspath(docking_dir))
        if "new_docking" in base:
            return "new_docking"
        return "docking"

    def _register_sdf(self, sdf_path: str, docking_dir: str) -> None:
        if not is_valid_sdf(sdf_path):
            return
        label = self._folder_label(sdf_path, docking_dir)
        self._sdf_source[sdf_path] = label
        try:
            for mol in Chem.SDMolSupplier(sdf_path, removeHs=False):
                if mol is None:
                    continue
                for key in _mol_keys(mol):
                    if key.startswith("inchikey:"):
                        if key not in self._by_inchikey:
                            self._by_inchikey[key] = sdf_path
                    elif key not in self._by_smiles:
                        self._by_smiles[key] = sdf_path
                break
        except Exception:
            can = _canonical_smiles_from_sdf(sdf_path)
            if can and can not in self._by_smiles:
                self._by_smiles[can] = sdf_path

    def build(self) -> None:
        if self._built:
            return

        for docking_dir in self.docking_dirs:
            if not os.path.isdir(docking_dir):
                continue

            candidates: list[tuple[int, str, str]] = []

            for pattern in ("**/mol*_out.sdf", "**/*_out.sdf"):
                for sdf_path in glob.glob(os.path.join(docking_dir, pattern), recursive=True):
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

            candidates.sort(key=lambda x: (x[2], x[0], x[1]))

            for _pose_rank, sdf_path, h in candidates:
                if h and h not in self._by_hash:
                    self._by_hash[h] = sdf_path
                self._register_sdf(sdf_path, docking_dir)

        self._built = True
        dirs_label = ", ".join(self.docking_dirs) or "(none)"
        n_docking = sum(1 for v in self._sdf_source.values() if v == "docking")
        n_new = sum(1 for v in self._sdf_source.values() if v == "new_docking")
        print(
            f"Dock pose index: {len(self._by_hash)} by hash, "
            f"{len(self._by_smiles)} by SMILES, {len(self._by_inchikey)} by InChIKey"
        )
        print(f"  SDF files: {n_docking} in docking/, {n_new} in new_docking/")
        print(f"  Searched: {dirs_label}")

    def register_pose(self, smiles: str, sdf_path: str, docking_dir: str = "") -> None:
        can = canonicalize(smiles)
        if not can or not os.path.isfile(sdf_path):
            return
        self._by_smiles[can] = sdf_path
        self._by_hash[_smiles_hash(can)] = sdf_path
        ik = _inchikey_from_smiles(can)
        if ik:
            self._by_inchikey[f"inchikey:{ik}"] = sdf_path
        label = "new_docking" if docking_dir and "new_docking" in os.path.abspath(docking_dir) else "docking"
        self._sdf_source[sdf_path] = label

    def find_with_source(self, smiles: str) -> tuple[Optional[str], str]:
        """Return (sdf_path, source_label). source_label: docking|new_docking|not_found."""
        self.build()
        can = canonicalize(smiles)
        if not can:
            return None, "not_found"

        sdf = None
        h = _smiles_hash(can)
        if h in self._by_hash:
            sdf = self._by_hash[h]
        elif can in self._by_smiles:
            sdf = self._by_smiles[can]
        else:
            ik = _inchikey_from_smiles(can)
            if ik and f"inchikey:{ik}" in self._by_inchikey:
                sdf = self._by_inchikey[f"inchikey:{ik}"]

        if sdf and os.path.isfile(sdf):
            if not is_valid_sdf(sdf):
                sdf = None
            else:
                return sdf, self._sdf_source.get(sdf, "docking")
        return None, "not_found"

    def invalidate(self, smiles: str, sdf_path: str) -> None:
        """Remove a bad pose mapping so it can be re-docked."""
        can = canonicalize(smiles)
        if can:
            self._by_smiles.pop(can, None)
            self._by_hash.pop(_smiles_hash(can), None)
            ik = _inchikey_from_smiles(can)
            if ik:
                self._by_inchikey.pop(f"inchikey:{ik}", None)
        self._sdf_source.pop(sdf_path, None)

    def find(self, smiles: str) -> Optional[str]:
        sdf, _ = self.find_with_source(smiles)
        return sdf


_POSE_CACHE: Optional[DockPoseCache] = None


def get_pose_cache(docking_dirs: list[str]) -> DockPoseCache:
    global _POSE_CACHE
    key = tuple(docking_dirs)
    if _POSE_CACHE is None or tuple(_POSE_CACHE.docking_dirs) != key:
        _POSE_CACHE = DockPoseCache(list(docking_dirs))
    return _POSE_CACHE


def _run_asp122(
    receptor_pdb: str,
    sdf: str,
    smiles: str,
    asp_resid: int,
    verbose: bool,
) -> int:
    """Return total ProLIF contact count at ASP122 (all interaction types)."""
    contacts, polar, _, breakdown = analyze_asp_interactions(
        receptor_pdb, sdf, smiles, asp_residue=asp_resid
    )
    if verbose:
        detail = _format_asp_breakdown(breakdown, polar)
        print(f"       ASP{asp_resid} contacts: {contacts} ({detail})")
    return contacts


def compute_asp122_for_smiles(
    smiles: str,
    receptor_pdb: str,
    pose_cache: DockPoseCache,
    asp_resid: int = 122,
    gnina_config: Optional[GninaProlifConfig] = None,
    allow_redock: bool = False,
    mol_index: int = 0,
    verbose: bool = VERBOSE,
) -> tuple[int, str, bool]:
    """Return (contact_count, sdf_path, was_newly_docked)."""
    smi_short = smiles if len(smiles) <= 50 else smiles[:47] + "..."
    sdf, source = pose_cache.find_with_source(smiles)

    if sdf and os.path.isfile(sdf):
        if not is_valid_sdf(sdf):
            if verbose:
                print(f"  [{mol_index + 1}] INVALID SDF ({source}): {_pose_label(sdf)}")
            pose_cache.invalidate(smiles, sdf)
            sdf = None
        else:
            if verbose:
                print(f"  [{mol_index + 1}] POSE FOUND ({source}): {_pose_label(sdf)}")
            try:
                contacts = _run_asp122(receptor_pdb, sdf, smiles, asp_resid, verbose)
                return contacts, sdf, False
            except Exception as exc:
                if verbose:
                    print(f"       ProLIF FAILED: {exc}")
                pose_cache.invalidate(smiles, sdf)
                sdf = None

    if sdf:
        return 0, "", False

    if not allow_redock or gnina_config is None:
        if verbose:
            print(f"  [{mol_index + 1}] NO POSE — skipping (re-dock disabled): {smi_short}")
        return 0, "", False

    if verbose:
        print(f"  [{mol_index + 1}] NO POSE — GNINA docking → {gnina_config.output_root}")
        print(f"       SMILES: {smi_short}")

    res = run_molecule_pipeline(smiles, mol_index, 1, gnina_config)
    if not res.docking_ok or not is_valid_sdf(res.out_sdf):
        if verbose:
            print(f"       GNINA FAILED: {res.error or 'invalid/empty output SDF'}")
        return 0, "", False

    pose_cache.register_pose(smiles, res.out_sdf, gnina_config.output_root)
    if verbose:
        print(f"       GNINA OK: {_pose_label(res.out_sdf)}")
    try:
        contacts = _run_asp122(receptor_pdb, res.out_sdf, smiles, asp_resid, verbose)
    except Exception as exc:
        if verbose:
            print(f"       ProLIF FAILED after GNINA: {exc}")
        pose_cache.invalidate(smiles, res.out_sdf)
        return 0, "", False
    return contacts, res.out_sdf, True


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
    pose_cache: Optional[DockPoseCache] = None,
    gnina_config: Optional[GninaProlifConfig] = None,
    asp_resid: int = ASP_RESIDUE,
    compute_asp: bool = RUN_PROLIF,
    allow_redock: bool = False,
    apply_tyr_filter: bool = APPLY_TYR_FILTER,
) -> pd.DataFrame:
    col_smiles = _find_column(df, COLUMN_ALIASES["smiles"])
    col_tyr = _find_column(df, COLUMN_ALIASES["tyr_count"])
    col_pic50 = _find_column(df, COLUMN_ALIASES["pic50"])
    col_sol = _find_column(df, COLUMN_ALIASES["sol"])
    col_dock = _find_column(df, COLUMN_ALIASES["docking"])
    col_asp = _find_column(df, COLUMN_ALIASES["asp122"])

    if not col_smiles:
        raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")

    print(
        f"Detected columns: SMILES={col_smiles!r}, TYR={col_tyr!r}, "
        f"pIC50={col_pic50!r}, Sol={col_sol!r}, Dock={col_dock!r}"
    )
    print(f"Filters: APPLY_TYR_FILTER={apply_tyr_filter}, REQUIRE_ASP122={require_asp122}")

    work = df.copy()
    print(f"Input rows: {len(work)}")

    if col_tyr:
        work["_tyr"] = work[col_tyr].apply(_to_float).round().astype(int)
        _print_series_stats(f"TYR56 pi-pi ({col_tyr})", work["_tyr"])
    else:
        work["_tyr"] = 0
        print("WARNING: no TYR column found — defaulting counts to 0")

    if apply_tyr_filter and col_tyr:
        before = len(work)
        work = work[work["_tyr"] >= tyr_min]
        if tyr_max is not None:
            work = work[work["_tyr"] <= tyr_max]
        print(f"After TYR56 pi-pi {_tyr_filter_label(tyr_min, tyr_max)}: {len(work)} (removed {before - len(work)})")
        if work.empty and before > 0:
            max_tyr = int(df[col_tyr].apply(_to_float).round().max())
            print(
                f"  Hint: max TYR pi-pi count in input was {max_tyr}; "
                f"lower TYR_MIN if you want to keep molecules with fewer stacks"
            )
    elif apply_tyr_filter and not col_tyr:
        print("WARNING: APPLY_TYR_FILTER=True but no TYR column — skipping TYR filter")
    else:
        print("TYR filter skipped (APPLY_TYR_FILTER=False)")

    if work.empty:
        return work

    if compute_asp and receptor_pdb and pose_cache is not None:
        pose_cache.build()
        print(f"Computing ASP{asp_resid} via ProLIF (existing poses first)...")
        unique_smiles = work[col_smiles].dropna().unique()
        asp_contact_cache: dict[str, int] = {}
        sdf_cache: dict[str, str] = {}
        missing: list[str] = []
        redocked = 0
        found_existing = 0
        for i, smi in enumerate(unique_smiles, 1):
            contacts, sdf_cache[smi], newly_docked = compute_asp122_for_smiles(
                smi,
                receptor_pdb,
                pose_cache,
                asp_resid=asp_resid,
                gnina_config=gnina_config,
                allow_redock=allow_redock,
                mol_index=i - 1,
                verbose=VERBOSE,
            )
            asp_contact_cache[smi] = contacts
            if not sdf_cache[smi]:
                missing.append(smi)
            elif newly_docked:
                redocked += 1
            else:
                found_existing += 1
            if not VERBOSE and (i % 10 == 0 or i == len(unique_smiles)):
                n_pos = sum(1 for v in asp_contact_cache.values() if v > 0)
                print(f"  ProLIF ASP{asp_resid}: {i}/{len(unique_smiles)} ({n_pos} with ASP contact)")
        print(
            f"  Summary: {found_existing} poses from docking/, "
            f"{redocked} newly docked → new_docking/, {len(missing)} failed"
        )
        _print_series_stats("ASP122 contact counts", pd.Series(list(asp_contact_cache.values())))
        if missing:
            print(f"  WARNING: {len(missing)} SMILES still have no pose after docking")
        work["_asp122_count"] = work[col_smiles].map(asp_contact_cache).fillna(0).astype(int)
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
    new_docking_dir = str(_P(NEW_DOCKING_DIR))
    allow_redock = REDock_IF_MISSING

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

    if allow_redock:
        os.makedirs(new_docking_dir, exist_ok=True)

    # Search existing docking first, then any previously new-docked poses
    pose_cache = get_pose_cache([docking_dir, new_docking_dir])

    print("=== select_top_molecules ===")
    print(f"Input:        {input_csv}")
    print(f"Output:       {output_csv}")
    print(f"Receptor:     {RECEPTOR_PDB}")
    print(f"Docking dir:  {docking_dir}")
    print(f"New docking:  {new_docking_dir}  (missing poses → here)")
    print(f"Re-dock miss: {allow_redock}")
    print(f"Top N:        {TOP_N}")
    print(f"TYR filter:   {_tyr_filter_label(TYR_MIN, TYR_MAX)}")
    print(f"ASP filter:   ProLIF contact at ASP{ASP_RESIDUE}  {residue_ids('ASP', ASP_RESIDUE)}")
    print()

    gnina_config = None
    if allow_redock and AUTOBOX_LIGAND:
        gnina_config = GninaProlifConfig(
            receptor_path=RECEPTOR_PDB,
            autobox_ligand=AUTOBOX_LIGAND,
            gnina_executable=GNINA_EXECUTABLE,
            output_root=new_docking_dir,
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
        pose_cache=pose_cache if RUN_PROLIF else None,
        gnina_config=gnina_config,
        asp_resid=ASP_RESIDUE,
        compute_asp=RUN_PROLIF,
        allow_redock=allow_redock,
        apply_tyr_filter=APPLY_TYR_FILTER,
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
