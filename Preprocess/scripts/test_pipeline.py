#!/usr/bin/env python3
"""
test_pipeline.py — Manual test for GNINA docking + ProLIF on docking outputs.

Uses YOUR existing dock.py for docking, then runs ProLIF on mol0_out.sdf (pose 1).
Prints every file generated, docking score, TYR interactions, and tiered rewards.

Usage:
    python Preprocess/scripts/test_pipeline.py '<SMILES>'

Environment overrides (optional):
    DOCK_PY        path to dock.py
    RECEPTOR_PDB   receptor PDB for ProLIF
    OUTPUT_DIR     output folder (default: pipeline_output)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

# Import shared ProLIF + reward helpers (same logic used by REINVENT components)
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from reinvent_gnina_backend import (  # noqa: E402
    affinity_to_reward,
    analyze_tyr_interactions,
    tyr_count_to_reward,
)

# ============================================================
# CONFIG — edit paths or set env vars
# ============================================================
DOCK_PY = os.environ.get(
    "DOCK_PY",
    str(_SCRIPTS / "dock.py"),  # falls back to repo dock.py if you add one
)
# If your dock.py lives elsewhere, set:
# export DOCK_PY=/home/genai/Vishnu/psearch-master/reinvent-local-main/Preprocess/scripts/dock.py

RECEPTOR_PDB = os.environ.get(
    "RECEPTOR_PDB",
    "/home/genai/navneet/iict/pdl1/docking_TL_dataset/receptor.pdb",
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "pipeline_output")


def run(cmd: list) -> None:
    print("\n>> " + " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def clean_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES — RDKit could not parse input.")
    return Chem.MolToSmiles(mol, canonical=True)


def run_docking(smiles: str, output_dir: Path) -> tuple[str, str]:
    """Call your dock.py. Returns (log_file, out_sdf)."""
    input_csv = output_dir / "input.csv"
    pd.DataFrame({"SMILES": [smiles]}).to_csv(input_csv, index=False)

    run(["python", DOCK_PY, str(input_csv), str(output_dir) + "/"])

    log_file = output_dir / "mol0_log.txt"
    out_sdf = output_dir / "mol0_out.sdf"

    if not log_file.exists():
        raise FileNotFoundError(f"Missing: {log_file}")
    if not out_sdf.exists():
        raise FileNotFoundError(f"Missing: {out_sdf}")

    return str(log_file), str(out_sdf)


def extract_affinity(log_file: str) -> float:
    text = Path(log_file).read_text()
    m = re.search(r"^\s*1\s+(-?\d+(?:\.\d+)?)", text, re.MULTILINE)
    if m:
        return float(m.group(1))
    m = re.search(r"Docking Score:\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    raise ValueError("Could not parse affinity from GNINA log.")


def list_output_files(output_dir: Path) -> list[Path]:
    files = sorted(output_dir.rglob("*"))
    return [f for f in files if f.is_file()]


def print_file_inventory(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("OUTPUT FILES GENERATED")
    print("=" * 60)
    files = list_output_files(output_dir)
    if not files:
        print("(no files)")
        return
    for f in files:
        size_kb = f.stat().st_size / 1024
        rel = f.relative_to(output_dir)
        tag = ""
        if f.name == "mol0_out.sdf":
            tag = "  ← GNINA poses (pose 1 = best)"
        elif f.name == "mol0_log.txt":
            tag = "  ← GNINA log (affinity in row 1)"
        elif f.name.endswith(".mol") or f.name == "mol0.sdf":
            tag = "  ← intermediate (can delete)"
        print(f"  {rel}  ({size_kb:.1f} KB){tag}")


def print_docking_result(affinity: float, reward: float) -> None:
    print("\n" + "=" * 60)
    print("DOCKING RESULT")
    print("=" * 60)
    print(f"  Best affinity (mode 1):  {affinity:.2f} kcal/mol")
    print(f"  Docking reward:          {reward:.1f}")
    print("    <= -12  → 1.0 (high)")
    print("    -12 to -10 → 0.5 (medium)")
    print("    > -10  → 0.0 (low)")


def print_prolif_result(count: int, pi_count: int, details: list, reward: float) -> None:
    print("\n" + "=" * 60)
    print("PROLIF — TYR INTERACTIONS (best pose only)")
    print("=" * 60)
    print(f"  Total TYR interactions:  {count}")
    print(f"  Pi-stacking with TYR:    {pi_count}")
    print(f"  TYR reward:              {reward:.1f}")
    print("    >= 2  → 1.0 (high)")
    print("    1     → 0.5 (ok)")
    print("    0     → 0.0 (none)")

    if not details:
        print("\n  No TYR interactions detected.")
        return

    print()
    for i, ix in enumerate(details, 1):
        print(f"  {i}. {ix['protein']}  —  {ix['interaction']}")


def save_summary(
    output_dir: Path,
    smiles: str,
    affinity: float,
    dock_reward: float,
    tyr_count: int,
    pi_count: int,
    tyr_reward: float,
    details: list,
) -> None:
    path = output_dir / "test_summary.txt"
    with open(path, "w") as f:
        f.write(f"SMILES: {smiles}\n\n")
        f.write(f"Affinity: {affinity:.2f} kcal/mol\n")
        f.write(f"Docking reward: {dock_reward}\n\n")
        f.write(f"TYR interactions: {tyr_count}\n")
        f.write(f"TYR pi-stacking: {pi_count}\n")
        f.write(f"TYR reward: {tyr_reward}\n\n")
        for ix in details:
            f.write(f"  {ix['protein']} — {ix['interaction']}\n")
    print(f"\nSummary saved: {path}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python test_pipeline.py '<SMILES>'")
        sys.exit(1)

    raw = sys.argv[1]
    print("=" * 60)
    print("INPUT SMILES")
    print("=" * 60)
    print(raw)

    smiles = clean_smiles(raw)
    print(f"\nCanonical: {smiles}")

    output_dir = Path(OUTPUT_DIR)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # ── Step 1: GNINA docking via your dock.py ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 — GNINA DOCKING (dock.py)")
    print("=" * 60)
    log_file, out_sdf = run_docking(smiles, output_dir)
    affinity = extract_affinity(log_file)
    dock_reward = affinity_to_reward(affinity)

    print_file_inventory(output_dir)
    print_docking_result(affinity, dock_reward)

    # ── Step 2: ProLIF on best pose from mol0_out.sdf ───────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — PROLIF (receptor.pdb + pose 1 from mol0_out.sdf)")
    print("=" * 60)
    try:
        tyr_count, pi_count, details = analyze_tyr_interactions(RECEPTOR_PDB, out_sdf)
        tyr_reward = tyr_count_to_reward(tyr_count)
        print_prolif_result(tyr_count, pi_count, details, tyr_reward)
    except Exception as exc:
        print(f"\n  [!] ProLIF failed: {exc}")
        tyr_count, pi_count, details, tyr_reward = 0, 0, [], 0.0

    save_summary(output_dir, smiles, affinity, dock_reward,
                 tyr_count, pi_count, tyr_reward, details)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"  Output dir:  {output_dir.resolve()}")
    print(f"  Key files:   mol0_out.sdf, mol0_log.txt, test_summary.txt")


if __name__ == "__main__":
    main()
