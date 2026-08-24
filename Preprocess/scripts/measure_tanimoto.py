#!/usr/bin/env python3
"""
measure_tanimoto.py
===================
Measure max (and mean) Morgan Tanimoto vs the JAK2 reference set (pIC50 6–11).

Only the SMILES column is read from the input — extra REINVENT columns are ignored.
Output CSV has no Score / reward columns.

Usage:
  python Preprocess/scripts/measure_tanimoto.py \\
      --input_csv results/jak2_rl_toml_1.csv \\
      --output_csv results/jak2_rl_tanimoto.csv

  python Preprocess/scripts/measure_tanimoto.py \\
      --input_csv results/jak2_mol2mol_candidates.csv \\
      --reference data/jak2_TL_train.smi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tanimoto_utils import (  # noqa: E402
    build_morgan_fps,
    load_reference_fps,
    max_tanimoto_per_molecule,
    mean_tanimoto_per_molecule,
    read_smiles_from_csv,
)

DROP_DISPLAY_COLS = {
    "score",
    "agent",
    "prior",
    "target",
    "smiles_state",
    "scaffold",
    "step",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure Tanimoto vs JAK2 reference (SMILES-only input).")
    p.add_argument("--input_csv", required=True, help="Generated molecules CSV (any width; SMILES column required)")
    p.add_argument("--output_csv", required=True, help="Output CSV path")
    p.add_argument("--target", default="jak2", choices=["jak2"], help="Preset reference dataset")
    p.add_argument("--reference", default=None, help="Override reference .smi or .csv")
    p.add_argument("--min_pic50", type=float, default=6.0, help="JAK2 reference lower pIC50 bound")
    p.add_argument("--max_pic50", type=float, default=11.0, help="JAK2 reference upper pIC50 bound")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--top_n", type=int, default=10, help="Rows to print in summary table")
    p.add_argument("--dedupe", action="store_true", help="Keep unique SMILES only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv).expanduser().resolve()
    output_path = Path(args.output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Reading SMILES from: {input_path}")
    query_smiles = read_smiles_from_csv(input_path)
    if args.dedupe:
        query_smiles = list(dict.fromkeys(query_smiles))
    print(f"    Query molecules: {len(query_smiles)}")

    ref_fps, ref_valid, ref_label = load_reference_fps(
        target=args.target,
        reference_path=args.reference,
        min_pic50=args.min_pic50,
        max_pic50=args.max_pic50,
        radius=args.radius,
    )
    print(f"[*] Reference: {ref_label}")
    print(f"    Valid reference fingerprints: {len(ref_fps):,}")

    query_fps, valid_smiles = build_morgan_fps(query_smiles, radius=args.radius)
    invalid = len(query_smiles) - len(valid_smiles)
    if invalid:
        print(f"    Skipped {invalid} invalid query SMILES")

    max_tan = max_tanimoto_per_molecule(query_fps, ref_fps)
    mean_tan = mean_tanimoto_per_molecule(query_fps, ref_fps)

    df_out = pd.DataFrame({
        "SMILES": valid_smiles,
        "max_tanimoto": max_tan,
        "mean_tanimoto": mean_tan,
    })
    df_out = df_out.sort_values("max_tanimoto", ascending=False).reset_index(drop=True)
    df_out.insert(0, "rank", range(1, len(df_out) + 1))
    df_out.to_csv(output_path, index=False)
    print(f"[+] Saved → {output_path}")

    print("\n" + "=" * 60)
    print(" TANIMOTO SUMMARY (vs JAK2 reference pIC50 6–11)")
    print("=" * 60)
    print(f"  Mean max Tanimoto : {np.mean(max_tan):.3f}")
    print(f"  Median max Tanimoto: {np.median(max_tan):.3f}")
    print(f"  % >= 0.40         : {100 * np.mean(np.array(max_tan) >= 0.4):.1f}%")
    print(f"  Exact copies      : {sum(1 for x in max_tan if x >= 0.999):,}")

    display_cols = [c for c in df_out.columns if c.lower() not in DROP_DISPLAY_COLS]
    print(f"\nTop {min(args.top_n, len(df_out))} (no reward Score column):")
    print(df_out.head(args.top_n)[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
