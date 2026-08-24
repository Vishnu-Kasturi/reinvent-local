#!/usr/bin/env python3
"""
validate_tanimoto_pipeline.py
=============================
Validates RL / sampling output against the JAK2 reference (pIC50 6–11).

Reads only the SMILES column from the RL CSV (ignores reward / Score columns).
Writes an enriched results CSV and a JSON summary.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tanimoto_utils import (  # noqa: E402
    build_morgan_fps,
    load_reference_fps,
    max_tanimoto_per_molecule,
    read_smiles_from_csv,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _latest_rl_csv(results_dir: Path, target: str) -> Path | None:
    patterns = [
        f"{target}_rl_toml_*.csv",
        f"{target}_rl_*.csv",
        f"{target}_mol2mol*.csv",
    ]
    files: list[Path] = []
    for pat in patterns:
        files.extend(results_dir.glob(pat))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["jak2", "pd1_pdl1"])
    parser.add_argument("--raw_csv", default=None, help="Reference dataset (default: JAK2 preprocess 6–11)")
    parser.add_argument("--input_csv", default=None, help="Generated molecules CSV (default: latest RL output)")
    parser.add_argument("--output_csv", default=None, help="Enriched output CSV path")
    parser.add_argument("--min_pic50", type=float, default=6.0)
    parser.add_argument("--max_pic50", type=float, default=11.0)
    args = parser.parse_args()

    results_dir = _REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Reference fingerprints
    if args.target == "jak2":
        ref_fps, _, ref_label = load_reference_fps(
            target="jak2",
            reference_path=args.raw_csv,
            min_pic50=args.min_pic50,
            max_pic50=args.max_pic50,
        )
    else:
        if not args.raw_csv:
            args.raw_csv = str(_REPO_ROOT / "Preprocess" / "Data_pd1_pdl1" / "pd1_pdl1_pic50_raw.csv")
        ref_fps, _, ref_label = load_reference_fps(reference_path=args.raw_csv)

    print(f"[*] Reference: {ref_label} ({len(ref_fps):,} fingerprints)")

    # Input CSV
    if args.input_csv:
        rl_path = Path(args.input_csv).expanduser().resolve()
    else:
        found = _latest_rl_csv(results_dir, args.target)
        if found is None:
            print(f"[!] No RL output found for {args.target} in {results_dir}")
            return
        rl_path = found

    if not rl_path.exists():
        print(f"[!] Input CSV not found: {rl_path}")
        return

    print(f"[*] Validating: {rl_path}")
    query_smiles = read_smiles_from_csv(rl_path)
    query_fps, valid_smiles = build_morgan_fps(query_smiles)
    if not query_fps:
        print("[!] No valid query SMILES")
        return

    max_tanimotos = max_tanimoto_per_molecule(query_fps, ref_fps)

    mean_sim = float(np.mean(max_tanimotos))
    med_sim = float(np.median(max_tanimotos))
    gt_40 = float(np.mean(np.array(max_tanimotos) >= 0.4))
    exact = int(sum(1 for x in max_tanimotos if x >= 0.999))

    print("\n" + "=" * 50)
    print(f" {args.target.upper()} TANIMOTO VALIDATION")
    print("=" * 50)
    print(f"  Mean max Tanimoto : {mean_sim:.3f}")
    print(f"  Median max Tanimoto: {med_sim:.3f}")
    print(f"  % >= 0.4 similarity: {gt_40:.1%}")
    print(f"  Exact copies       : {exact} ({exact / len(max_tanimotos):.1%})")

    # Enriched CSV — SMILES + Tanimoto only (no Score)
    out_csv = Path(args.output_csv) if args.output_csv else results_dir / f"{args.target}_rl_tanimoto_enriched.csv"
    df_out = pd.DataFrame({
        "SMILES": valid_smiles,
        "max_tanimoto": max_tanimotos,
    }).sort_values("max_tanimoto", ascending=False)
    df_out.insert(0, "rank", range(1, len(df_out) + 1))
    df_out.to_csv(out_csv, index=False)
    print(f"[*] Enriched CSV → {out_csv}")

    json_path = results_dir / f"{args.target}_rl_tanimoto.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "mean": mean_sim,
            "median": med_sim,
            "gt_0.4": gt_40,
            "exact_copies": exact,
            "reference": ref_label,
            "input_csv": str(rl_path),
            "output_csv": str(out_csv),
        }, fh, indent=2)

    plt.figure(figsize=(8, 5))
    plt.hist(max_tanimotos, bins=30, alpha=0.7, color="green")
    plt.axvline(mean_sim, color="red", linestyle="dashed", linewidth=1)
    plt.title(f"{args.target.upper()} Generated vs Reference ({ref_label})")
    plt.xlabel("Max Tanimoto Similarity")
    plt.ylabel("Frequency")
    png_path = results_dir / f"{args.target}_rl_tanimoto.png"
    plt.savefig(png_path)
    plt.close()
    print(f"[*] Histogram → {png_path}")


if __name__ == "__main__":
    main()
