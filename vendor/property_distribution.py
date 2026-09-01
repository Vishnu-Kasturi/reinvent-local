#!/usr/bin/env python3
"""
property_distribution.py — SMILES-only CSV → property stats (no filtering).

Computes MW, LogP, HBD, HBA, TPSA, Rot for every valid SMILES and writes:
  {prefix}_properties.csv   — per-molecule values
  {prefix}_distribution.txt — min/max/mean/median/percentiles + histogram bins

Reference ranges (from filter_csv defaults) are shown for comparison only.

Run:
  python3 property_distribution.py my_molecules.csv
  python3 property_distribution.py my_molecules.csv --out results/my_dist
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import MolLogP, MolWt, NumHAcceptors, NumHDonors, TPSA
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

SMILES_NAMES = ("smiles", "canonical_smiles", "input_smiles", "SMILES")
PROP_COLS = ("MW", "LogP", "HBD", "HBA", "TPSA", "Rot")
REFERENCE_RANGES = {
    "MW": [0, 500],
    "LogP": [-5, 5],
    "HBD": [0, 5],
    "HBA": [0, 10],
    "TPSA": [0, 200],
    "Rot": [0, 10],
}

DEDUPE_SMILES = False
HISTOGRAM_BINS = 10


def read_smiles(csv_path: Path, dedupe: bool = False) -> list[str]:
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    first = text.splitlines()[0] if text else ""
    sep = ";" if first.count(";") > first.count(",") else ","
    header = pd.read_csv(csv_path, sep=sep, nrows=0)
    col = next((c for c in header.columns if c.strip().lower() in {n.lower() for n in SMILES_NAMES}), None)
    if not col:
        raise SystemExit(f"No SMILES column in {csv_path}. Columns: {list(header.columns)}")
    s = pd.read_csv(csv_path, sep=sep, usecols=[col], dtype=str)[col]
    smiles = s.dropna().astype(str).str.strip().loc[lambda x: x != ""].tolist()
    if dedupe:
        smiles = list(dict.fromkeys(smiles))
    return smiles


def properties_for_smiles(smiles: str) -> dict[str, float] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MW": MolWt(mol),
        "LogP": MolLogP(mol),
        "HBD": float(NumHDonors(mol)),
        "HBA": float(NumHAcceptors(mol)),
        "TPSA": TPSA(mol),
        "Rot": float(CalcNumRotatableBonds(mol)),
    }


def _resolve_input(path: str) -> Path:
    p = Path(path).expanduser()
    if p.exists():
        return p.resolve()
    if not p.suffix and Path(f"{p}.csv").exists():
        return Path(f"{p}.csv").resolve()
    raise SystemExit(f"File not found: {path}  (tried also {path}.csv)")


def _percentiles(series: pd.Series) -> dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {}
    return {
        "min": float(s.min()),
        "p5": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
        "std": float(s.std(ddof=0)),
    }


def _histogram_lines(series: pd.Series, bins: int = 10) -> list[str]:
    s = series.dropna()
    if s.empty:
        return ["  (no data)"]
    counts, edges = np.histogram(s, bins=bins)
    lines = []
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        bar = "#" * int(40 * c / max(counts.max(), 1))
        lines.append(f"  [{lo:8.3g}, {hi:8.3g})  {int(c):5d}  {bar}")
    return lines


def write_distribution(path: Path, inp: Path, df: pd.DataFrame, n_invalid: int) -> None:
    n = len(df)
    lines = [
        "PROPERTY DISTRIBUTION REPORT",
        "=" * 60,
        f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Input CSV : {inp}",
        f"Valid     : {n}",
        f"Invalid   : {n_invalid}",
        "",
        "Reference ranges (informational — NOT applied in this script):",
    ]
    for prop in PROP_COLS:
        lo, hi = REFERENCE_RANGES[prop]
        lines.append(f"  {prop:6s} : [{lo}, {hi}]")

    for prop in PROP_COLS:
        stats = _percentiles(df[prop])
        lo_ref, hi_ref = REFERENCE_RANGES[prop]
        in_range = df[prop].between(lo_ref, hi_ref).sum() if n else 0
        pct_in = 100 * in_range / n if n else 0

        lines += [
            "",
            f"{prop}",
            "-" * 40,
            f"  min    : {stats.get('min', float('nan')):.4g}",
            f"  p5     : {stats.get('p5', float('nan')):.4g}",
            f"  p25    : {stats.get('p25', float('nan')):.4g}",
            f"  median : {stats.get('median', float('nan')):.4g}",
            f"  mean   : {stats.get('mean', float('nan')):.4g}",
            f"  p75    : {stats.get('p75', float('nan')):.4g}",
            f"  p95    : {stats.get('p95', float('nan')):.4g}",
            f"  max    : {stats.get('max', float('nan')):.4g}",
            f"  std    : {stats.get('std', float('nan')):.4g}",
            f"  in ref range [{lo_ref},{hi_ref}] : {in_range}/{n} ({pct_in:.1f}%)",
            "  histogram:",
        ]
        lines.extend(_histogram_lines(df[prop], HISTOGRAM_BINS))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Property distribution from SMILES CSV")
    p.add_argument("input_csv", help="CSV with SMILES column")
    p.add_argument("--out", default=None, help="Output prefix (default: input stem + _dist)")
    p.add_argument("--dedupe", action="store_true", help="Unique SMILES only")
    args = p.parse_args()

    inp = _resolve_input(args.input_csv)
    prefix = Path(args.out).resolve() if args.out else inp.with_suffix("").resolve().parent / f"{inp.stem}_dist"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    smiles_list = read_smiles(inp, dedupe=args.dedupe or DEDUPE_SMILES)
    print(f"[*] Computing properties for {len(smiles_list)} SMILES from {inp.name}")

    rows: list[dict] = []
    n_invalid = 0
    for smi in smiles_list:
        props = properties_for_smiles(smi)
        if props is None:
            n_invalid += 1
            continue
        rows.append({"SMILES": smi, **props})

    df = pd.DataFrame(rows)
    props_path = Path(f"{prefix}_properties.csv")
    dist_path = Path(f"{prefix}_distribution.txt")

    df.to_csv(props_path, index=False)
    write_distribution(dist_path, inp, df, n_invalid)

    print(f"[+] Properties → {props_path}")
    print(f"[+] Distribution → {dist_path}")
    if n_invalid:
        print(f"[!] Skipped {n_invalid} invalid SMILES")


if __name__ == "__main__":
    main()
