#!/usr/bin/env python3
"""
filter_csv.py — run from vendor/ folder next to your CSVs and rd_filters/

Setup (once):
  conda create -n rd_filters python=3.10 -y
  conda activate rd_filters
  conda install -c conda-forge rdkit pandas -y
  pip install docopt
  pip install -e ./rd_filters

Run:
  python filter_csv.py my_molecules.csv
  python filter_csv.py my_molecules.csv --out results/my_run
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from importlib import resources
from pathlib import Path

import pandas as pd

SMILES_NAMES = ("smiles", "canonical_smiles", "input_smiles", "SMILES")


def read_smiles(csv_path: Path) -> list[str]:
    sep = ";" if ";" in csv_path.read_text(encoding="utf-8", errors="replace").splitlines()[0] else ","
    header = pd.read_csv(csv_path, sep=sep, nrows=0)
    col = next((c for c in header.columns if c.strip().lower() in {n.lower() for n in SMILES_NAMES}), None)
    if not col:
        raise SystemExit(f"No SMILES column in {csv_path}. Columns: {list(header.columns)}")
    s = pd.read_csv(csv_path, sep=sep, usecols=[col], dtype=str)[col]
    return s.dropna().astype(str).str.strip().loc[lambda x: x != ""].tolist()


def main() -> None:
    p = argparse.ArgumentParser(description="RD Filters on a CSV (SMILES column only)")
    p.add_argument("input_csv", help="Your CSV file (in this folder or path)")
    p.add_argument("--out", default=None, help="Output prefix (default: input name + _rd)")
    p.add_argument("--np", type=int, default=1, help="CPU cores (default 1 for laptop)")
    args = p.parse_args()

    try:
        from rd_filters.rd_filters import RDFilters
    except ImportError:
        raise SystemExit(
            "rd_filters not installed. Run:\n"
            "  pip install docopt\n"
            "  pip install -e ./rd_filters"
        ) from None

    inp = Path(args.input_csv).resolve()
    prefix = Path(args.out).resolve() if args.out else inp.with_suffix("").resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    alerts = str(resources.files("rd_filters") / "data" / "alert_collection.csv")
    rules_path = str(resources.files("rd_filters") / "data" / "rules.json")
    with open(rules_path, encoding="utf-8") as fh:
        rule_dict = json.load(fh)

    smiles = read_smiles(inp)
    print(f"Filtering {len(smiles)} molecules from {inp.name}")

    rf = RDFilters(alerts)
    alert_names = [k.replace("Rule_", "") for k, v in rule_dict.items() if k.startswith("Rule") and v]
    rf.build_rule_list(alert_names)

    rows = [rf.evaluate([s, f"mol_{i}"]) for i, s in enumerate(smiles)]
    df = pd.DataFrame(rows, columns=["SMILES", "NAME", "FILTER", "MW", "LogP", "HBD", "HBA", "TPSA", "Rot"])

    def passes(r):
        if r["FILTER"] != "OK":
            return False
        for prop in ("MW", "LogP", "HBD", "HBA", "TPSA", "Rot"):
            lo, hi = rule_dict[prop]
            if not (lo <= r[prop] <= hi):
                return False
        return True

    df["rd_filter_pass"] = df.apply(passes, axis=1)
    df["rd_filter_reason"] = df["FILTER"].where(df["FILTER"] != "OK", "OK")

    flagged = f"{prefix}_flagged.csv"
    passed = f"{prefix}_passed.csv"
    df[["SMILES", "rd_filter_pass", "rd_filter_reason", "MW", "LogP", "HBD", "HBA", "TPSA", "Rot"]].to_csv(flagged, index=False)
    df.loc[df["rd_filter_pass"], ["SMILES"]].to_csv(passed, index=False)

    n = int(df["rd_filter_pass"].sum())
    print(f"Done: {n}/{len(df)} passed")
    print(f"  {flagged}")
    print(f"  {passed}")


if __name__ == "__main__":
    main()
