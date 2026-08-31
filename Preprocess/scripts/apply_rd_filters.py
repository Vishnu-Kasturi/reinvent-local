#!/usr/bin/env python3
"""
apply_rd_filters.py
===================
Apply Pat Walters rd_filters (ChEMBL structural alerts + RDKit property limits)
to a CSV file. Only the SMILES column is read; all other columns are ignored.

Outputs:
  1. *_flagged.csv  — SMILES + rd_filter_pass + rd_filter_reason (+ properties)
  2. *_passed.csv   — SMILES only, rows where rd_filter_pass is True

Usage:
  python Preprocess/scripts/apply_rd_filters.py \\
      --input_csv results/top_10_hits.csv \\
      --output_prefix results/top_10_rd

Install (offline / DGX — no git+ pip):
  pip install -e ./vendor/rd_filters

On laptop: git clone https://github.com/PatWalters/rd_filters
Copy vendor/rd_filters/ folder to DGX via WinSCP, then pip install -e ./vendor/rd_filters
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from importlib import resources
from pathlib import Path

import pandas as pd

SMILES_COLUMN_NAMES = (
    "smiles",
    "canonical_smiles",
    "input_smiles",
    "generated_smiles",
)


def _detect_sep(path: Path) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
    if header.count(";") > header.count(",") and header.count(";") > header.count("\t"):
        return ";"
    if header.count("\t") > header.count(","):
        return "\t"
    return ","


def find_smiles_column(columns) -> str | None:
    lower_map = {c.strip().lower(): c for c in columns}
    for name in SMILES_COLUMN_NAMES:
        if name in lower_map:
            return lower_map[name]
    return None


def read_smiles_from_csv(path: str | Path) -> list[str]:
    """Read only the SMILES column; ignore all other columns."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    sep = _detect_sep(path)
    header = pd.read_csv(path, sep=sep, nrows=0)
    smi_col = find_smiles_column(header.columns)
    if smi_col is None:
        raise ValueError(
            f"No SMILES column in {path}. "
            f"Expected one of {SMILES_COLUMN_NAMES}; got {list(header.columns)}"
        )

    series = pd.read_csv(path, sep=sep, usecols=[smi_col], dtype=str)[smi_col]
    return series.dropna().astype(str).str.strip().loc[lambda s: s != ""].tolist()


def _resolve_rules_path(rules_path: str | None) -> str:
    if rules_path:
        p = Path(rules_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Rules file not found: {p}")
        return str(p)
    default = resources.files("rd_filters") / "data" / "rules.json"
    return str(default)


def _resolve_alerts_path(alerts_path: str | None) -> str:
    if alerts_path:
        p = Path(alerts_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Alerts file not found: {p}")
        return str(p)
    default = resources.files("rd_filters") / "data" / "alert_collection.csv"
    return str(default)


def _load_rule_dict(rules_path: str) -> dict:
    with open(rules_path, encoding="utf-8") as fh:
        return json.load(fh)


def _build_evaluator(alerts_path: str, rule_dict: dict):
    from rd_filters.rd_filters import RDFilters

    rf = RDFilters(alerts_path)
    alert_names = [
        key.replace("Rule_", "")
        for key, enabled in rule_dict.items()
        if key.startswith("Rule") and enabled
    ]
    rf.build_rule_list(alert_names)
    return rf


def _passes_property_filters(row: pd.Series, rule_dict: dict) -> bool:
    for prop in ("MW", "LogP", "HBD", "HBA", "TPSA", "Rot"):
        lo, hi = rule_dict[prop]
        val = row[prop]
        if pd.isna(val) or not (lo <= val <= hi):
            return False
    return True


def _evaluate_batch(args_tuple):
    batch, alerts_path, rule_dict = args_tuple
    rf = _build_evaluator(alerts_path, rule_dict)
    return [rf.evaluate([smi, f"mol_{i}"]) for i, smi in batch]


def _filter_reason(row: pd.Series, rule_dict: dict) -> str:
    if row["FILTER"] != "OK":
        return str(row["FILTER"])
    for prop in ("MW", "LogP", "HBD", "HBA", "TPSA", "Rot"):
        lo, hi = rule_dict[prop]
        val = row[prop]
        if pd.isna(val) or not (lo <= val <= hi):
            return f"{prop} out of range [{lo}, {hi}]"
    return "OK"


def run_filters(
    smiles_list: list[str],
    alerts_path: str,
    rules_path: str,
    n_cpus: int | None = None,
) -> pd.DataFrame:
    rule_dict = _load_rule_dict(rules_path)
    n_cpus = n_cpus or mp.cpu_count()

    # Assign stable names for rd_filters evaluate()
    input_data = [[smi, f"mol_{i}"] for i, smi in enumerate(smiles_list)]

    if n_cpus <= 1 or len(input_data) < 50:
        rf = _build_evaluator(alerts_path, rule_dict)
        rows = [rf.evaluate(item) for item in input_data]
    else:
        chunk_size = max(1, len(input_data) // n_cpus)
        chunks = [
            (input_data[i : i + chunk_size], alerts_path, rule_dict)
            for i in range(0, len(input_data), chunk_size)
        ]
        with mp.Pool(n_cpus) as pool:
            nested = pool.map(_evaluate_batch, chunks)
        rows = [row for batch in nested for row in batch]

    df = pd.DataFrame(
        rows,
        columns=["SMILES", "NAME", "FILTER", "MW", "LogP", "HBD", "HBA", "TPSA", "Rot"],
    )
    df["rd_filter_pass"] = df.apply(
        lambda r: r["FILTER"] == "OK" and _passes_property_filters(r, rule_dict),
        axis=1,
    )
    df["rd_filter_reason"] = df.apply(lambda r: _filter_reason(r, rule_dict), axis=1)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply rd_filters to a CSV (SMILES column only).")
    p.add_argument("--input_csv", required=True, help="Input CSV with a SMILES column")
    p.add_argument(
        "--output_prefix",
        required=True,
        help="Output path prefix (writes PREFIX_flagged.csv and PREFIX_passed.csv)",
    )
    p.add_argument("--rules", default=None, help="rules.json path (default: rd_filters package)")
    p.add_argument("--alerts", default=None, help="alert_collection.csv path (default: package)")
    p.add_argument("--np", type=int, default=None, help="CPU cores (default: all)")
    p.add_argument("--dedupe", action="store_true", help="Deduplicate SMILES before filtering")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from rd_filters.rd_filters import RDFilters  # noqa: F401
    except ImportError as exc:
        print(
            "rd_filters is not installed.\n"
            "Install from the bundled copy:\n"
            "  pip install -e ./vendor/rd_filters",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    input_path = Path(args.input_csv).expanduser().resolve()
    prefix = Path(args.output_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    flagged_path = Path(f"{prefix}_flagged.csv")
    passed_path = Path(f"{prefix}_passed.csv")

    print(f"[*] Reading SMILES from: {input_path}")
    smiles = read_smiles_from_csv(input_path)
    if args.dedupe:
        smiles = list(dict.fromkeys(smiles))
    print(f"    Molecules to filter: {len(smiles)}")

    alerts_path = _resolve_alerts_path(args.alerts)
    rules_path = _resolve_rules_path(args.rules)
    print(f"[*] Alerts: {alerts_path}")
    print(f"[*] Rules:  {rules_path}")

    df = run_filters(smiles, alerts_path, rules_path, n_cpus=args.np)

    out_flagged = df[["SMILES", "rd_filter_pass", "rd_filter_reason", "MW", "LogP", "HBD", "HBA", "TPSA", "Rot"]]
    out_flagged.to_csv(flagged_path, index=False)

    out_passed = df.loc[df["rd_filter_pass"], ["SMILES"]].reset_index(drop=True)
    out_passed.to_csv(passed_path, index=False)

    n_pass = int(df["rd_filter_pass"].sum())
    n_total = len(df)
    pct = 100.0 * n_pass / n_total if n_total else 0.0

    print(f"[+] Flagged CSV → {flagged_path}")
    print(f"[+] Passed CSV  → {passed_path}  ({n_pass}/{n_total} passed, {pct:.1f}%)")

    print("\nTop failures:")
    fails = out_flagged[~out_flagged["rd_filter_pass"]].head(5)
    if fails.empty:
        print("  (none)")
    else:
        print(fails[["SMILES", "rd_filter_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
