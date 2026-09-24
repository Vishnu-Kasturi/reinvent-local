#!/usr/bin/env python3
"""
filter_csv.py — RD Filters on CSV (SMILES column). Edit CONFIG below, then run.

Property limits: all must pass (hard filter).
Alert sets: each enabled set must have zero SMARTS hits to count as "set passed".
alert_set_score = (sets passed) / (enabled sets)
Overall pass: all properties OK AND alert_set_score >= ALERT_SET_PASS_FRACTION

If the input CSV has pIC50 / Solubility / Tyrosine / docking columns, they are
copied through to all output files (canonical names: pIC50, Solubility,
Tyrosine_PiStacking, Docking_Score).

Setup (once):
  pip install rdkit pandas docopt
  pip install -e ./rd_filters

Run (from vendor/):
  python filter_csv.py my_molecules.csv
  python filter_csv.py my_molecules.csv --out results/my_run
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import pandas as pd
from rdkit import Chem

from chem_utils import (
    PROP_COLS,
    calc_properties,
    load_input_table,
    resolve_score_columns,
)

# =============================================================================
# CONFIG
# =============================================================================

PROPERTY_LIMITS = {
    "MW": [350, 710],
    "LogP": [2.0, 7.0],
    "HBD": [0, 5],
    "HBA": [0, 12],
    "TPSA": [50, 170],
    "Rot": [0, 12],
}

ALERT_SETS = {
    "BMS": True,
    "Dundee": True,
    "Glaxo": True,
    "Inpharmatica": True,
    "LINT": True,
    "MLSMR": True,
    "PAINS": True,
    "SureChEMBL": True,
}

ALERT_SET_PASS_FRACTION = 0.5

ALERTS_CSV = None
DEDUPE_SMILES = False

# =============================================================================


def _enabled_alert_sets() -> list[str]:
    return [name for name, on in ALERT_SETS.items() if on]


def _resolve_alerts_path() -> str:
    if ALERTS_CSV:
        p = Path(ALERTS_CSV).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"ALERTS_CSV not found: {p}")
        return str(p)
    return str(resources.files("rd_filters") / "data" / "alert_collection.csv")


def _build_rules_by_set(alerts_path: str, enabled_sets: list[str]) -> dict[str, list]:
    from rd_filters.rd_filters import RDFilters

    rules: dict[str, list] = {}
    for name in enabled_sets:
        rf = RDFilters(alerts_path)
        rf.build_rule_list([name])
        rules[name] = rf.rule_list
    return rules


def _check_property_limits(props: dict[str, float], limits: dict) -> tuple[bool, list[str], list[str]]:
    passed, failed = [], []
    for prop in PROP_COLS:
        lo, hi = limits[prop]
        val = props[prop]
        if lo <= val <= hi:
            passed.append(f"{prop}=[{lo},{hi}]")
        else:
            failed.append(f"{prop}={val:.3g} not in [{lo},{hi}]")
    return len(failed) == 0, passed, failed


def _check_alert_set(mol: Chem.Mol, rule_list: list) -> tuple[bool, str]:
    for smarts_mol, max_val, desc in rule_list:
        if len(mol.GetSubstructMatches(smarts_mol)) > max_val:
            return False, f"{desc} > {max_val}"
    return True, ""


def _alert_set_score(n_sets_passed: int, n_sets_total: int) -> float:
    if n_sets_total == 0:
        return 1.0
    return round(n_sets_passed / n_sets_total, 4)


def evaluate_molecule(
    smiles: str,
    name: str,
    rules_by_set: dict[str, list],
    limits: dict,
) -> dict:
    row: dict = {"SMILES": smiles, "NAME": name}
    enabled = list(rules_by_set.keys())
    n_sets = len(enabled)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        row.update({p: float("nan") for p in PROP_COLS})
        row["property_pass"] = False
        row["alert_sets_passed"] = ""
        row["alert_sets_failed"] = ""
        row["alert_set_score"] = 0.0
        row["rd_filter_pass"] = False
        row["filters_passed"] = ""
        row["filters_failed"] = "INVALID_SMILES"
        return row

    props = calc_properties(mol)
    row.update(props)

    prop_ok, prop_pass_list, prop_fail_list = _check_property_limits(props, limits)
    row["property_pass"] = prop_ok

    sets_ok: list[str] = []
    sets_bad: list[str] = []
    set_fail_reasons: list[str] = []

    for set_name in enabled:
        ok, reason = _check_alert_set(mol, rules_by_set[set_name])
        if ok:
            sets_ok.append(set_name)
        else:
            sets_bad.append(set_name)
            set_fail_reasons.append(f"{set_name}:{reason}")

    n_ok = len(sets_ok)
    score = _alert_set_score(n_ok, n_sets)

    row["alert_sets_passed"] = "; ".join(sets_ok)
    row["alert_sets_failed"] = "; ".join(sets_bad)
    row["alert_set_score"] = score

    overall = prop_ok and (score >= ALERT_SET_PASS_FRACTION)
    row["rd_filter_pass"] = overall

    passed_parts = list(prop_pass_list)
    if sets_ok:
        passed_parts.append(f"AlertSets({','.join(sets_ok)})")
    failed_parts = list(prop_fail_list) + set_fail_reasons
    if n_sets > 0 and score < ALERT_SET_PASS_FRACTION:
        failed_parts.append(
            f"alert_set_score={score} < {ALERT_SET_PASS_FRACTION} "
            f"({n_ok}/{n_sets} sets passed)"
        )

    row["filters_passed"] = "; ".join(passed_parts)
    row["filters_failed"] = "; ".join(failed_parts)
    return row


def _attach_score_columns(df: pd.DataFrame, source: pd.DataFrame, score_map: dict[str, str]) -> pd.DataFrame:
    for canonical, src_col in score_map.items():
        df[canonical] = source[src_col].values
    return df


def write_summary(
    path: Path,
    inp: Path,
    limits: dict,
    alerts_path: str,
    df: pd.DataFrame,
    score_map: dict[str, str],
) -> None:
    enabled = _enabled_alert_sets()
    n = len(df)
    n_pass = int(df["rd_filter_pass"].sum())

    lines = [
        "RD FILTERS RUN SUMMARY",
        "=" * 60,
        f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Input CSV : {inp}",
        f"Alerts CSV: {alerts_path}",
        "",
        "PASSTHROUGH COLUMNS",
        "-" * 40,
    ]
    if score_map:
        for canonical, src in score_map.items():
            lines.append(f"  {canonical} ← {src}")
    else:
        lines.append("  (none detected in input)")

    lines += [
        "",
        "PROPERTY LIMITS (all required — hard filter)",
        "-" * 40,
    ]
    for prop in PROP_COLS:
        lo, hi = limits[prop]
        lines.append(f"  {prop:6s} : {lo} to {hi}")

    lines += [
        "",
        "ALERT SET SCORING",
        "-" * 40,
        f"  Enabled sets           : {', '.join(enabled) if enabled else '(none)'}",
        f"  Pass fraction required : {ALERT_SET_PASS_FRACTION} "
        f"(>= {int(ALERT_SET_PASS_FRACTION * len(enabled))} of {len(enabled)} sets)"
        if enabled
        else "",
        "  alert_set_score        : sets_passed / enabled_sets",
        "",
        "RESULTS",
        "-" * 40,
        f"  Total molecules : {n}",
        f"  Overall pass    : {n_pass} ({100 * n_pass / n:.1f}%)" if n else "  Overall pass    : 0",
        f"  Overall fail    : {n - n_pass}",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="RD Filters on CSV (edit CONFIG in this file)")
    p.add_argument("input_csv", help="Input CSV with SMILES column")
    p.add_argument("--out", default=None, help="Output prefix (default: input stem)")
    args = p.parse_args()

    try:
        import rd_filters  # noqa: F401
    except ImportError:
        raise SystemExit(
            "rd_filters not installed. Run:\n"
            "  pip install rdkit pandas docopt\n"
            "  pip install -e ./rd_filters"
        ) from None

    inp = Path(args.input_csv).expanduser().resolve()
    prefix = Path(args.out).resolve() if args.out else inp.with_suffix("").resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    limits = {k: list(v) for k, v in PROPERTY_LIMITS.items()}
    alerts_path = _resolve_alerts_path()
    enabled_sets = _enabled_alert_sets()

    input_df, _ = load_input_table(inp, dedupe=DEDUPE_SMILES)
    score_map = resolve_score_columns(input_df.columns)

    print(f"[*] Alerts file : {alerts_path}")
    print(f"[*] Alert sets  : {', '.join(enabled_sets) or '(none)'}")
    print(f"[*] Need score >= {ALERT_SET_PASS_FRACTION}")
    if score_map:
        print(f"[*] Passthrough : {', '.join(f'{k}←{v}' for k, v in score_map.items())}")

    rules_by_set = _build_rules_by_set(alerts_path, enabled_sets)
    smiles = input_df["SMILES"].tolist()
    print(f"[*] Filtering {len(smiles)} molecules from {inp.name}")

    rows = [evaluate_molecule(smi, f"mol_{i}", rules_by_set, limits) for i, smi in enumerate(smiles)]
    df = pd.DataFrame(rows)
    df = _attach_score_columns(df, input_df, score_map)

    filter_cols = [
        "SMILES",
        "rd_filter_pass",
        "property_pass",
        "alert_set_score",
        "alert_sets_passed",
        "alert_sets_failed",
        "filters_passed",
        "filters_failed",
        *PROP_COLS,
    ]
    score_cols = list(score_map.keys())
    out_cols = filter_cols + [c for c in score_cols if c not in filter_cols]

    flagged_path = Path(f"{prefix}_flagged.csv")
    passed_path = Path(f"{prefix}_passed.csv")
    failed_path = Path(f"{prefix}_failed.csv")
    summary_path = Path(f"{prefix}_summary.txt")

    df[out_cols].to_csv(flagged_path, index=False)
    df.loc[df["rd_filter_pass"], out_cols].to_csv(passed_path, index=False)
    df.loc[~df["rd_filter_pass"], out_cols].to_csv(failed_path, index=False)
    write_summary(summary_path, inp, limits, alerts_path, df, score_map)

    n_pass = int(df["rd_filter_pass"].sum())
    print(f"[+] Flagged ({len(df)})     → {flagged_path}")
    print(f"[+] Passed  ({n_pass}/{len(df)}) → {passed_path}")
    print(f"[+] Failed  ({len(df) - n_pass}/{len(df)}) → {failed_path}")
    print(f"[+] Summary              → {summary_path}")


if __name__ == "__main__":
    main()
