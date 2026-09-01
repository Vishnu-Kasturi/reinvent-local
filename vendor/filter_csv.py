#!/usr/bin/env python3
"""
filter_csv.py — RD Filters on CSV (SMILES column only). Edit CONFIG below, then run.

Setup (once):
  pip install rdkit pandas docopt
  pip install -e ./rd_filters

Run:
  python filter_csv.py my_molecules.csv
  python filter_csv.py my_molecules.csv --out results/my_run

Outputs:
  {prefix}_passed.csv   — passing SMILES + filters_passed
  {prefix}_failed.csv   — failing SMILES + filters_failed
  {prefix}_summary.txt  — settings used + run statistics
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import MolLogP, MolWt, NumHAcceptors, NumHDonors, TPSA
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

# =============================================================================
# CONFIG — edit only this block, then run the script
# =============================================================================

# Property limits [min, max] inclusive
PROPERTY_LIMITS = {
    "MW": [0, 500],
    "LogP": [-5, 5],
    "HBD": [0, 5],
    "HBA": [0, 10],
    "TPSA": [0, 200],
    "Rot": [0, 10],
}

# ChEMBL structural alert sets (True = enabled)
# Sets: BMS, Dundee, Glaxo, Inpharmatica, LINT, MLSMR, PAINS, SureChEMBL
ALERT_SETS = {
    "BMS": False,
    "Dundee": False,
    "Glaxo": False,
    "Inpharmatica": True,
    "LINT": False,
    "MLSMR": False,
    "PAINS": False,
    "SureChEMBL": False,
}

# Path to alert SMARTS file (None = use rd_filters package default)
ALERTS_CSV = None

# Deduplicate SMILES before filtering
DEDUPE_SMILES = False

# =============================================================================
# End CONFIG
# =============================================================================

SMILES_NAMES = ("smiles", "canonical_smiles", "input_smiles", "SMILES")
PROP_COLS = ("MW", "LogP", "HBD", "HBA", "TPSA", "Rot")


def _config_rule_dict() -> dict:
    rules = {k: list(v) for k, v in PROPERTY_LIMITS.items()}
    for name, enabled in ALERT_SETS.items():
        rules[f"Rule_{name}"] = bool(enabled)
    return rules


def _enabled_alert_sets() -> list[str]:
    return [name for name, on in ALERT_SETS.items() if on]


def _resolve_alerts_path() -> str:
    if ALERTS_CSV:
        p = Path(ALERTS_CSV).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"ALERTS_CSV not found: {p}")
        return str(p)
    return str(resources.files("rd_filters") / "data" / "alert_collection.csv")


def read_smiles(csv_path: Path) -> list[str]:
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    first = text.splitlines()[0] if text else ""
    sep = ";" if first.count(";") > first.count(",") else ","
    header = pd.read_csv(csv_path, sep=sep, nrows=0)
    col = next((c for c in header.columns if c.strip().lower() in {n.lower() for n in SMILES_NAMES}), None)
    if not col:
        raise SystemExit(f"No SMILES column in {csv_path}. Columns: {list(header.columns)}")
    s = pd.read_csv(csv_path, sep=sep, usecols=[col], dtype=str)[col]
    smiles = s.dropna().astype(str).str.strip().loc[lambda x: x != ""].tolist()
    if DEDUPE_SMILES:
        smiles = list(dict.fromkeys(smiles))
    return smiles


def _calc_properties(mol: Chem.Mol) -> dict[str, float]:
    return {
        "MW": MolWt(mol),
        "LogP": MolLogP(mol),
        "HBD": float(NumHDonors(mol)),
        "HBA": float(NumHAcceptors(mol)),
        "TPSA": TPSA(mol),
        "Rot": float(CalcNumRotatableBonds(mol)),
    }


def _build_alert_rules(alerts_path: str, enabled_sets: list[str]):
    from rd_filters.rd_filters import RDFilters

    rf = RDFilters(alerts_path)
    rf.build_rule_list(enabled_sets)
    return rf.rule_list


def evaluate_molecule(
    smiles: str,
    name: str,
    rule_list: list,
    rule_dict: dict,
) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    row: dict = {"SMILES": smiles, "NAME": name}

    if mol is None:
        row.update({p: float("nan") for p in PROP_COLS})
        row["rd_filter_pass"] = False
        row["filters_passed"] = ""
        row["filters_failed"] = "INVALID_SMILES"
        return row

    props = _calc_properties(mol)
    row.update(props)

    passed: list[str] = []
    failed: list[str] = []

    for prop in PROP_COLS:
        lo, hi = rule_dict[prop]
        val = props[prop]
        label = f"{prop}=[{lo},{hi}]"
        if lo <= val <= hi:
            passed.append(label)
        else:
            failed.append(f"{prop}={val:.3g} not in [{lo},{hi}]")

    alert_hits: list[str] = []
    for smarts_mol, max_val, desc in rule_list:
        if len(mol.GetSubstructMatches(smarts_mol)) > max_val:
            alert_hits.append(f"{desc} > {max_val}")

    sets_label = "+".join(_enabled_alert_sets()) or "none"
    if alert_hits:
        failed.extend(alert_hits)
    else:
        passed.append(f"Structural({sets_label})")

    row["rd_filter_pass"] = len(failed) == 0
    row["filters_passed"] = "; ".join(passed)
    row["filters_failed"] = "; ".join(failed) if failed else ""
    return row


def write_summary(
    path: Path,
    inp: Path,
    rule_dict: dict,
    alerts_path: str,
    df: pd.DataFrame,
) -> None:
    enabled = _enabled_alert_sets()
    n = len(df)
    n_pass = int(df["rd_filter_pass"].sum())
    n_fail = n - n_pass

    lines = [
        "RD FILTERS RUN SUMMARY",
        "=" * 60,
        f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Input CSV : {inp}",
        f"Alerts CSV: {alerts_path}",
        "",
        "PROPERTY LIMITS (inclusive min, max)",
        "-" * 40,
    ]
    for prop in PROP_COLS:
        lo, hi = rule_dict[prop]
        lines.append(f"  {prop:6s} : {lo} to {hi}")

    lines += [
        "",
        "STRUCTURAL ALERT SETS (ChEMBL / rd_filters)",
        "-" * 40,
    ]
    for name, on in ALERT_SETS.items():
        mark = "ENABLED" if on else "disabled"
        lines.append(f"  {name:14s} : {mark}")

    lines += [
        "",
        "OTHER SETTINGS",
        "-" * 40,
        f"  DEDUPE_SMILES : {DEDUPE_SMILES}",
        f"  Enabled sets  : {', '.join(enabled) if enabled else '(none)'}",
        "",
        "RESULTS",
        "-" * 40,
        f"  Total molecules : {n}",
        f"  Passed          : {n_pass} ({100 * n_pass / n:.1f}%)" if n else "  Passed          : 0",
        f"  Failed          : {n_fail} ({100 * n_fail / n:.1f}%)" if n else "  Failed          : 0",
    ]

    if n_fail > 0:
        lines += ["", "TOP FAILURE REASONS", "-" * 40]
        fail_df = df.loc[~df["rd_filter_pass"]]
        reasons: dict[str, int] = {}
        for txt in fail_df["filters_failed"]:
            for part in str(txt).split("; "):
                part = part.strip()
                if not part:
                    continue
                key = part.split("=")[0] if "=" in part else part.split(" >")[0]
                reasons[key] = reasons.get(key, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"  {count:4d}  {reason}")

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

    rule_dict = _config_rule_dict()
    alerts_path = _resolve_alerts_path()
    enabled_sets = _enabled_alert_sets()

    if not enabled_sets:
        print("[!] Warning: no structural alert sets enabled in CONFIG")

    print(f"[*] Alerts file : {alerts_path}")
    print(f"[*] Alert sets  : {', '.join(enabled_sets) or '(none)'}")
    rule_list = _build_alert_rules(alerts_path, enabled_sets)

    smiles = read_smiles(inp)
    print(f"[*] Filtering {len(smiles)} molecules from {inp.name}")

    rows = [
        evaluate_molecule(smi, f"mol_{i}", rule_list, rule_dict)
        for i, smi in enumerate(smiles)
    ]
    df = pd.DataFrame(rows)

    out_cols = ["SMILES", "rd_filter_pass", "filters_passed", "filters_failed", *PROP_COLS]
    passed_path = Path(f"{prefix}_passed.csv")
    failed_path = Path(f"{prefix}_failed.csv")
    summary_path = Path(f"{prefix}_summary.txt")

    df_pass = df.loc[df["rd_filter_pass"], ["SMILES", "filters_passed", *PROP_COLS]]
    df_fail = df.loc[~df["rd_filter_pass"], ["SMILES", "filters_failed", *PROP_COLS]]

    df_pass.to_csv(passed_path, index=False)
    df_fail.to_csv(failed_path, index=False)
    write_summary(summary_path, inp, rule_dict, alerts_path, df)

    n_pass = len(df_pass)
    print(f"[+] Passed  ({n_pass}/{len(df)}) → {passed_path}")
    print(f"[+] Failed  ({len(df_fail)}/{len(df)}) → {failed_path}")
    print(f"[+] Summary              → {summary_path}")


if __name__ == "__main__":
    main()
