#!/usr/bin/env python3
"""
select_top_mols.py — Top-N molecules from RD-filter passed CSVs (runs 5–8).

Reads *_passed.csv from filter_csv.py (keeps alert_set_score + properties),
predicts pIC50 and solubility (logS) with PD1-PDL1 XGBoost models, ranks by
weighted composite (alert score + pIC50 + sol), writes top 10 per run.

Setup (once):
  pip install rdkit pandas xgboost numpy scikit-learn

Models live next to this script (vendor folder only):
  vendor/Preprocess/final_acc/pd1_pdl1_pic50_final_acc_model.ubj
  vendor/Preprocess/final_acc/pd1_pdl1_pic50_final_acc_scaler.pkl
  vendor/Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj
  vendor/Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl

Run (edit CONFIG below, then):
  python3 select_top_mols.py
  python3 select_top_mols.py --workdir ~/vendor

Outputs (per run):
  {out_dir}/{run}_top10.csv
  {out_dir}/{run}_top10.smi          (mol2mol / REINVENT leads)
  {out_dir}/all_runs_top10.csv
  {out_dir}/select_top_mols_summary.txt
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# =============================================================================
# CONFIG — edit only this block
# =============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = _SCRIPT_DIR  # Preprocess/ lives here (same folder as this script)

PIC50_MODEL = VENDOR_DIR / "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_model.ubj"
PIC50_SCALER = VENDOR_DIR / "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_scaler.pkl"
SOL_MODEL = VENDOR_DIR / "Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj"
SOL_SCALER = VENDOR_DIR / "Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl"

TOP_N = 10

# Ranking weights (higher = more influence on composite score)
WEIGHT_ALERT = 3.0
WEIGHT_PIC50 = 5.0
WEIGHT_SOL = 5.0

# Working directory: where your run*_passed.csv files live (and outputs go)
WORKDIR = _SCRIPT_DIR

# Each run: label + passed CSV from filter_csv (relative to WORKDIR unless absolute)
RUNS = [
    {"name": "run5", "input": "run5_100_passed.csv"},
    {"name": "run6", "input": "run6_100_passed.csv"},
    {"name": "run7", "input": "run7_100_passed.csv"},
    {"name": "run8", "input": "run8_100_passed.csv"},
]

OUT_DIR = WORKDIR  # set to another path to write outputs elsewhere
WRITE_SMI = True   # mol2mol-style .smi (SMILES<TAB>molID)

# If passed CSV has no alert_set_score, try sibling *_flagged.csv automatically
FALLBACK_TO_FLAGGED = True

# =============================================================================

SMILES_NAMES = ("smiles", "canonical_smiles", "input_smiles", "SMILES")
ALERT_COLS = ("alert_set_score", "alert score", "alert_score")


def _resolve_path(base: Path, path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _find_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in lower:
            return lower[alias.lower()]
    for col in df.columns:
        for alias in aliases:
            if alias.lower() in col.lower():
                return col
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    first = text.splitlines()[0] if text else ""
    sep = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(path, sep=sep)


def _flagged_sibling(passed_path: Path) -> Path | None:
    stem = passed_path.name
    for suffix in ("_passed.csv", "_passed.CSV"):
        if stem.endswith(suffix):
            candidate = passed_path.with_name(stem[: -len(suffix)] + "_flagged.csv")
            if candidate.is_file():
                return candidate
    if passed_path.stem.endswith("_passed"):
        candidate = passed_path.with_name(passed_path.stem[:-7] + "_flagged.csv")
        if candidate.is_file():
            return candidate
    return None


def load_run_table(passed_path: Path) -> pd.DataFrame:
    if not passed_path.is_file():
        raise FileNotFoundError(f"Passed CSV not found: {passed_path}")

    df = _read_csv(passed_path)
    col_smiles = _find_column(df, SMILES_NAMES)
    if not col_smiles:
        raise SystemExit(f"No SMILES column in {passed_path}. Columns: {list(df.columns)}")

    col_alert = _find_column(df, ALERT_COLS)
    if col_alert is None and FALLBACK_TO_FLAGGED:
        flagged = _flagged_sibling(passed_path)
        if flagged:
            df_flag = _read_csv(flagged)
            col_flag_smi = _find_column(df_flag, SMILES_NAMES)
            col_flag_alert = _find_column(df_flag, ALERT_COLS)
            if col_flag_smi and col_flag_alert:
                merge_cols = [col_flag_smi, col_flag_alert]
                for extra in ("alert_sets_passed", "alert_sets_failed", "property_pass"):
                    c = _find_column(df_flag, (extra,))
                    if c:
                        merge_cols.append(c)
                sub = df_flag[merge_cols].drop_duplicates(subset=[col_flag_smi])
                df = df.merge(
                    sub,
                    left_on=col_smiles,
                    right_on=col_flag_smi,
                    how="left",
                    suffixes=("", "_flag"),
                )
                col_alert = col_flag_alert if col_flag_alert in df.columns else f"{col_flag_alert}_flag"

    work = pd.DataFrame()
    work["SMILES"] = df[col_smiles].astype(str).str.strip()
    work = work[work["SMILES"] != ""].copy()

    if col_alert and col_alert in df.columns:
        work["alert_set_score"] = pd.to_numeric(df.loc[work.index, col_alert], errors="coerce")
    else:
        work["alert_set_score"] = 1.0  # passed-only file without scores

    # carry filter metadata when present
    for src, dst in (
        ("alert_sets_passed", "alert_sets_passed"),
        ("alert_sets_failed", "alert_sets_failed"),
        ("property_pass", "property_pass"),
        ("MW", "MW"),
        ("LogP", "LogP"),
        ("HBD", "HBD"),
        ("HBA", "HBA"),
        ("TPSA", "TPSA"),
        ("Rot", "Rot"),
    ):
        c = _find_column(df, (src,))
        if c:
            work[dst] = df.loc[work.index, c].values

    work = work.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)
    return work


def _minmax(series: pd.Series) -> pd.Series:
    vals = series.astype(float)
    lo, hi = vals.min(), vals.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.5, index=series.index)
    return (vals - lo) / (hi - lo)


def _weighted_mean(scores: list[float], weights: list[float]) -> float:
    valid = [(s, w) for s, w in zip(scores, weights) if np.isfinite(s)]
    if not valid:
        return float("nan")
    s_arr = np.array([s for s, _ in valid], dtype=np.float64)
    w_arr = np.array([w for _, w in valid], dtype=np.float64)
    return float(np.average(s_arr, weights=w_arr))


def _import_compute_features():
    sys.path.insert(0, str(_SCRIPT_DIR))
    try:
        from pd1_pdl1_features import compute_features  # noqa: WPS433

        return compute_features
    except ImportError as exc:
        raise SystemExit(
            f"Missing feature module: {_SCRIPT_DIR / 'pd1_pdl1_features.py'}\n"
            "Copy pd1_pdl1_features.py into your vendor folder."
        ) from exc


def load_models():
    for label, path in (
        ("pIC50 model", PIC50_MODEL),
        ("pIC50 scaler", PIC50_SCALER),
        ("Sol model", SOL_MODEL),
        ("Sol scaler", SOL_SCALER),
    ):
        if not path.is_file():
            raise SystemExit(
                f"Missing {label}:\n  {path}\n\n"
                f"Expected under vendor folder:\n"
                f"  {VENDOR_DIR / 'Preprocess/final_acc/'}"
            )

    compute_features = _import_compute_features()

    bst_pic50 = xgb.Booster()
    bst_pic50.load_model(str(PIC50_MODEL))
    bst_sol = xgb.Booster()
    bst_sol.load_model(str(SOL_MODEL))

    return compute_features, bst_pic50, bst_sol


def predict_pic50_sol(
    smiles: list[str],
    compute_features,
    bst_pic50: xgb.Booster,
    bst_sol: xgb.Booster,
) -> tuple[list[float], list[float]]:
    X_p, mask_p = compute_features(smiles, str(PIC50_SCALER))
    X_p = X_p[:, :2415]
    preds_p = bst_pic50.predict(xgb.DMatrix(X_p))

    X_s, mask_s = compute_features(smiles, str(SOL_SCALER))
    preds_s = bst_sol.predict(xgb.DMatrix(X_s))

    pic50 = [float(preds_p[i]) if mask_p[i] else float("nan") for i in range(len(smiles))]
    sol = [float(preds_s[i]) if mask_s[i] else float("nan") for i in range(len(smiles))]
    return pic50, sol


def rank_run(df: pd.DataFrame, run_name: str) -> pd.DataFrame:
    work = df.copy()
    alert_n = _minmax(work["alert_set_score"].fillna(0.0))
    pic50_n = _minmax(work["pIC50"])
    sol_n = _minmax(work["Solubility"])

    work["composite"] = [
        _weighted_mean([a, p, s], [WEIGHT_ALERT, WEIGHT_PIC50, WEIGHT_SOL])
        for a, p, s in zip(alert_n, pic50_n, sol_n)
    ]

    ranked = work.sort_values(
        by=["composite", "alert_set_score", "Solubility", "pIC50"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked.insert(0, "run", run_name)
    return ranked


def format_top(df: pd.DataFrame, n: int) -> pd.DataFrame:
    top = df.head(n).copy().reset_index(drop=True)
    top["molID"] = np.arange(len(top), dtype=int)

    base_cols = [
        "run", "rank", "molID", "SMILES",
        "alert_set_score", "pIC50", "Solubility", "composite",
    ]
    optional = [
        "alert_sets_passed", "alert_sets_failed", "property_pass",
        "MW", "LogP", "HBD", "HBA", "TPSA", "Rot",
    ]
    out_cols = [c for c in base_cols if c in top.columns]
    out_cols += [c for c in optional if c in top.columns and c not in out_cols]
    return top[out_cols]


def write_smi(path: Path, df: pd.DataFrame) -> None:
    lines = []
    for i, row in df.iterrows():
        smi = str(row["SMILES"])
        mol_id = int(row["molID"]) if "molID" in df.columns else int(i)
        if Chem.MolFromSmiles(smi) is None:
            continue
        lines.append(f"{smi}\tmol{mol_id}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_summary(path: Path, summaries: list[str]) -> None:
    header = [
        "SELECT TOP MOLECULES — SUMMARY",
        "=" * 60,
        f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Vendor dir: {VENDOR_DIR}",
        f"Top N     : {TOP_N}",
        f"Weights   : alert={WEIGHT_ALERT}, pIC50={WEIGHT_PIC50}, sol={WEIGHT_SOL}",
        "",
    ]
    path.write_text("\n".join(header + summaries) + "\n", encoding="utf-8")


def process_runs(workdir: Path, out_dir: Path) -> None:
    compute_features, bst_pic50, bst_sol = load_models()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_top: list[pd.DataFrame] = []
    summary_lines: list[str] = []

    print(f"[*] Vendor dir: {VENDOR_DIR}")
    print(f"[*] Work dir  : {workdir}")
    print(f"[*] Output dir: {out_dir}")
    print(f"[*] Top N={TOP_N} | weights alert={WEIGHT_ALERT} pIC50={WEIGHT_PIC50} sol={WEIGHT_SOL}\n")

    for spec in RUNS:
        run_name = spec["name"]
        passed_path = _resolve_path(workdir, spec["input"])
        print(f"--- {run_name} ← {passed_path.name} ---")

        if not passed_path.is_file():
            print(f"    SKIP: file not found → {passed_path}\n")
            continue

        try:
            df = load_run_table(passed_path)
        except FileNotFoundError as exc:
            print(f"    SKIP: {exc}\n")
            continue
        print(f"    Molecules: {len(df)}")

        pic50, sol = predict_pic50_sol(df["SMILES"].tolist(), compute_features, bst_pic50, bst_sol)
        df["pIC50"] = pic50
        df["Solubility"] = sol

        n_valid_p = sum(np.isfinite(p) for p in pic50)
        n_valid_s = sum(np.isfinite(s) for s in sol)
        print(f"    Valid pIC50 preds: {n_valid_p}/{len(df)}")
        print(f"    Valid Sol preds  : {n_valid_s}/{len(df)}")

        ranked = rank_run(df, run_name)
        top = format_top(ranked, TOP_N)

        if top.empty:
            print(f"    WARNING: no molecules for {run_name}\n")
            continue

        csv_path = out_dir / f"{run_name}_top{TOP_N}.csv"
        top.to_csv(csv_path, index=False)
        print(f"    → {csv_path}  (cols: {', '.join(top.columns)})")

        if WRITE_SMI:
            smi_path = out_dir / f"{run_name}_top{TOP_N}.smi"
            write_smi(smi_path, top)
            print(f"    → {smi_path}")

        all_top.append(top)

        summary_lines += [
            f"RUN: {run_name}",
            "-" * 40,
            f"  Input     : {passed_path}",
            f"  Input n   : {len(df)}",
            f"  Top saved : {len(top)}",
            f"  pIC50     : min={top['pIC50'].min():.3f} max={top['pIC50'].max():.3f} mean={top['pIC50'].mean():.3f}",
            f"  Sol       : min={top['Solubility'].min():.3f} max={top['Solubility'].max():.3f} mean={top['Solubility'].mean():.3f}",
            f"  Alert scr : min={top['alert_set_score'].min():.3f} max={top['alert_set_score'].max():.3f}",
            "",
            "  Top 3 preview:",
        ]
        preview_cols = ["rank", "molID", "pIC50", "Solubility", "alert_set_score", "composite"]
        summary_lines.append(top[preview_cols].head(3).to_string(index=False))
        summary_lines.append("")

    if not all_top:
        raise SystemExit("No runs produced output. Check RUNS paths in CONFIG.")

    combined = pd.concat(all_top, ignore_index=True)
    combined["molID"] = np.arange(len(combined), dtype=int)
    combined_path = out_dir / f"all_runs_top{TOP_N}.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\n[+] Combined → {combined_path}")

    summary_path = out_dir / "select_top_mols_summary.txt"
    write_summary(summary_path, summary_lines)
    print(f"[+] Summary  → {summary_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Top molecules from RD-filter passed CSVs (edit CONFIG in file)")
    p.add_argument(
        "--workdir",
        default=None,
        help="Directory with run*_passed.csv files (default: WORKDIR in CONFIG)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory (default: OUT_DIR in CONFIG)",
    )
    args = p.parse_args()

    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else WORKDIR.resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else OUT_DIR.resolve()
    process_runs(workdir, out_dir)


if __name__ == "__main__":
    main()
