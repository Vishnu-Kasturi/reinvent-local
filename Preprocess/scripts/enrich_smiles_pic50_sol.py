#!/usr/bin/env python3
"""
Add predicted pIC50 and Solubility (logS) columns from a SMILES table (CSV or Excel).

Recognizes SMILES column aliases: SMILES, smiles, canonical_smiles, input_smiles.

Usage:
  python Preprocess/scripts/enrich_smiles_pic50_sol.py --input molecules.csv --output enriched.csv
  python Preprocess/scripts/enrich_smiles_pic50_sol.py --input Selected_mols.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "Preprocess" / "scripts"
sys.path.insert(0, str(_REPO_ROOT / "REINVENT4"))
sys.path.insert(0, str(_SCRIPTS))

from score_cluster_smiles_csv import (  # noqa: E402
    PIC50_MODEL,
    PIC50_SCALER,
    SOL_MODEL,
    SOL_SCALER,
    XGB_DEVICE,
    _find_smiles_col,
    _load_input_table,
)

# ── CONFIG (used when CLI flags omitted) ─────────────────────────────────────
INPUT_PATH = _REPO_ROOT / "data/sowmya/Selected_mols_filtered.xlsx"
OUTPUT_PATH = None  # default: <input_stem>_pic50_sol.csv next to input
# ─────────────────────────────────────────────────────────────────────────────


def _load_booster(path: Path, device: str) -> xgb.Booster:
    booster = xgb.Booster({"device": device})
    booster.load_model(str(path))
    return booster


def _predict_column(
    smiles: list[str],
    model: xgb.Booster,
    scaler: Path,
    feature_fn,
) -> list[float]:
    features, mask = feature_fn(smiles, str(scaler))
    preds = model.predict(xgb.DMatrix(features, nthread=-1))
    return [
        round(float(preds[i]), 4) if mask[i] and np.isfinite(preds[i]) else np.nan
        for i in range(len(smiles))
    ]


def enrich_table(input_path: Path, output_path: Path, device: str = XGB_DEVICE) -> Path:
    from reinvent_plugins.components.nophyschem_features import (
        compute_features as compute_pic50_features,
    )
    from reinvent_plugins.components.pd1_pdl1_features import (
        compute_features as compute_sol_features,
    )

    df = _load_input_table(input_path)
    smi_col = _find_smiles_col(df)
    print(f"Loaded {len(df)} rows from {input_path} (column {smi_col!r})")

    out = df.copy()
    smiles = out[smi_col].astype(str).str.strip().tolist()

    pic50_model = _load_booster(PIC50_MODEL, device)
    sol_model = _load_booster(SOL_MODEL, device)
    out["pIC50"] = _predict_column(smiles, pic50_model, PIC50_SCALER, compute_pic50_features)
    out["Solubility"] = _predict_column(smiles, sol_model, SOL_SCALER, compute_sol_features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"saved -> {output_path}")
    print(
        f"pIC50 valid: {out['pIC50'].notna().sum()}/{len(out)}  "
        f"Solubility valid: {out['Solubility'].notna().sum()}/{len(out)}"
    )
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(description="Add pIC50 and Solubility columns from SMILES")
    p.add_argument("--input", type=Path, default=None, help="Input .csv or .xlsx")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <input>_pic50_sol.csv)",
    )
    p.add_argument("--device", default=XGB_DEVICE, help="XGBoost device, e.g. cuda:0 or cpu")
    args = p.parse_args()

    inp = (args.input or INPUT_PATH).expanduser().resolve()
    if not inp.is_absolute():
        inp = (_REPO_ROOT / inp).resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    if args.output:
        out = args.output.expanduser().resolve()
        if not out.is_absolute():
            out = (_REPO_ROOT / out).resolve()
    else:
        default_out = OUTPUT_PATH
        if default_out:
            out = Path(default_out).expanduser().resolve()
        else:
            out = inp.with_name(f"{inp.stem}_pic50_sol.csv")

    enrich_table(inp, out, device=args.device)


if __name__ == "__main__":
    main()
