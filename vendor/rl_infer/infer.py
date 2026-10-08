#!/usr/bin/env python3
"""
infer.py — batch inference for MW, QED, pIC50, solubility rewards.

Run:
  cd ~/vendor/rl_infer
  python3 infer.py smiles.csv
  python3 infer.py smiles.csv --out rewards.csv

Input CSV: any column named SMILES / smiles / canonical_smiles.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_INFER_DIR = Path(__file__).resolve().parent
if str(_INFER_DIR) not in sys.path:
    sys.path.insert(0, str(_INFER_DIR))

from RL import get_predictor, get_reward_breakdown

SMILES_NAMES = ("smiles", "canonical_smiles", "input_smiles", "SMILES")


def _find_smiles_col(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for name in SMILES_NAMES:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise SystemExit(f"No SMILES column found. Columns: {list(df.columns)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Infer MW/QED/pIC50/sol rewards for SMILES CSV")
    p.add_argument("input_csv", help="CSV with SMILES column")
    p.add_argument("--out", default=None, help="Output CSV (default: <input>_rewards.csv)")
    args = p.parse_args()

    inp = Path(args.input_csv).expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"File not found: {inp}")

    df = pd.read_csv(inp)
    col = _find_smiles_col(df)
    predictor = get_predictor()

    rows = []
    for smi in df[col].astype(str):
        smi = smi.strip()
        if not smi or smi.lower() == "nan":
            continue
        row = get_reward_breakdown(smi, predictor)
        row["SMILES"] = smi
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out).expanduser().resolve() if args.out else inp.with_name(f"{inp.stem}_rewards.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[+] Wrote {len(out_df)} rows → {out_path}")
    if not out_df.empty:
        print(out_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
