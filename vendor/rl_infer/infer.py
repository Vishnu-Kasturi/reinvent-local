#!/usr/bin/env python3
"""Batch reward inference (no docking). Call init_scorers() first."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

_INFER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_INFER_DIR))

from RL import init_scorers
from reward_config import apply_reward_config, compute_reward, describe_active_rewards

SMILES_NAMES = ("smiles", "canonical_smiles", "input_smiles", "SMILES")


def _find_smiles_col(df):
    lower = {c.lower(): c for c in df.columns}
    for name in SMILES_NAMES:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise SystemExit(f"No SMILES column. Columns: {list(df.columns)}")


def breakdown(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"reward": 0.0, "SMILES": smiles}

    reward, terms = compute_reward(None, smiles)
    row = {"SMILES": smiles, "reward": reward}
    for key, val in terms.items():
        row[f"reward_{key}"] = val
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    init_scorers()
    apply_reward_config()
    print("Reward components:", describe_active_rewards())
    inp = Path(args.input_csv).expanduser().resolve()
    df = pd.read_csv(inp)
    col = _find_smiles_col(df)
    rows = [breakdown(s.strip()) for s in df[col].astype(str) if s.strip() and s.lower() != "nan"]
    out = Path(args.out) if args.out else inp.with_name(f"{inp.stem}_rewards.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[+] {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
