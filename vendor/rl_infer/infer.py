#!/usr/bin/env python3
"""Batch reward inference (no docking). Call init_scorers() first."""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

_INFER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_INFER_DIR))

from RL import init_scorers
from pic50_scorer import calculateScore as calculatePic50
from reinvent_transforms import (
    REWARD_WEIGHTS,
    transform_pic50,
    transform_solubility,
    weighted_geometric_mean,
)
from sol_scorer import calculateScore as calculateSolubility

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

    pic50 = calculatePic50(smiles)
    sol = calculateSolubility(smiles)
    if not math.isfinite(pic50) or not math.isfinite(sol):
        return {
            "SMILES": smiles,
            "reward": 0.0,
            "reward_pic50": 0.0,
            "reward_sol": 0.0,
            "pic50": pic50,
            "solubility": sol,
        }

    reward_pic50 = transform_pic50(pic50)
    reward_sol = transform_solubility(sol)
    reward = weighted_geometric_mean([
        (reward_pic50, REWARD_WEIGHTS["pic50"]),
        (reward_sol, REWARD_WEIGHTS["solubility"]),
    ])

    return {
        "SMILES": smiles,
        "reward": reward,
        "reward_pic50": reward_pic50,
        "reward_sol": reward_sol,
        "pic50": pic50,
        "solubility": sol,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    init_scorers()
    inp = Path(args.input_csv).expanduser().resolve()
    df = pd.read_csv(inp)
    col = _find_smiles_col(df)
    rows = [breakdown(s.strip()) for s in df[col].astype(str) if s.strip() and s.lower() != "nan"]
    out = Path(args.out) if args.out else inp.with_name(f"{inp.stem}_rewards.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[+] {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
