#!/usr/bin/env python3
"""Batch reward inference (no docking). Call init_scorers() first."""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

_INFER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_INFER_DIR))

from RL import geometric_mean_rewards, init_scorers
from pic50_scorer import calculateScore as calculatePic50
from sascorer import calculateScore as calculateSA
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

    sas = calculateSA(mol)
    reward2 = math.exp(sas / 3.0)
    clogp = Descriptors.MolLogP(mol)
    reward3 = 11.0 if -1 <= clogp <= 3 else 1.0
    mw = Descriptors.MolWt(mol)
    reward4 = 11.0 if 380 <= mw <= 810 else 1.0
    qed = QED.qed(mol)
    reward5 = math.exp(qed / 0.3)
    pic50 = calculatePic50(smiles)
    reward6 = math.exp(pic50 / 3.0) if math.isfinite(pic50) else 0.0
    sol = calculateSolubility(smiles)
    reward7 = math.exp((sol - (-13.17)) / 3.0) if math.isfinite(sol) else 0.0

    terms = [reward2, reward3, reward4, reward5, reward6, reward7]
    return {
        "SMILES": smiles,
        "reward": geometric_mean_rewards(terms),
        "reward_sa": reward2,
        "reward_logp": reward3,
        "reward_mw": reward4,
        "reward_qed": reward5,
        "reward_pic50": reward6,
        "reward_sol": reward7,
        "sa": sas,
        "logp": clogp,
        "mw": mw,
        "qed": qed,
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
