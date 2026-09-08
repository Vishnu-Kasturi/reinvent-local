"""Synthetic accessibility reward — exp(SA / 3.0), same as original RL.py."""

from __future__ import annotations

import math

from rdkit import Chem

from sascorer import calculateScore as sa_raw

SA_SCALE = 3.0


def calculateSA(mol: Chem.Mol) -> float:
    return sa_raw(mol)


def calculateScore(mol: Chem.Mol) -> float:
    return float(math.exp(calculateSA(mol) / SA_SCALE))
