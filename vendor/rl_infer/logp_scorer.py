"""logP band reward — same as original RL.py."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors

LOGP_MIN = -1.0
LOGP_MAX = 3.0
IN_BAND_SCORE = 11.0
OUT_BAND_SCORE = 1.0


def calculateLogP(mol: Chem.Mol) -> float:
    return float(Descriptors.MolLogP(mol))


def calculateScore(mol: Chem.Mol) -> float:
    clogp = calculateLogP(mol)
    if LOGP_MIN <= clogp <= LOGP_MAX:
        return IN_BAND_SCORE
    return OUT_BAND_SCORE
