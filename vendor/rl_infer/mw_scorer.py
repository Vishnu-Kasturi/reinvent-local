"""Molecular weight scorer for CVAE_RL-style rewards."""

from __future__ import annotations

import math

from rdkit import Chem
from rdkit.Chem import Descriptors

# PD1-PDL1-like range (edit for your target)
MW_MIN = 380.0
MW_MAX = 810.0
MW_IDEAL = 600.0
MW_SIGMA = 120.0


def calculateMW(mol: Chem.Mol) -> float:
    """Raw molecular weight (Da)."""
    return float(Descriptors.MolWt(mol))


def calculateScore(mol: Chem.Mol) -> float:
    """
    MW reward in the same spirit as the logP band in RL.py:
      high score inside [MW_MIN, MW_MAX], low outside.
    """
    mw = calculateMW(mol)
    if MW_MIN <= mw <= MW_MAX:
        return 11.0
    return 1.0


def calculateGaussianScore(mol: Chem.Mol) -> float:
    """Smooth MW reward peaked at MW_IDEAL."""
    mw = calculateMW(mol)
    z = (mw - MW_IDEAL) / MW_SIGMA
    return float(math.exp(-0.5 * z * z))
