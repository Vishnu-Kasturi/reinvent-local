"""QED scorer for CVAE_RL-style rewards."""

from __future__ import annotations

import math

from rdkit import Chem
from rdkit.Chem import QED


def calculateQED(mol: Chem.Mol) -> float:
    """Raw QED in [0, 1]."""
    try:
        return float(QED.qed(mol))
    except Exception:
        return 0.0


def calculateScore(mol: Chem.Mol) -> float:
    """
    QED reward — mirrors SA-style exp transform:
      reward = exp(QED / scale)
    """
    qed = calculateQED(mol)
    return float(math.exp(qed / 0.3))
