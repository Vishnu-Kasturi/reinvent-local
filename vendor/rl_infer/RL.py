"""
RL.py — combined reward for MW, QED, pIC50, solubility.

Mirrors the CVAE_RL pattern used with sascorer.py:
  from mw_scorer import calculateScore as mw_score
  from qed_scorer import calculateScore as qed_score
  ...

Usage:
    from RL import get_reward, get_predictor

    predictor = get_predictor()          # load XGBoost models once
    reward = get_reward(smiles, predictor)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

_INFER_DIR = Path(__file__).resolve().parent
if str(_INFER_DIR) not in sys.path:
    sys.path.insert(0, str(_INFER_DIR))

from mw_scorer import calculateScore as mw_score
from pic50_scorer import Pic50Predictor, predictPic50
from qed_scorer import calculateScore as qed_score
from sol_scorer import SolPredictor, predictSolubility


@dataclass
class RewardPredictor:
    pic50: Pic50Predictor
    sol: SolPredictor


_predictor: RewardPredictor | None = None


def get_predictor() -> RewardPredictor:
    """Lazy-load pIC50 and solubility models (call once per RL run)."""
    global _predictor
    if _predictor is None:
        _predictor = RewardPredictor(pic50=Pic50Predictor(), sol=SolPredictor())
    return _predictor


def get_reward(docking_outfile, smiles, predictor=None) -> float:
    """
    Combined reward from MW, QED, pIC50, and solubility.

    CVAE_RL signature: get_reward(docking_outfile, smiles, predictor)
      docking_outfile — ignored (kept for backward compatibility)
      smiles          — generated SMILES
      predictor       — RewardPredictor from get_predictor()

    Final reward = mean of four terms.
    """
    if predictor is None:
        predictor = get_predictor()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    reward_mw = mw_score(mol)
    reward_qed = qed_score(mol)
    reward_pic50 = _pic50_reward(smiles, predictor.pic50)
    reward_sol = _sol_reward(smiles, predictor.sol)

    return float(np.mean([reward_mw, reward_qed, reward_pic50, reward_sol]))


def get_reward_smiles_only(smiles: str, predictor: RewardPredictor | None = None) -> float:
    """Shorthand: get_reward(None, smiles, predictor)."""
    return get_reward(None, smiles, predictor)


def get_reward_breakdown(smiles: str, predictor: RewardPredictor | None = None) -> dict:
    """Return per-term rewards plus raw property values."""
    if predictor is None:
        predictor = get_predictor()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "reward": 0.0,
            "reward_mw": 0.0,
            "reward_qed": 0.0,
            "reward_pic50": 0.0,
            "reward_sol": 0.0,
            "mw": float("nan"),
            "qed": float("nan"),
            "pic50": float("nan"),
            "solubility": float("nan"),
        }

    from mw_scorer import calculateMW
    from qed_scorer import calculateQED

    reward_mw = mw_score(mol)
    reward_qed = qed_score(mol)
    reward_pic50 = _pic50_reward(smiles, predictor.pic50)
    reward_sol = _sol_reward(smiles, predictor.sol)
    pic50 = predictPic50(smiles, predictor.pic50)
    sol = predictSolubility(smiles, predictor.sol)

    return {
        "reward": float(np.mean([reward_mw, reward_qed, reward_pic50, reward_sol])),
        "reward_mw": reward_mw,
        "reward_qed": reward_qed,
        "reward_pic50": reward_pic50,
        "reward_sol": reward_sol,
        "mw": calculateMW(mol),
        "qed": calculateQED(mol),
        "pic50": pic50,
        "solubility": sol,
    }


def _pic50_reward(smiles: str, predictor: Pic50Predictor) -> float:
    from pic50_scorer import calculateScore

    return calculateScore(smiles, predictor)


def _sol_reward(smiles: str, predictor: SolPredictor) -> float:
    from sol_scorer import calculateScore

    return calculateScore(smiles, predictor)
