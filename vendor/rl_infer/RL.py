"""
RL.py — RBDD CVAE_RL reward entry point.

For vis_rl.py (docking + QSAR + tyrosine):
    Edit vendor/rl_infer/reward_config.py  — set enabled True/False per reward
    OR Sample_index.txt                  — reward_docking=1, reward_tyrosine=0, etc.

    from RL import init_scorers, get_reward
    init_scorers()
    reward = get_reward(dock_result, smiles)

For offline CSV scoring (infer.py — MW/QED/pIC50/sol only):
    from RL import get_predictor, get_reward_breakdown
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
from pic50_scorer import Pic50Predictor, predictPic50, readModel as readPic50Model
from qed_scorer import calculateScore as qed_score
from reward_config import compute_reward, needs_dock_prolif
from sol_scorer import SolPredictor, predictSolubility, readModel as readSolModel

try:
    from sascorer import readFragmentScores
except ImportError:
    def readFragmentScores():
        pass

try:
    from dock_prolif_backend import SingleMoleculeCache
except ImportError:
    SingleMoleculeCache = None  # type: ignore


@dataclass
class RewardPredictor:
    pic50: Pic50Predictor
    sol: SolPredictor


_predictor: RewardPredictor | None = None


def init_scorers() -> None:
    """Call once before vis_rl RL training."""
    readFragmentScores()
    readPic50Model()
    readSolModel()


def get_reward(dock_result, smiles: str, predictor=None) -> float:
    """RBDD RL reward — uses reward_config toggles (geometric mean of enabled terms)."""
    from rdkit import Chem

    if Chem.MolFromSmiles(smiles) is None:
        return 0.0
    total, _breakdown = compute_reward(dock_result, smiles)
    return total


def get_predictor() -> RewardPredictor:
    """Lazy-load pIC50 and solubility models (call once per RL run)."""
    global _predictor
    if _predictor is None:
        _predictor = RewardPredictor(pic50=Pic50Predictor(), sol=SolPredictor())
    return _predictor


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
