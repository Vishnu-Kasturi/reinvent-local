"""
RL.py — combined reward: docking + SA + logP + MW + QED + pIC50 + solubility.

REINVENT-style geometric mean over all active terms.

Usage:
    from RL import get_reward, get_predictor

    predictor = get_predictor()
    reward = get_reward(docking_outfile, smiles, predictor)
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

from docking_scorer import calculateScore as docking_score
from logp_scorer import calculateLogP, calculateScore as logp_score
from mw_scorer import calculateMW, calculateScore as mw_score
from pic50_scorer import Pic50Predictor, predictPic50
from qed_scorer import calculateQED, calculateScore as qed_score
from sa_scorer import calculateSA, calculateScore as sa_score
from sol_scorer import SolPredictor, predictSolubility


@dataclass
class RewardPredictor:
    pic50: Pic50Predictor
    sol: SolPredictor


_predictor: RewardPredictor | None = None


def geometric_mean_rewards(rewards: list[float]) -> float:
    """
    REINVENT-style aggregation: geometric mean of component rewards.

    All terms must be finite and > 0; otherwise returns 0.0.
    """
    if not rewards:
        return 0.0
    vals = np.array(rewards, dtype=np.float64)
    if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(vals))))


def _collect_reward_terms(
    docking_outfile,
    smiles: str,
    predictor: RewardPredictor,
    mol: Chem.Mol,
) -> dict[str, float]:
    """Compute all component rewards (original + new)."""
    terms: dict[str, float] = {}

    dock = docking_score(docking_outfile)
    if dock is not None:
        terms["reward_docking"] = dock

    terms["reward_sa"] = sa_score(mol)
    terms["reward_logp"] = logp_score(mol)
    terms["reward_mw"] = mw_score(mol)
    terms["reward_qed"] = qed_score(mol)
    terms["reward_pic50"] = _pic50_reward(smiles, predictor.pic50)
    terms["reward_sol"] = _sol_reward(smiles, predictor.sol)

    return terms


def get_predictor() -> RewardPredictor:
    """Lazy-load pIC50 and solubility models (call once per RL run)."""
    global _predictor
    if _predictor is None:
        _predictor = RewardPredictor(pic50=Pic50Predictor(), sol=SolPredictor())
    return _predictor


def get_reward(docking_outfile, smiles, predictor=None) -> float:
    """
    Combined reward (geometric mean of all terms).

    Original terms: docking, SA, logP
    New terms:      MW, QED, pIC50, solubility

    docking_outfile: path from perform_docking() — omit/None to skip docking term
    """
    if predictor is None:
        predictor = get_predictor()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    terms = _collect_reward_terms(docking_outfile, smiles, predictor, mol)
    return geometric_mean_rewards(list(terms.values()))


def get_reward_breakdown(docking_outfile, smiles: str, predictor: RewardPredictor | None = None) -> dict:
    """Per-term rewards, raw values, and combined geometric mean."""
    if predictor is None:
        predictor = get_predictor()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"reward": 0.0}

    terms = _collect_reward_terms(docking_outfile, smiles, predictor, mol)
    dock_raw = None
    if docking_outfile:
        try:
            from docking_scorer import extract_docking_score

            dock_raw = extract_docking_score(docking_outfile)
        except Exception:
            dock_raw = float("nan")

    return {
        "reward": geometric_mean_rewards(list(terms.values())),
        **terms,
        "docking_score": dock_raw,
        "sa": calculateSA(mol),
        "logp": calculateLogP(mol),
        "mw": calculateMW(mol),
        "qed": calculateQED(mol),
        "pic50": predictPic50(smiles, predictor.pic50),
        "solubility": predictSolubility(smiles, predictor.sol),
    }


def _pic50_reward(smiles: str, predictor: Pic50Predictor) -> float:
    from pic50_scorer import calculateScore

    return calculateScore(smiles, predictor)


def _sol_reward(smiles: str, predictor: SolPredictor) -> float:
    from sol_scorer import calculateScore

    return calculateScore(smiles, predictor)
