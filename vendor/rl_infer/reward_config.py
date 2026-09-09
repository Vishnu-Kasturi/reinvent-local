"""Modular RL reward config: enable/disable components and set weights."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Tuple

from pic50_scorer import calculateScore as calculatePic50
from sol_scorer import calculateScore as calculateSolubility
from reinvent_transforms import (
    transform_docking,
    transform_pic50,
    transform_solubility,
    transform_tyrosine,
    weighted_geometric_mean,
)

# Edit defaults here, or override in Sample_index.txt (see apply_reward_config).
DEFAULT_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "solubility": {"enabled": True, "weight": 5.0},
    "pic50": {"enabled": True, "weight": 4.0},
    "tyrosine": {"enabled": True, "weight": 4.0},
    "docking": {"enabled": True, "weight": 2.0},
}

_COMPONENTS: Dict[str, Dict[str, Any]] = copy.deepcopy(DEFAULT_COMPONENTS)


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def apply_reward_config(index_data: Optional[dict] = None) -> None:
    """Load enable/weight overrides from index file or reset to defaults."""
    global _COMPONENTS
    _COMPONENTS = copy.deepcopy(DEFAULT_COMPONENTS)
    if not index_data:
        return

    for name in _COMPONENTS:
        flag_key = f"reward_{name}"
        weight_key = f"weight_{name}"
        if flag_key in index_data:
            _COMPONENTS[name]["enabled"] = _as_bool(index_data[flag_key])
        if weight_key in index_data:
            _COMPONENTS[name]["weight"] = float(index_data[weight_key])
    validate_reward_config()


def get_reward_components() -> Dict[str, Dict[str, Any]]:
    return copy.deepcopy(_COMPONENTS)


def needs_dock_prolif() -> bool:
    return _COMPONENTS["docking"]["enabled"] or _COMPONENTS["tyrosine"]["enabled"]


def describe_active_rewards() -> str:
    parts = []
    for name, cfg in _COMPONENTS.items():
        state = "ON" if cfg["enabled"] else "OFF"
        parts.append(f"{name}={state}(w={cfg['weight']})")
    return ", ".join(parts)


def compute_reward(dock_result, smiles: str) -> Tuple[float, Dict[str, float]]:
    """
    Build reward from enabled components only (weighted geometric mean).
    Returns (total_reward, per_component_scores).
    """
    breakdown: Dict[str, float] = {}
    pairs = []

    if _COMPONENTS["docking"]["enabled"]:
        if dock_result is None or not dock_result.docking_ok:
            return 0.0, breakdown
        score = transform_docking(dock_result.affinity)
        if score <= 0:
            return 0.0, breakdown
        breakdown["docking"] = score
        pairs.append((score, _COMPONENTS["docking"]["weight"]))

    if _COMPONENTS["tyrosine"]["enabled"]:
        if dock_result is None:
            return 0.0, breakdown
        score = transform_tyrosine(dock_result.tyr_pi_stacking_count)
        if score <= 0:
            return 0.0, breakdown
        breakdown["tyrosine"] = score
        pairs.append((score, _COMPONENTS["tyrosine"]["weight"]))

    if _COMPONENTS["pic50"]["enabled"]:
        pic50 = calculatePic50(smiles)
        if not math.isfinite(pic50):
            return 0.0, breakdown
        score = transform_pic50(pic50)
        if score <= 0:
            return 0.0, breakdown
        breakdown["pic50"] = score
        pairs.append((score, _COMPONENTS["pic50"]["weight"]))

    if _COMPONENTS["solubility"]["enabled"]:
        sol = calculateSolubility(smiles)
        if not math.isfinite(sol):
            return 0.0, breakdown
        score = transform_solubility(sol)
        if score <= 0:
            return 0.0, breakdown
        breakdown["solubility"] = score
        pairs.append((score, _COMPONENTS["solubility"]["weight"]))

    if not pairs:
        return 0.0, breakdown

    return weighted_geometric_mean(pairs), breakdown


def validate_reward_config() -> None:
    if not any(cfg["enabled"] for cfg in _COMPONENTS.values()):
        raise ValueError(
            "All reward components are disabled. Enable at least one in "
            "reward_config.py or Sample_index.txt (reward_pic50=1, etc.)"
        )
