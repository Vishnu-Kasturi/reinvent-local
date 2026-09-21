"""
Modular RBDD RL reward config — turn any component ON/OFF and set weights.

Edit flags below, OR override per run in Sample_index.txt:

  reward_docking=1
  reward_pic50=1
  reward_solubility=1
  reward_tyrosine=0
  weight_docking=2.0
  weight_pic50=4.0
  weight_solubility=5.0
  weight_tyrosine=4.0

  use_arithmetic_mean=1      # gentler total reward (~0.2–0.4 typical)
  use_geometric_mean=0       # strict REINVENT-style (often ~0.05–0.15)

  use_multinomial_sampling=1 # stochastic decoder (more diversity)
  use_binomial_sampling=0    # greedy argmax decoder (more repeat)

Used by vis_rl.py via apply_reward_config(index_data) at startup.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Tuple

from decoder_sampling import apply_sampling_config, describe_sampling_mode
from pic50_scorer import calculateScore as calculatePic50
from sol_scorer import calculateScore as calculateSolubility
from reinvent_transforms import (
    transform_docking,
    transform_pic50,
    transform_solubility,
    transform_tyrosine,
    weighted_arithmetic_mean,
    weighted_geometric_mean,
)

# ── Aggregation: enable exactly ONE ──────────────────────────────────────────
USE_ARITHMETIC_MEAN = True   # recommended for RBDD RL (less punishing)
USE_GEOMETRIC_MEAN = False
# ─────────────────────────────────────────────────────────────────────────────

# ── Reward components: enable (True/False) + weight ─────────────────────────
DEFAULT_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "solubility": {"enabled": True, "weight": 5.0},
    "pic50":      {"enabled": True, "weight": 4.0},
    "tyrosine":   {"enabled": True, "weight": 4.0},
    "docking":    {"enabled": True, "weight": 2.0},
}
# ─────────────────────────────────────────────────────────────────────────────

_COMPONENTS: Dict[str, Dict[str, Any]] = copy.deepcopy(DEFAULT_COMPONENTS)


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def apply_reward_config(index_data: Optional[dict] = None) -> None:
    """Load enable/weight/aggregation/sampling overrides from index file."""
    global _COMPONENTS, USE_ARITHMETIC_MEAN, USE_GEOMETRIC_MEAN

    _COMPONENTS = copy.deepcopy(DEFAULT_COMPONENTS)
    USE_ARITHMETIC_MEAN = True
    USE_GEOMETRIC_MEAN = False

    if index_data:
        for name in _COMPONENTS:
            flag_key = f"reward_{name}"
            weight_key = f"weight_{name}"
            if flag_key in index_data:
                _COMPONENTS[name]["enabled"] = _as_bool(index_data[flag_key])
            if weight_key in index_data:
                _COMPONENTS[name]["weight"] = float(index_data[weight_key])

        if "use_arithmetic_mean" in index_data:
            USE_ARITHMETIC_MEAN = _as_bool(index_data["use_arithmetic_mean"])
        if "use_geometric_mean" in index_data:
            USE_GEOMETRIC_MEAN = _as_bool(index_data["use_geometric_mean"])

        apply_sampling_config(index_data)

    validate_reward_config()


def get_reward_components() -> Dict[str, Dict[str, Any]]:
    return copy.deepcopy(_COMPONENTS)


def needs_dock_prolif() -> bool:
    return _COMPONENTS["docking"]["enabled"] or _COMPONENTS["tyrosine"]["enabled"]


def describe_active_rewards() -> str:
    agg = "arithmetic" if USE_ARITHMETIC_MEAN else "geometric"
    parts = [f"agg={agg}", f"sampling={describe_sampling_mode()}"]
    for name, cfg in _COMPONENTS.items():
        state = "ON" if cfg["enabled"] else "OFF"
        parts.append(f"{name}={state}(w={cfg['weight']})")
    return ", ".join(parts)


def aggregate_scores(pairs: list[tuple[float, float]]) -> float:
    """Combine component scores using enabled aggregation mode."""
    if USE_ARITHMETIC_MEAN:
        return weighted_arithmetic_mean(pairs)
    return weighted_geometric_mean(pairs)


def compute_reward(dock_result, smiles: str) -> Tuple[float, Dict[str, float]]:
    """
    Build reward from enabled components only.
    Returns (total_reward, per_component_scores).
    """
    breakdown: Dict[str, float] = {}
    pairs = []

    if _COMPONENTS["docking"]["enabled"]:
        if dock_result is None or not dock_result.docking_ok:
            return 0.0, breakdown
        score = transform_docking(dock_result.affinity)
        if score <= 0 and USE_GEOMETRIC_MEAN:
            return 0.0, breakdown
        breakdown["docking"] = score
        pairs.append((score, _COMPONENTS["docking"]["weight"]))

    if _COMPONENTS["tyrosine"]["enabled"]:
        if dock_result is None:
            return 0.0, breakdown
        score = transform_tyrosine(dock_result.tyr_pi_stacking_count)
        if score <= 0 and USE_GEOMETRIC_MEAN:
            return 0.0, breakdown
        breakdown["tyrosine"] = score
        pairs.append((score, _COMPONENTS["tyrosine"]["weight"]))

    if _COMPONENTS["pic50"]["enabled"]:
        pic50 = calculatePic50(smiles)
        if not math.isfinite(pic50):
            return 0.0, breakdown
        score = transform_pic50(pic50)
        if score <= 0 and USE_GEOMETRIC_MEAN:
            return 0.0, breakdown
        breakdown["pic50"] = score
        pairs.append((score, _COMPONENTS["pic50"]["weight"]))

    if _COMPONENTS["solubility"]["enabled"]:
        sol = calculateSolubility(smiles)
        if not math.isfinite(sol):
            return 0.0, breakdown
        score = transform_solubility(sol)
        if score <= 0 and USE_GEOMETRIC_MEAN:
            return 0.0, breakdown
        breakdown["solubility"] = score
        pairs.append((score, _COMPONENTS["solubility"]["weight"]))

    if not pairs:
        return 0.0, breakdown

    total = aggregate_scores(pairs)
    if USE_GEOMETRIC_MEAN and total <= 0:
        return 0.0, breakdown
    return total, breakdown


def validate_reward_config() -> None:
    if not any(cfg["enabled"] for cfg in _COMPONENTS.values()):
        raise ValueError(
            "All reward components are disabled. Enable at least one in "
            "reward_config.py or Sample_index.txt (reward_pic50=1, etc.)"
        )
    if USE_ARITHMETIC_MEAN == USE_GEOMETRIC_MEAN:
        raise ValueError(
            "Enable exactly one aggregation mode: "
            "USE_ARITHMETIC_MEAN or USE_GEOMETRIC_MEAN "
            "(or use_arithmetic_mean=1 / use_geometric_mean=1 in Sample_index.txt)"
        )
