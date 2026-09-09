"""REINVENT4-compatible score transforms and weighted geometric mean."""

import math

import numpy as np

# iict_new_reward.toml stage scoring (QSAR sigmoid transforms).
PIC50_SIGMOID = {"low": 5.0, "high": 12.0, "k": 0.4}
SOL_DOUBLE_SIGMOID = {
    "low": -4.0,
    "high": 2.0,
    "coef_div": 100.0,
    "coef_si": 10.0,
    "coef_se": 10.0,
}

# Tiered docking / tyrosine weights from iict_new_reward.toml.
REWARD_WEIGHTS = {
    "solubility": 5.0,
    "pic50": 4.0,
    "tyrosine": 3.0,
    "docking": 2.0,
}


def _hard_sigmoid(x, k):
    return float((k * x > 0))


def _stable_sigmoid(x, k, base_10=True):
    h = k * x
    if base_10:
        h = h * math.log(10)
    if h >= 0:
        return 1.0 / (1.0 + math.exp(-h))
    exp_h = math.exp(h)
    return exp_h / (1.0 + exp_h)


def sigmoid_transform(value, low, high, k):
    x = value - (high + low) / 2.0
    if (high - low) == 0:
        return _hard_sigmoid(x, 10.0 * k)
    k_eff = 10.0 * k / (high - low)
    return _stable_sigmoid(x, k_eff)


def double_sigmoid_transform(value, low, high, coef_div, coef_si, coef_se):
    x = float(value)
    x_center = (high - low) / 2.0 + low

    if x < x_center:
        xl = x - low
        if coef_div == 0:
            left = _hard_sigmoid(xl, coef_si)
        else:
            left = _stable_sigmoid(xl, coef_si / coef_div)
        return left

    xr = x - high
    if coef_div == 0:
        right = 1.0 - _hard_sigmoid(xr, coef_se)
    else:
        right = 1.0 - _stable_sigmoid(xr, coef_se / coef_div)
    return right


def transform_pic50(pic50):
    p = PIC50_SIGMOID
    return sigmoid_transform(pic50, p["low"], p["high"], p["k"])


def transform_solubility(logs):
    p = SOL_DOUBLE_SIGMOID
    return double_sigmoid_transform(
        logs, p["low"], p["high"], p["coef_div"], p["coef_si"], p["coef_se"]
    )


def weighted_geometric_mean(scores_and_weights):
    """REINVENT geometric_mean: prod(score_i ** (w_i / sum(w)))."""
    if not scores_and_weights:
        return 0.0

    scores = np.array([s for s, _ in scores_and_weights], dtype=np.float64)
    weights = np.array([w for _, w in scores_and_weights], dtype=np.float64)

    if not np.all(np.isfinite(scores)) or np.any(scores <= 0):
        return 0.0

    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return 0.0

    scores = np.maximum(scores, 1e-8)
    exponents = weights / weight_sum
    return float(np.prod(scores ** exponents))
