"""
Decoder token sampling toggles for vis_rl.py.

USE_MULTINOMIAL_SAMPLING = True  → stochastic sample from softmax (explores)
USE_BINOMIAL_SAMPLING    = False → greedy argmax (deterministic, peaked)

Override in Sample_index.txt:
  use_multinomial_sampling=1
  use_binomial_sampling=0
"""

from __future__ import annotations

import torch

USE_MULTINOMIAL_SAMPLING = True
USE_BINOMIAL_SAMPLING = False


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def apply_sampling_config(index_data: dict | None = None) -> None:
    """Load sampling toggles from index file or reset to module defaults."""
    global USE_MULTINOMIAL_SAMPLING, USE_BINOMIAL_SAMPLING

    if index_data is None:
        return

    if "use_multinomial_sampling" in index_data:
        USE_MULTINOMIAL_SAMPLING = _as_bool(index_data["use_multinomial_sampling"])
    if "use_binomial_sampling" in index_data:
        USE_BINOMIAL_SAMPLING = _as_bool(index_data["use_binomial_sampling"])

    validate_sampling_config()


def validate_sampling_config() -> None:
    if USE_MULTINOMIAL_SAMPLING == USE_BINOMIAL_SAMPLING:
        raise ValueError(
            "Enable exactly one decoder mode: "
            "USE_MULTINOMIAL_SAMPLING or USE_BINOMIAL_SAMPLING "
            "(or use_multinomial_sampling=1 / use_binomial_sampling=1 in Sample_index.txt)"
        )


def describe_sampling_mode() -> str:
    if USE_MULTINOMIAL_SAMPLING:
        return "multinomial (stochastic)"
    return "binomial (greedy argmax)"


def sample_next_token(probs: torch.Tensor) -> int:
    """
    Pick next SMILES character index from decoder softmax probabilities.

    probs: 1D float tensor (vocab_size,) or (1, 1, vocab_size)
    """
    validate_sampling_config()
    flat = probs.view(-1)

    if USE_BINOMIAL_SAMPLING:
        return int(torch.argmax(flat).item())

    # Multinomial: one draw from categorical distribution
    return int(torch.multinomial(flat, 1).item())
