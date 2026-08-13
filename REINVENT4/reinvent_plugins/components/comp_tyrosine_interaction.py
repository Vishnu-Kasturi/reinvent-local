"""
TyrosineInteraction — ProLIF TYR interaction reward for REINVENT4
"""
from __future__ import annotations

__all__ = ["TyrosineInteraction"]

import logging
import os
import sys
import traceback
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "Preprocess", "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from reinvent_gnina_backend import BatchCache, build_config_from_params  # noqa: E402

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    receptor_path: List[str]
    autobox_ligand: List[str]
    gnina_executable: List[str] = None
    output_root: List[str] = None
    cnn_scoring: List[str] = None
    timeout_sec: List[str] = None
    keep_outputs: List[str] = None


@add_tag("__component")
class TyrosineInteraction:
    """Tiered tyrosine interaction reward via ProLIF on GNINA best pose."""

    def __init__(self, params: Parameters):
        self.smiles_type = "rdkit_smiles"
        self.number_of_endpoints = 2
        self.config = build_config_from_params(params)
        self.config.validate()
        logger.info(f"[TyrosineInteraction] receptor={self.config.receptor_path}")

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> ComponentResults:
        n = len(smilies)
        rewards = np.zeros(n, dtype=np.float32)
        raw_counts = np.zeros(n, dtype=np.float32)
        try:
            results = BatchCache.get_or_run(smilies, self.config)
            n_ok = 0
            for i, res in enumerate(results):
                rewards[i] = res.tyr_interaction_reward
                raw_counts[i] = float(res.tyr_interaction_count)
                if res.prolif_ok:
                    n_ok += 1
            logger.info(
                f"[TyrosineInteraction] batch={n} analyzed={n_ok} "
                f"mean_reward={rewards.mean():.3f}"
            )
            return ComponentResults(
                [rewards, raw_counts],
                metadata={"tyr_pi_stacking": [r.tyr_pi_stacking_count for r in results]},
            )
        except Exception as exc:
            logger.error(f"[TyrosineInteraction] {exc}\n{traceback.format_exc()}")
            return ComponentResults([
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
            ])
