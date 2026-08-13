"""
TyrosineInteraction — ProLIF TYR56 pi-pi stacking reward for REINVENT4
-----------------------------------------------------------------------
Counts ONLY pi-pi stacking between the ligand (GNINA best pose) and
TYR residue 56. All other TYR residues and interaction types are ignored.

Tiered reward:
    >= 2 pi-pi stacking at TYR56  →  1.0  (high)
    1  pi-pi stacking at TYR56    →  0.5  (ok)
    0                              →  0.0  (no reward)

TOML:
-----
[[stage.scoring.component.TyrosineInteraction.endpoint]]
name                  = "TyrInteractionReward"
weight                = 2.0
params.receptor_path  = ["docking/receptor.pdb"]
params.autobox_ligand = ["docking/ref_ligand.pdb"]
params.tyr_residue    = ["56"]
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
    tyr_residue: List[str] = None


@add_tag("__component")
class TyrosineInteraction:
    """
    REINVENT4 component — tiered tyrosine interaction reward.

    Endpoints returned to REINVENT:
      [0] TyrInteractionReward    — 0.0 / 0.5 / 1.0  (use weight > 0)
      [1] TyrInteractionCount_raw — integer count     (weight = 0, logging only)
    """

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
            for i, res in enumerate(results):
                rewards[i] = res.tyr_interaction_reward
                raw_counts[i] = float(res.tyr_interaction_count)
            logger.info(
                f"[TyrosineInteraction] n={n} mean_reward={rewards.mean():.3f} "
                f"max_tyr={raw_counts.max():.0f}"
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
