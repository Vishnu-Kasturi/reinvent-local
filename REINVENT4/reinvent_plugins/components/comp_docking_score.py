"""
DockingScore — GNINA docking reward component for REINVENT4
------------------------------------------------------------
Tiered reward from GNINA best-pose affinity (mode 1 in mol0_log.txt):

    affinity <= -12.0 kcal/mol  →  1.0  (high)
    -12.0 < affinity <= -10.0   →  0.5  (medium)
    affinity > -10.0            →  0.0  (low)
    failure (bad SMILES, GNINA crash)  →  0.0

TOML:
-----
[[stage.scoring.component]]
[stage.scoring.component.DockingScore]
[[stage.scoring.component.DockingScore.endpoint]]
name                  = "DockingReward"
weight                = 3.0
params.receptor_path  = ["docking/receptor.pdb"]
params.autobox_ligand = ["docking/ref_ligand.pdb"]
params.gnina_executable = ["gnina"]
params.output_root    = ["docking_runs"]
[[stage.scoring.component.DockingScore.endpoint]]
name                  = "DockingAffinity_raw"
weight                = 0.0
params.receptor_path  = ["docking/receptor.pdb"]
params.autobox_ligand = ["docking/ref_ligand.pdb"]
params.gnina_executable = ["gnina"]
params.output_root    = ["docking_runs"]
"""
from __future__ import annotations

__all__ = ["DockingScore"]

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
class DockingScore:
    """
    REINVENT4 component — tiered GNINA docking reward.

    Endpoints returned to REINVENT:
      [0] DockingReward       — 0.0 / 0.5 / 1.0  (use weight > 0)
      [1] DockingAffinity_raw — kcal/mol           (weight = 0, logging only)
    """

    def __init__(self, params: Parameters):
        self.smiles_type = "rdkit_smiles"
        self.number_of_endpoints = 2
        self.config = build_config_from_params(params)
        self.config.validate()
        logger.info(f"[DockingScore] receptor={self.config.receptor_path}")

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> ComponentResults:
        n = len(smilies)
        rewards = np.zeros(n, dtype=np.float32)
        raw_affinity = np.full(n, np.nan, dtype=np.float32)
        try:
            results = BatchCache.get_or_run(smilies, self.config)
            for i, res in enumerate(results):
                rewards[i] = res.docking_reward
                if res.docking_ok:
                    raw_affinity[i] = res.affinity
            logger.info(
                f"[DockingScore] n={n} mean_reward={rewards.mean():.3f} "
                f"best={np.nanmin(raw_affinity):.2f} kcal/mol"
            )
            return ComponentResults([rewards, raw_affinity])
        except Exception as exc:
            logger.error(f"[DockingScore] {exc}\n{traceback.format_exc()}")
            return ComponentResults([
                np.zeros(n, dtype=np.float32),
                np.full(n, np.nan, dtype=np.float32),
            ])
