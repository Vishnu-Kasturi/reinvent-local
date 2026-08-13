"""
GninaDock — On-the-fly Gnina docking scoring component for REINVENT4
-----------------------------------------------------------------------
Docks each generated SMILES against a fixed receptor using Gnina and returns
binding affinity (kcal/mol) as a reward signal during RL or Mol2Mol runs.

TOML usage (add to any stage.scoring section):
------------------------------------------------
[[stage.scoring.component]]
[stage.scoring.component.GninaDock]
[[stage.scoring.component.GninaDock.endpoint]]
name                    = "GninaAffinity"
weight                  = 3.0
transform.type          = "reverse_sigmoid"
transform.high          = -6.0
transform.low           = -12.0
transform.k             = 0.3
params.receptor_path    = ["docking/receptor.pdbqt"]
params.gnina_executable = ["gnina"]
params.center_x         = ["10.0"]
params.center_y         = ["20.0"]
params.center_z         = ["30.0"]
params.size_x           = ["20.0"]
params.size_y           = ["20.0"]
params.size_z           = ["20.0"]
params.exhaustiveness   = ["4"]
params.num_modes        = ["1"]
params.cnn_scoring      = ["rescore"]
params.docking_mode     = ["full"]
params.n_workers        = ["4"]
params.use_gpu          = ["true"]

# Raw affinity for logging (weight=0, not used in reward)
[[stage.scoring.component.GninaDock.endpoint]]
name                    = "GninaAffinity_raw"
weight                  = 0.0
params.receptor_path    = ["docking/receptor.pdbqt"]
params.gnina_executable = ["gnina"]
params.center_x         = ["10.0"]
params.center_y         = ["20.0"]
params.center_z         = ["30.0"]
params.size_x           = ["20.0"]
params.size_y           = ["20.0"]
params.size_z           = ["20.0"]
params.exhaustiveness   = ["4"]
params.num_modes        = ["1"]
params.cnn_scoring      = ["rescore"]
params.docking_mode     = ["full"]
params.n_workers        = ["4"]
params.use_gpu          = ["true"]

Alternative: autobox from a co-crystallized ligand instead of explicit center/size:
  params.autobox_ligand = ["docking/ref_ligand.sdf"]
  params.autobox_add    = ["4.0"]
  (omit center_x/y/z)
"""
from __future__ import annotations

__all__ = ["GninaDock"]

import logging
import os
import sys
import traceback
from typing import List, Optional

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles

# Import the batch docker from Preprocess/scripts
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "Preprocess", "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from gnina_dock import GninaConfig, GninaDocker  # noqa: E402

logger = logging.getLogger("reinvent")


def _parse_float(val: Optional[str]) -> Optional[float]:
    if val is None or val == "":
        return None
    return float(val)


def _parse_bool(val: str) -> bool:
    return str(val).lower() in ("true", "1", "yes")


@add_tag("__parameters")
@dataclass
class Parameters:
    receptor_path: List[str]
    gnina_executable: List[str] = None
    center_x: List[str] = None
    center_y: List[str] = None
    center_z: List[str] = None
    size_x: List[str] = None
    size_y: List[str] = None
    size_z: List[str] = None
    autobox_ligand: List[str] = None
    autobox_add: List[str] = None
    exhaustiveness: List[str] = None
    num_modes: List[str] = None
    cnn_scoring: List[str] = None
    docking_mode: List[str] = None
    n_workers: List[str] = None
    use_gpu: List[str] = None
    cache_dir: List[str] = None
    timeout_sec: List[str] = None


@add_tag("__component")
class GninaDock:
    """
    REINVENT4 scoring component: on-the-fly Gnina molecular docking.

    Returns TWO endpoints:
      [0] Raw affinity (kcal/mol)  →  'GninaAffinity'      (weight > 0, use reverse_sigmoid transform)
      [1] Raw affinity (kcal/mol)  →  'GninaAffinity_raw'   (weight=0, logging only)

    More negative affinity = stronger binding. Use reverse_sigmoid transform
    with high=-6, low=-12 so better binders get higher reward.
    """

    def __init__(self, params: Parameters):
        self.smiles_type = "rdkit_smiles"
        self.number_of_endpoints = 2

        p = params
        self.config = GninaConfig(
            receptor_path=p.receptor_path[0],
            gnina_executable=(p.gnina_executable or ["gnina"])[0],
            center_x=_parse_float((p.center_x or [None])[0]),
            center_y=_parse_float((p.center_y or [None])[0]),
            center_z=_parse_float((p.center_z or [None])[0]),
            size_x=float((p.size_x or ["20.0"])[0]),
            size_y=float((p.size_y or ["20.0"])[0]),
            size_z=float((p.size_z or ["20.0"])[0]),
            autobox_ligand=(p.autobox_ligand or [None])[0],
            autobox_add=float((p.autobox_add or ["4.0"])[0]),
            exhaustiveness=int((p.exhaustiveness or ["4"])[0]),
            num_modes=int((p.num_modes or ["1"])[0]),
            cnn_scoring=(p.cnn_scoring or ["rescore"])[0],
            docking_mode=(p.docking_mode or ["full"])[0],
            n_workers=int((p.n_workers or ["4"])[0]),
            use_gpu=_parse_bool((p.use_gpu or ["true"])[0]),
            cache_dir=(p.cache_dir or [None])[0],
            timeout_sec=int((p.timeout_sec or ["120"])[0]),
        )
        self.config.validate()
        self._docker = GninaDocker(self.config)
        logger.info(
            f"[GninaDock] Initialized: receptor={self.config.receptor_path}, "
            f"mode={self.config.docking_mode}, workers={self.config.n_workers}, "
            f"gpu={self.config.use_gpu}"
        )

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> ComponentResults:
        n = len(smilies)
        scores_affinity = np.full(n, np.nan, dtype=np.float32)
        scores_cnn = np.full(n, np.nan, dtype=np.float32)

        try:
            results = self._docker.dock_batch(smilies)
            n_ok = 0
            for i, res in enumerate(results):
                if res.success and np.isfinite(res.affinity):
                    scores_affinity[i] = res.affinity
                    scores_cnn[i] = res.cnn_score
                    n_ok += 1

            valid = scores_affinity[np.isfinite(scores_affinity)]
            if len(valid) > 0:
                logger.info(
                    f"[GninaDock] Batch: {n} | docked={n_ok} | "
                    f"mean_affinity={valid.mean():.2f} | best={valid.min():.2f} kcal/mol"
                )
            else:
                logger.warning(f"[GninaDock] ALL {n} molecules failed docking.")

            return ComponentResults(
                [scores_affinity, scores_affinity.copy()],
                metadata={"cnn_score": scores_cnn.tolist()},
            )

        except Exception as exc:
            logger.error(f"[GninaDock] Error: {exc}\n{traceback.format_exc()}")
            return ComponentResults([np.full(n, np.nan, dtype=np.float32)])
