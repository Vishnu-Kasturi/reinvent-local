"""
Custom REINVENT4 Scoring Component: Tanimoto Penalty vs Training Set
=====================================================================
Prevents reinforcement learning from copy-pasting/regurgitating the training set.

Returns TWO endpoints:
1. TanimotoPenalty:
     - 1.0 if max_similarity < threshold
     - scales linearly down to 0.0 if max_similarity >= threshold
2. MaxTanimoto (raw):
     - The actual maximum similarity value [0, 1] for logging.
"""
from __future__ import annotations

__all__ = ["TanimotoPenalty"]

import logging
import os
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from .component_results import ComponentResults
from .add_tag import add_tag

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    smiles_file: List[str]
    threshold: List[float]
    radius: List[int]


@add_tag("__component")
class TanimotoPenalty:
    """
    Scoring component that penalizes high similarity to a reference set of SMILES
    (e.g., the training set used for Transfer Learning).
    """

    def __init__(self, params: Parameters):
        self.smiles_file = params.smiles_file[0]
        self.threshold = float(params.threshold[0])
        self.radius = int(params.radius[0])

        if not os.path.exists(self.smiles_file):
            raise FileNotFoundError(
                f"[TanimotoPenalty] reference SMILES file not found: {self.smiles_file}"
            )

        logger.info(
            f"[TanimotoPenalty] Loading reference set from {self.smiles_file}..."
        )
        self.ref_fps = []
        with open(self.smiles_file, "r") as f:
            for line in f:
                smi = line.strip()
                if not smi:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius=self.radius, nBits=2048
                    )
                    self.ref_fps.append(fp)

        logger.info(
            f"[TanimotoPenalty] Loaded {len(self.ref_fps)} valid reference fingerprints. "
            f"Threshold={self.threshold:.2f}, Radius={self.radius}"
        )

    def __call__(self, smilies: List[str]) -> ComponentResults:
        n = len(smilies)
        scores_penalty = np.zeros(n, dtype=np.float32)
        scores_max_sim = np.zeros(n, dtype=np.float32)

        if not self.ref_fps:
            # If no reference, no penalty and 0 similarity
            return ComponentResults([np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32)])

        for i, smi in enumerate(smilies):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                scores_penalty[i] = np.nan
                scores_max_sim[i] = np.nan
                continue

            # Compute ECFP4 fingerprint for query
            query_fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.radius, nBits=2048
            )

            # Compute similarity to all reference compounds
            sims = DataStructs.BulkTanimotoSimilarity(query_fp, self.ref_fps)
            max_sim = float(max(sims))

            # Store raw similarity
            scores_max_sim[i] = max_sim

            # Calculate penalty
            if max_sim < self.threshold:
                scores_penalty[i] = 1.0
            else:
                # Linearly drop score from 1.0 (at threshold) to 0.0 (at 1.0 similarity)
                denom = 1.0 - self.threshold
                if denom <= 0:
                    scores_penalty[i] = 0.0
                else:
                    score = 1.0 - ((max_sim - self.threshold) / denom)
                    scores_penalty[i] = float(np.clip(score, 0.0, 1.0))

        return ComponentResults([scores_penalty, scores_max_sim])
