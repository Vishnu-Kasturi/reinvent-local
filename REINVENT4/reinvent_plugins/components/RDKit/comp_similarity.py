"""Tanimoto similarity and Jaccard distance"""

from __future__ import annotations

__all__ = ["TanimotoSimilarity", "TanimotoDistance"]

import os
import warnings
from typing import List, Optional

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from reinvent.chemistry import conversions
from reinvent.chemistry.similarity import calculate_tanimoto, calculate_tanimoto_batch
from ..component_results import ComponentResults
from ..add_tag import add_tag


def _load_smiles_from_file(smiles_file: str) -> List[str]:
    if not os.path.exists(smiles_file):
        raise FileNotFoundError(f"{__name__}: reference SMILES file not found: {smiles_file}")

    smilies = []
    with open(smiles_file, "r") as handle:
        for line in handle:
            smi = line.strip()
            if smi and not smi.startswith("#"):
                smilies.append(smi)

    if not smilies:
        raise ValueError(f"{__name__}: no SMILES found in {smiles_file}")

    return smilies


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    radius: List[int]
    use_counts: List[bool]
    use_features: List[bool]
    smiles: Optional[List[Optional[List[str]]]] = Field(default=None)
    smiles_file: Optional[List[Optional[str]]] = Field(default=None)


@add_tag("__component")
class TanimotoSimilarity:
    """Compute the Tanimoto similarity

    Scoring component to compute the Tanimoto similarity between the provided
    SMILES and the generated molecule.  Supports fingerprint radius, count
    fingerprints and the use of pharmacophore-like features (see
    https://doi.org/10.1002/(SICI)1097-0290(199824)61:1%3C47::AID-BIT9%3E3.0.CO;2-Z).
    """

    def __init__(self, params: Parameters):
        self.fp_params = []
        n_endpoints = len(params.radius)
        smiles_list = params.smiles if params.smiles is not None else [None] * n_endpoints
        smiles_file_list = (
            params.smiles_file if params.smiles_file is not None else [None] * n_endpoints
        )

        for smilies, smiles_file, radius, use_counts, use_features in zip(
            smiles_list,
            smiles_file_list,
            params.radius,
            params.use_counts,
            params.use_features,
        ):
            if smiles_file:
                smilies = _load_smiles_from_file(smiles_file)
            elif not smilies:
                raise ValueError(f"{__name__}: either smiles or smiles_file must be provided")

            fingerprints = conversions.smiles_to_fingerprints(
                smilies, radius=radius, use_counts=use_counts, use_features=use_features
            )

            if not fingerprints:
                raise ValueError(f"{__name__}: unable to convert any SMILES to fingerprints")

            use_max_similarity = bool(smiles_file) and len(fingerprints) > 1
            self.fp_params.append(
                (fingerprints, radius, use_counts, use_features, use_max_similarity)
            )

        self.number_of_endpoints = len(self.fp_params)

    def __call__(self, smilies: List[str]) -> np.array:
        scores = []

        for fingerprints, radius, use_counts, use_features, use_max_similarity in self.fp_params:
            query_fingerprints = conversions.smiles_to_fingerprints(
                smilies, radius=radius, use_counts=use_counts, use_features=use_features
            )

            if use_max_similarity:
                scores.append(calculate_tanimoto(query_fingerprints, fingerprints))
            elif len(fingerprints) == 1:
                scores.append(
                    calculate_tanimoto_batch(fingerprints[0], query_fingerprints)
                )
            else:
                scores.extend(
                    [
                        calculate_tanimoto_batch(fingerprint, query_fingerprints)
                        for fingerprint in fingerprints
                    ]
                )

        return ComponentResults(scores)


# for backward compatibility
@add_tag("__component")
class TanimotoDistance(TanimotoSimilarity):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TanimotoDistance is deprecated and will be removed in a future release. "
            "Please use TanimotoSimilarity instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
