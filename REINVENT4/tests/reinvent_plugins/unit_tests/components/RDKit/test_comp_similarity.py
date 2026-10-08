import os
import tempfile

import pytest

import numpy as np

from reinvent_plugins.components.RDKit.comp_similarity import (
    Parameters,
    TanimotoDistance,
    TanimotoSimilarity,
)


@pytest.mark.parametrize(
    "smiles, radius, use_counts, use_features, expected_results",
    [
        (
            ["c1ccccc1", "Cc1ccccc1"],
            2,
            False,
            False,
            [1.0, 0.27272727, 0.27272727, 1.0],
        ),
        (
            ["c1ccccc1", "Cc1ccccc1"],
            2,
            True,
            False,
            [1.0, 0.31034483, 0.31034483, 1.0],
        ),
        (
            ["c1ccccc1", "Cc1ccccc1"],
            2,
            False,
            True,
            [1.0, 0.375, 0.375, 1.0],
        ),
        (
            ["c1ccccc1", "Cc1ccccc1"],
            2,
            True,
            True,
            [1.0, 0.58333333, 0.58333333, 1.0],
        ),
    ],
)
def test_comp_similarity(smiles, radius, use_counts, use_features, expected_results):
    params = Parameters(
        radius=[radius],
        use_counts=[use_counts],
        use_features=[use_features],
        smiles=[smiles],
    )
    td = TanimotoDistance(params)
    results = np.concatenate(td(smiles).scores)
    assert np.allclose(results, expected_results)


def test_comp_similarity_smiles_file_only():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".smi") as handle:
        handle.write("c1ccccc1\nCc1ccccc1\n")
        smiles_file = handle.name

    try:
        params = Parameters(
            radius=[2],
            use_counts=[False],
            use_features=[False],
            smiles_file=[smiles_file],
        )
        component = TanimotoSimilarity(params)
        results = component(["c1ccccc1"]).scores[0]
        assert np.isclose(results[0], 1.0)
    finally:
        os.unlink(smiles_file)
