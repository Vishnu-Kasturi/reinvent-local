"""
sascorer.py — SA score (Ertl & Schuffenhauer).

Vendored for offline use. Set FPSCORES_PATH to your fpscores.pkl.gz location.
"""

from __future__ import annotations

import gzip
import math
import pickle
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

_fscores = None

# Edit: path to fpscores.pkl.gz (without .pkl.gz suffix)
FPSCORES_PATH = Path(__file__).resolve().parent / "fpscores"


def readFragmentScores(name: str | Path = FPSCORES_PATH) -> None:
    global _fscores
    path = Path(name)
    gz = path if path.suffix == ".gz" else Path(f"{path}.pkl.gz")
    if not gz.is_file():
        raise FileNotFoundError(
            f"SA fpscores not found: {gz}\n"
            "Download or copy fpscores.pkl.gz and set FPSCORES_PATH in sascorer.py"
        )
    data = pickle.load(gzip.open(gz, "rb"))
    out_dict: dict = {}
    for i in data:
        for j in range(1, len(i)):
            out_dict[i[j]] = float(i[0])
    _fscores = out_dict


def numBridgeheadsAndSpiro(mol, ri=None):
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return n_bridgehead, n_spiro


def calculateScore(m: Chem.Mol) -> float:
    if _fscores is None:
        readFragmentScores()

    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bit_id, v in fps.items():
        nf += v
        score1 += _fscores.get(bit_id, -4) * v
    score1 /= nf

    n_atoms = m.GetNumAtoms()
    n_chiral = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    n_bridgeheads, n_spiro = numBridgeheadsAndSpiro(m, ri)
    n_macrocycles = sum(1 for x in ri.AtomRings() if len(x) > 8)

    size_penalty = n_atoms**1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridgeheads + 1)
    macrocycle_penalty = math.log10(2) if n_macrocycles > 0 else 0.0

    score2 = 0.0 - size_penalty - stereo_penalty - spiro_penalty - bridge_penalty - macrocycle_penalty

    score3 = 0.0
    if n_atoms > len(fps):
        score3 = math.log(float(n_atoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    lo, hi = -4.0, 2.5
    sascore = 11.0 - (sascore - lo + 1) / (hi - lo) * 9.0
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return float(sascore)
