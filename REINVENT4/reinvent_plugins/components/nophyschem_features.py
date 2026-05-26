"""
Shared feature extraction (2415-dim, NO physchem) for:
  - JAK2 pIC50 scoring component
  - PD1-PDL1 pIC50 scoring component (final_acc model, no physchem)

Feature order (2415 dims):
  200 RDKit 2D descriptors (scaled)
  2048 ECFP4 fingerprint bits
  167 MACCS keys

Matches xgboost_2.py with --no_physchem, used to train the final_acc models.
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

_RDKIT_EXCLUDE = {
    "Ipc",
    "BCUT2D_MWHI", "BCUT2D_MWLOW",
    "BCUT2D_CHGHI", "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",  "BCUT2D_MRLOW",
}
_ALL_RDKIT = [(name, fn) for name, fn in Descriptors.descList
              if name not in _RDKIT_EXCLUDE]
RDKIT_DESC_LIST = _ALL_RDKIT[:200]

ECFP4_NBITS  = 2048
ECFP4_RADIUS = 2
MACCS_NBITS  = 167

# 200 + 2048 + 167 = 2415 (no physchem)
EXPECTED_FEATURE_DIM = 200 + ECFP4_NBITS + MACCS_NBITS


def _rdkit_descriptors(mol) -> np.ndarray:
    vals = []
    for _, fn in RDKIT_DESC_LIST:
        try:
            v = fn(mol)
            vals.append(0.0 if (v is None or math.isnan(v) or math.isinf(v)) else float(v))
        except Exception:
            vals.append(0.0)
    return np.array(vals, dtype=np.float32)


def _ecfp4(mol) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, ECFP4_RADIUS, nBits=ECFP4_NBITS)
    return np.array(fp, dtype=np.float32)


def _maccs(mol) -> np.ndarray:
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.float32)


def compute_features(smiles_list: list[str], scaler_path: str):
    """
    Compute the 2415-dim feature matrix for a list of SMILES.

    Feature order: 200 RDKit | 2048 ECFP4 | 167 MACCS
    Scaler applies to the 200 RDKit continuous descriptors (indices 0:200).

    Returns
    -------
    X : np.ndarray, shape (n, 2415)
    valid_mask : list[bool]
    """
    valid_mask = []
    rdkit_list, ecfp4_list, maccs_list = [], [], []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            valid_mask.append(False)
            rdkit_list.append(np.zeros(200, dtype=np.float32))
            ecfp4_list.append(np.zeros(ECFP4_NBITS, dtype=np.float32))
            maccs_list.append(np.zeros(MACCS_NBITS, dtype=np.float32))
            continue
        valid_mask.append(True)
        rdkit_list.append(_rdkit_descriptors(mol))
        ecfp4_list.append(_ecfp4(mol))
        maccs_list.append(_maccs(mol))

    rdkit_arr = np.stack(rdkit_list)   # (n, 200)
    ecfp4_arr = np.stack(ecfp4_list)   # (n, 2048)
    maccs_arr = np.stack(maccs_list)   # (n, 167)

    X = np.concatenate([rdkit_arr, ecfp4_arr, maccs_arr], axis=1)  # (n, 2415)

    # Scale RDKit continuous descriptors (indices 0:200)
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X[:, :200] = scaler.transform(X[:, :200]).astype(np.float32)

    # Zero out invalid rows
    for i, valid in enumerate(valid_mask):
        if not valid:
            X[i] = 0.0

    return X, valid_mask
