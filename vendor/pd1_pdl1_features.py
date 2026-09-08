"""
PD1-PDL1 feature extraction (vendored copy for offline vendor/ scripts).

Matches REINVENT4/reinvent_plugins/components/pd1_pdl1_features.py exactly.
"""
from __future__ import annotations

import math
import os
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, QED, rdMolDescriptors

_RDKIT_EXCLUDE = {
    "Ipc",
    "BCUT2D_MWHI", "BCUT2D_MWLOW",
    "BCUT2D_CHGHI", "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",  "BCUT2D_MRLOW",
}

_ALL_RDKIT = [(name, fn) for name, fn in Descriptors.descList if name not in _RDKIT_EXCLUDE]
RDKIT_DESC_LIST = _ALL_RDKIT[:200]

ECFP4_NBITS = 2048
ECFP4_RADIUS = 2
MACCS_NBITS = 167
PHYSCHEM_DIM = 12
EXPECTED_FEATURE_DIM = 200 + ECFP4_NBITS + MACCS_NBITS + PHYSCHEM_DIM


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


def _physchem(mol) -> np.ndarray:
    try:
        qed_val = QED.qed(mol)
    except Exception:
        qed_val = 0.0
    vals = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcFractionCSP3(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        qed_val,
    ]
    return np.array(vals, dtype=np.float32)


_SCALERS: dict[str, object] = {}


def get_scaler(scaler_path: str):
    if not scaler_path:
        return None
    if scaler_path not in _SCALERS:
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                _SCALERS[scaler_path] = pickle.load(f)
        else:
            return None
    return _SCALERS[scaler_path]


def compute_features(smiles_list: list[str], scaler_path: str):
    valid_mask = []
    rdkit_list, ecfp4_list, maccs_list, phys_list = [], [], [], []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            valid_mask.append(False)
            rdkit_list.append(np.zeros(200, dtype=np.float32))
            ecfp4_list.append(np.zeros(ECFP4_NBITS, dtype=np.float32))
            maccs_list.append(np.zeros(MACCS_NBITS, dtype=np.float32))
            phys_list.append(np.zeros(PHYSCHEM_DIM, dtype=np.float32))
            continue

        valid_mask.append(True)
        rdkit_list.append(_rdkit_descriptors(mol))
        ecfp4_list.append(_ecfp4(mol))
        maccs_list.append(_maccs(mol))
        phys_list.append(_physchem(mol))

    rdkit_arr = np.stack(rdkit_list)
    ecfp4_arr = np.stack(ecfp4_list)
    maccs_arr = np.stack(maccs_list)
    phys_arr = np.stack(phys_list)
    X = np.concatenate([rdkit_arr, ecfp4_arr, maccs_arr, phys_arr], axis=1)

    scaler = get_scaler(scaler_path)
    if scaler:
        n_features_scaler = scaler.mean_.shape[0] if hasattr(scaler, "mean_") else 200
        if n_features_scaler == 200:
            X[:, :200] = scaler.transform(X[:, :200]).astype(np.float32)
        else:
            cont_idx = list(range(200)) + list(range(200 + ECFP4_NBITS + MACCS_NBITS, EXPECTED_FEATURE_DIM))
            X[:, cont_idx] = scaler.transform(X[:, cont_idx]).astype(np.float32)

    for i, valid in enumerate(valid_mask):
        if not valid:
            X[i] = 0.0

    return X, valid_mask
