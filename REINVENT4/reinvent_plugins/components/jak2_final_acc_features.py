"""
Feature extraction for JAK2 pIC50 final_acc model.

Exactly matches jak_pic50.py extract_classical():
  Feature order (3451 dims):
    12  physchem    (MW, LogP, HBD, HBA, TPSA, RotBonds, Rings, AromaticRings,
                     HeavyAtoms, FracCSP3, Stereocenters, MolMR)
   167  MACCS keys
  2048  ECFP4 (Morgan r=2, 2048 bits)
  1024  RDKit Topological (maxPath=7, 1024 bits)
   200  RDKit 2D descriptors

The scaler (StandardScaler) covers the continuous columns:
  indices 0:12   (physchem)
  indices 12+167+2048+1024 : 3451  (RDKit 200 descriptors)
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors

_RDKIT_EXCLUDE = {
    "Ipc",
    "BCUT2D_MWHI", "BCUT2D_MWLOW",
    "BCUT2D_CHGHI", "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",  "BCUT2D_MRLOW",
}
_ALL_RDKIT = [(name, fn) for name, fn in Descriptors.descList if name not in _RDKIT_EXCLUDE]
RDKIT_DESC_LIST = _ALL_RDKIT[:200]

# 12 + 167 + 2048 + 1024 + 200 = 3451
EXPECTED_FEATURE_DIM = 12 + 167 + 2048 + 1024 + 200


def _physchem(mol) -> np.ndarray:
    """12 physicochemical features — matches jak_pic50.py compute_physchem()."""
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcFractionCSP3(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        Descriptors.MolMR(mol),
    ], dtype=np.float32)


def _rdkit_descriptors(mol) -> np.ndarray:
    vals = []
    for _, fn in RDKIT_DESC_LIST:
        try:
            v = fn(mol)
            vals.append(0.0 if (v is None or math.isnan(v) or math.isinf(v)) else float(v))
        except Exception:
            vals.append(0.0)
    return np.array(vals, dtype=np.float32)


def compute_features(smiles_list: list[str], scaler_path: str, physchem_scaler_path: str = None):
    """
    Compute the 3451-dim feature matrix matching jak_pic50.py.

    The scaler loaded from scaler_path scales BOTH physchem (0:12) and
    RDKit 200 descriptors (3251:3451) together via cont_idx.
    """
    valid_mask = []
    phys_list, maccs_list, ecfp4_list, topo_list, rdk_list = [], [], [], [], []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            valid_mask.append(False)
            phys_list.append(np.zeros(12,    dtype=np.float32))
            maccs_list.append(np.zeros(167,  dtype=np.float32))
            ecfp4_list.append(np.zeros(2048, dtype=np.float32))
            topo_list.append(np.zeros(1024,  dtype=np.float32))
            rdk_list.append(np.zeros(200,    dtype=np.float32))
            continue

        valid_mask.append(True)
        phys_list.append(_physchem(mol))
        maccs_list.append(np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32))
        ecfp4_list.append(np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048), dtype=np.float32))
        topo_list.append(np.array(
            Chem.RDKFingerprint(mol, maxPath=7, fpSize=1024), dtype=np.float32))
        rdk_list.append(_rdkit_descriptors(mol))

    phys_arr  = np.stack(phys_list)   # (n, 12)
    maccs_arr = np.stack(maccs_list)  # (n, 167)
    ecfp4_arr = np.stack(ecfp4_list)  # (n, 2048)
    topo_arr  = np.stack(topo_list)   # (n, 1024)
    rdk_arr   = np.stack(rdk_list)    # (n, 200)

    X = np.concatenate([phys_arr, maccs_arr, ecfp4_arr, topo_arr, rdk_arr], axis=1)  # (n, 3451)

    # Apply scaler to continuous indices: physchem (0:12) + RDKit (3251:3451)
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        cont_idx = list(range(12)) + list(range(12 + 167 + 2048 + 1024, EXPECTED_FEATURE_DIM))
        X[:, cont_idx] = scaler.transform(X[:, cont_idx]).astype(np.float32)

    # Zero out invalid rows
    for i, valid in enumerate(valid_mask):
        if not valid:
            X[i] = 0.0

    return X, valid_mask
