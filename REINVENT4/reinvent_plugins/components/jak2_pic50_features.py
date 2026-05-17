import os
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, Descriptors, rdMolDescriptors

_EXCLUDE = {
    "Ipc",
    "BCUT2D_MWHI","BCUT2D_MWLOW",
    "BCUT2D_CHGHI","BCUT2D_CHGLO",
    "BCUT2D_LOGPHI","BCUT2D_LOGPLOW",
    "BCUT2D_MRHI","BCUT2D_MRLOW",
}

RDKIT_DESC_FUNCS = [
    (name, func)
    for name, func in Descriptors.descList
    if name not in _EXCLUDE
]

def rdkit_descriptors(mol):
    vals = []
    for _, func in RDKIT_DESC_FUNCS:
        try:
            v = func(mol)
            vals.append(float(v) if (v is not None and np.isfinite(float(v))) else 0.0)
        except Exception:
            vals.append(0.0)
    return np.array(vals, dtype=np.float32)

def compute_features(smiles_list: list[str], scaler_path: str, physchem_scaler_path: str = None):
    valid_mask = []
    maccs_list, ecfp4_list, ecfp6_list, rdkit_list, physchem_list = [], [], [], [], []

    # Try loading RDKit descriptors scaler
    desc_scaler = None
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            desc_scaler = pickle.load(f)
            
    physchem_scaler = None
    if physchem_scaler_path and os.path.exists(physchem_scaler_path):
        with open(physchem_scaler_path, "rb") as f:
            physchem_scaler = pickle.load(f)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            valid_mask.append(False)
            physchem_list.append(np.zeros(11, dtype=np.float32))
            maccs_list.append(np.zeros(167, dtype=np.float32))
            ecfp4_list.append(np.zeros(2048, dtype=np.float32))
            ecfp6_list.append(np.zeros(2048, dtype=np.float32))
            rdkit_list.append(np.zeros(len(RDKIT_DESC_FUNCS), dtype=np.float32))
            continue
            
        valid_mask.append(True)
        
        # 11 Physchem
        physchem = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
            Descriptors.TPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
            mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcFractionCSP3(mol),
            len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ]
        physchem_list.append(physchem)
        
        # Fingerprints
        maccs_list.append(np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32))
        ecfp4_list.append(np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048), dtype=np.float32))
        ecfp6_list.append(np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048), dtype=np.float32))
            
        # RDKit 2D
        rdkit_list.append(rdkit_descriptors(mol))

    physchem_arr = np.array(physchem_list, dtype=np.float32)
    if physchem_scaler is not None and np.any(valid_mask):
        # We only want to scale valid physchems or just use transform
        physchem_arr = physchem_scaler.transform(physchem_arr).astype(np.float32)
        
    maccs_arr = np.stack(maccs_list)
    ecfp4_arr = np.stack(ecfp4_list)
    ecfp6_arr = np.stack(ecfp6_list)
    rdkit_arr = np.stack(rdkit_list)
    
    if desc_scaler is not None:
        rdkit_arr = desc_scaler.transform(rdkit_arr).astype(np.float32)
        
    # Set invalid rows to entirely zero to be safe
    for i, valid in enumerate(valid_mask):
        if not valid:
            physchem_arr[i] = 0.0
            rdkit_arr[i] = 0.0

    X = np.concatenate([physchem_arr, maccs_arr, ecfp4_arr, ecfp6_arr, rdkit_arr], axis=1)
    return X, valid_mask
