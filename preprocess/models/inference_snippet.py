
# ── Inference snippet (paste into your notebook) ──────────────
import joblib, pickle, numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdMorganFingerprint

# Load the single Neural Network model
nn_model = joblib.load("models/NN_model.joblib")

with open("data_splits/physchem_scaler.pkl","rb") as f:
    scaler = pickle.load(f)

with open("data_splits/feature_cols.txt") as f:
    FEATURE_COLS = [l.strip() for l in f]

def predict_pic50(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    rows = []
    for mol in mols:
        physchem = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
            Descriptors.TPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
            mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcFractionCSP3(mol),
            len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ]
        fp = rdMorganFingerprint.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        rows.append(physchem + list(map(int, fp.ToBitString())))

    import pandas as pd
    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    PHYSCHEM = ["mw","logp","hbd","hba","tpsa","rot_bonds",
                "rings","arom_rings","heavy_atoms","frac_csp3","stereo"]
    df[PHYSCHEM] = scaler.transform(df[PHYSCHEM])
    X = df[FEATURE_COLS].values.astype("float32")
    
    # Return the NN prediction
    return nn_model.predict(X)
