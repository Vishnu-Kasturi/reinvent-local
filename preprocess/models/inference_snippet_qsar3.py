
# qsar3 inference snippet
import joblib, pickle, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import RDKFingerprint

models = {n: joblib.load(f"models/{n}.joblib") for n in
          ["RF_balanced_qsar3","RF_extreme_qsar3",
           "XGB_conservative_qsar3","XGB_aggressive_qsar3"]}
meta = joblib.load("models/meta_ridge_qsar3.joblib")
with open("models/physchem_scaler_qsar3.pkl","rb") as f:
    scaler = pickle.load(f)

PHYSCHEM_COLS = ["mw","exact_mw","logp","hbd","hba","tpsa","rot_bonds",
                 "rings","arom_rings","heavy_atoms","frac_csp3","stereo"]

def predict_pic50(smiles_list):
    import pandas as pd
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        pc = [Descriptors.MolWt(mol), Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol),
              rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
              Descriptors.TPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
              rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
              mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcFractionCSP3(mol),
              len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))]
        ecfp4 = list(map(int, AllChem.GetMorganFingerprintAsBitVect(mol,2,2048).ToBitString()))
        rdkfp = list(map(int, RDKFingerprint(mol, fpSize=2048).ToBitString()))
        rows.append(pc + ecfp4 + rdkfp)
    ALL_COLS = PHYSCHEM_COLS + [f"ecfp4_{i}" for i in range(2048)] + [f"rdkfp_{i}" for i in range(2048)]
    df = pd.DataFrame(rows, columns=ALL_COLS)
    df[PHYSCHEM_COLS] = scaler.transform(df[PHYSCHEM_COLS])
    X = df.values.astype("float32")
    stack = np.column_stack([m.predict(X) for m in models.values()])
    return meta.predict(stack)
