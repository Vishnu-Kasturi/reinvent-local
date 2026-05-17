import pandas as pd
from rdkit import Chem
import os

def process_smiles(input_csv, output_smi):
    df = pd.read_csv(input_csv)
    if 'canon_smiles' in df.columns:
        smiles_col = 'canon_smiles'
    elif 'smiles' in df.columns:
        smiles_col = 'smiles'
    else:
        smiles_col = df.columns[0]
        
    valid_smiles = []
    invalid_count = 0
    
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            canon = Chem.MolToSmiles(mol)
            valid_smiles.append(canon)
        else:
            invalid_count += 1
            
    with open(output_smi, 'w') as f:
        for smi in valid_smiles:
            f.write(f"{smi}\n")
            
    return len(valid_smiles), invalid_count

train_csv = 'preprocess/data_splits/train.csv'
val_csv = 'preprocess/data_splits/val.csv'
train_smi = 'preprocess/jak2_train.smi'
val_smi = 'preprocess/jak2_val.smi'

train_valid, train_invalid = process_smiles(train_csv, train_smi)
val_valid, val_invalid = process_smiles(val_csv, val_smi)

print(f"Number of train molecules: {train_valid}")
print(f"Number of val molecules: {val_valid}")
print(f"Number of invalid molecules removed: {train_invalid + val_invalid}")
