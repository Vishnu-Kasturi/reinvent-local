import pandas as pd
from rdkit import Chem
import os
import random

def process_custom_data(input_csv, train_smi, val_smi):
    df = pd.read_csv(input_csv, sep=';')
    
    valid_smiles = []
    
    for smi in df['Smiles'].dropna():
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            # 1. Remove stereochemistry
            Chem.RemoveStereochemistry(mol)
            
            # 2. Keep only the largest fragment (desalt)
            frags = Chem.GetMolFrags(mol, asMols=True)
            if frags:
                largest_frag = max(frags, key=lambda m: m.GetNumHeavyAtoms())
                canon = Chem.MolToSmiles(largest_frag, isomericSmiles=False)
                
                if not '.' in canon and not '[Na+]' in canon:
                    valid_smiles.append(canon)
                    
    # Remove duplicates
    valid_smiles = list(set(valid_smiles))
    random.shuffle(valid_smiles)
    
    # 80/20 split
    split_idx = int(len(valid_smiles) * 0.8)
    train_data = valid_smiles[:split_idx]
    val_data = valid_smiles[split_idx:]
    
    with open(train_smi, 'w') as f:
        for smi in train_data:
            f.write(f"{smi}\n")
            
    with open(val_smi, 'w') as f:
        for smi in val_data:
            f.write(f"{smi}\n")
            
    return len(train_data), len(val_data)

input_csv = 'REINVENT4/custom_data/data.csv'
train_smi = 'REINVENT4/custom_data/custom_train.smi'
val_smi = 'REINVENT4/custom_data/custom_val.smi'

t, v = process_custom_data(input_csv, train_smi, val_smi)
print(f"Custom Data - Train: {t}, Val: {v}")
