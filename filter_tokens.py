import os
import sys

# Add REINVENT4 to path so we can import its tokenization
sys.path.append(os.path.abspath('REINVENT4'))

from reinvent.utils.config_parse import tokenize_smiles, validate_tokens

allowed_tokens = {'[S+]', '%10', '#', '[n+]', 'N', '7', 'Br', '4', '8', 's', '=', '[O-]', 'F', '$', '3', 'n', ')', '[N+]', 'o', 'O', 'C', 'S', '6', '(', '2', '1', '-', '[N-]', '5', 'c', '9', '^', 'Cl', '[nH]'}

def filter_smi(filepath):
    with open(filepath, 'r') as f:
        smiles = [line.strip() for line in f if line.strip()]
        
    valid_smiles = []
    invalid_count = 0
    
    for smi in smiles:
        try:
            tokens = set(tokenize_smiles(smi))
            # The allowed_tokens set provided in the error message doesn't contain '^' and '$' initially, 
            # but they are start/end tokens. REINVENT's validate_tokens adds them.
            # We can just use REINVENT's function:
            validate_tokens([smi], allowed_tokens)
            valid_smiles.append(smi)
        except ValueError:
            invalid_count += 1
            
    with open(filepath, 'w') as f:
        for smi in valid_smiles:
            f.write(smi + '\n')
            
    return len(valid_smiles), invalid_count

train_valid, train_invalid = filter_smi('REINVENT4/custom_data/custom_train.smi')
val_valid, val_invalid = filter_smi('REINVENT4/custom_data/custom_val.smi')

print(f"Train: kept {train_valid}, removed {train_invalid}")
print(f"Val: kept {val_valid}, removed {val_invalid}")
