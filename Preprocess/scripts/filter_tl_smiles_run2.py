"""
Filter TL SMILES for run2: removes molecules with SMILES > 120 chars or > 50 heavy atoms.
These are exactly the molecules REINVENT's default tokenizer rejects with "invalid" warnings.
"""
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

def is_valid(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return False
    if len(smi) > 120:     # REINVENT tokenizer max token length guard
        return False
    if mol.GetNumAtoms() > 50:   # Very large molecules tokenize poorly
        return False
    return True

for split in ['train', 'val']:
    in_path = f'data/pd1_pdl1_TL_{split}.smi'
    out_path = f'data/pd1_pdl1_TL_run2_{split}.smi'
    kept, dropped = 0, 0
    with open(in_path) as fin, open(out_path, 'w') as fout:
        for line in fin:
            smi = line.strip()
            if not smi:
                continue
            if is_valid(smi):
                fout.write(smi + '\n')
                kept += 1
            else:
                dropped += 1
    print(f"[{split}] Kept: {kept} | Dropped: {dropped} → {out_path}")
