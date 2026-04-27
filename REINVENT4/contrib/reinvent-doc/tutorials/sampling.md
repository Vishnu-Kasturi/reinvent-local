# Tutorial: Sampling

Sampling generates molecules from a prior (or a previously trained agent) without any training. It is the quickest way to inspect what a model produces and to verify your setup before running TL or RL.

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `model_file` | Path to the model file (`.prior` or `.chkpt`). |
| `smiles_file` | Path to the input SMILES file (required for LibInvent, LinkInvent, and Mol2Mol). |
| `output_file` | Path to the output CSV file where results will be saved. |
| `num_smiles` | Number of SMILES to generate. For generators with input (LibInvent, LinkInvent, Mol2Mol), this is the number generated *per input SMILES*. |
| `unique_molecules` | Remove duplicates and canonicalize SMILES before writing output. |
| `randomize_smiles` | Randomly shuffle atom order in the input SMILES before passing to the model. Improves diversity, especially for Mol2Mol. |
| `sample_strategy` | `multinomial` (default, stochastic) or `beamsearch` (deterministic, always unique). Mol2Mol only. |
| `temperature` | Controls randomness in multinomial sampling. `< 1` more deterministic, `> 1` more random. Default `1.0`. Mol2Mol only. |



## Configuration

All four generators share the same `run_type`. Only the `[parameters]` block differs.

### Reinvent — de novo, no input required

```toml
run_type = "sampling"
device = "cuda:0"  # or "cpu"

[parameters]
model_file = "priors/reinvent.prior"
output_file = "sampling.csv"
num_smiles = 100
unique_molecules = true
randomize_smiles = true
```

### LibInvent — decorate a scaffold

`smiles_file` contains one scaffold SMILES per line with attachment points marked as `[*:0]`, `[*:1]`, etc. Example: [doc/data/scaffolds.smi](../data/scaffolds.smi).

```toml
run_type = "sampling"
device = "cuda:0"

[parameters]
model_file = "priors/libinvent.prior"
smiles_file = "doc/data/scaffolds.smi"
output_file = "sampling.csv"
num_smiles = 100
unique_molecules = true
randomize_smiles = true
```

### LinkInvent — link two fragments

`smiles_file` contains two warhead SMILES per line separated by `|`, each with one attachment point `*`. Example: [doc/data/warheads.smi](../data/warheads.smi).

```toml
run_type = "sampling"
device = "cuda:0"

[parameters]
model_file = "priors/linkinvent.prior"
smiles_file = "doc/data/warheads.smi"
output_file = "sampling.csv"
num_smiles = 100
unique_molecules = true
randomize_smiles = true
```

### Mol2Mol — generate analogues

`smiles_file` contains one reference SMILES per line (optional name in the second column). Example: [doc/data/mol2mol.smi](../data/mol2mol.smi). Beam search is deterministic; multinomial sampling adds randomness via `temperature`.

```toml
run_type = "sampling"
device = "cuda:0"

[parameters]
model_file = "priors/mol2mol_medium_similarity.prior"
smiles_file = "doc/data/mol2mol.smi"
sample_strategy = "multinomial"  # or "beamsearch"
temperature = 1.0
output_file = "sampling.csv"
num_smiles = 100
unique_molecules = true
randomize_smiles = true
```



## Running

```bash
reinvent sampling.toml
```

## Output

The output CSV contains one row per generated molecule:

| Column | Description |
|--------|-------------|
| `SMILES` | Generated molecule in canonical SMILES |
| `Input_SMILES` | Seed SMILES (LibInvent, LinkInvent, Mol2Mol only) |
| `NLL` | Negative log-likelihood — lower means the model considers this molecule more probable under the prior |

## Example Input Files

Example input files for conditional generators are provided in `doc/data/`:

- [scaffolds.smi](../data/scaffolds.smi) — LibInvent: azanaphthalene scaffolds with two attachment points
- [warheads.smi](../data/warheads.smi) — LinkInvent: three warhead pairs
- [mol2mol.smi](../data/mol2mol.smi) — Mol2Mol: two ChEMBL compounds and celecoxib

## What to Check

- **Valid SMILES rate**: if many rows are empty or invalid, the prior may not match the generator type.
- **NLL distribution**: a narrow NLL range means low diversity; consider increasing `temperature` or disabling `unique_molecules` temporarily to inspect.
- **Chemical diversity**: sample output is a good sanity check before committing to a TL or RL run.
