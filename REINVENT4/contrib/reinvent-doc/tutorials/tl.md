# Tutorial: Transfer Learning

Transfer Learning (TL) retrains a prior on a focused SMILES dataset, producing an agent biased toward that chemical series. The output model can be used directly for sampling or as the starting point for a reinforcement learning run.

## When to Use TL

- You have a set of known active molecules for a target and want the generator to produce similar structures.
- You want to reduce the RL exploration burden by starting from a focused agent rather than a broad prior.
- You don't have a well-defined scoring function for RL but want to bias generation toward a specific chemical series.

## Input Data

TL reads SMILES from a plain text file, one per line. The first column is used; a second column (e.g. compound name or ID) is ignored.

Split your data into training and validation sets before running — REINVENT4 does not do this automatically. A typical split is 80/20.


## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `input_model_file` | Prior or checkpoint to start from |
| `smiles_file` | Training SMILES (first column used) |
| `validation_smiles_file` | Validation SMILES for monitoring overfitting |
| `output_model_file` | Path for the resulting agent (`.model`) |
| `num_epochs` | Number of training epochs; start with 50–100 and adjust based on validation loss |
| `save_every_n_epochs` | Frequency of intermediate checkpoints; useful for resuming or selecting the best epoch |
| `batch_size` | Number of SMILES per gradient step; smaller batches give noisier but faster updates |
| `tb_logdir` | Directory for TensorBoard logs (training and validation loss curves) |

## Configuration

### Reinvent

```toml
run_type = "transfer_learning"
device = "cuda:0"          # or "cpu"
tb_logdir = "tb_TL"        # TensorBoard log directory

[parameters]
input_model_file = "priors/reinvent.prior"
smiles_file = "doc/data/tl_reinvent.smi"
validation_smiles_file = "doc/data/tl_reinvent_val.smi"
output_model_file = "tl_agent.model"

num_epochs = 50
save_every_n_epochs = 10   # write a checkpoint every N epochs
batch_size = 50
```

### Mol2Mol

Mol2Mol is a conditional generator: at inference time it takes an input molecule and generates similar ones. TL therefore trains on (source, target) pairs rather than individual SMILES. The `pairs` block controls how these pairs are constructed from your dataset using Tanimoto similarity.

```toml
run_type = "transfer_learning"
device = "cuda:0"
tb_logdir = "tb_TL"

[parameters]
input_model_file = "priors/mol2mol_medium_similarity.prior"
smiles_file = "doc/data/tl_reinvent.smi"
validation_smiles_file = "doc/data/tl_reinvent_val.smi"
output_model_file = "tl_mol2mol.model"

num_epochs = 50
save_every_n_epochs = 10
batch_size = 50

pairs.type = "tanimoto"
pairs.lower_threshold = 0.7   # only pair molecules with Tanimoto >= 0.7
pairs.upper_threshold = 1.0   # set < 1.0 to exclude identical molecules from pairing
pairs.min_cardinality = 1     # discard source molecules that have fewer than N valid targets
pairs.max_cardinality = 199   # discard source molecules that have more than N valid targets (prevents very promiscuous sources from dominating training)
```

> **Note:** This tutorial covers Reinvent and Mol2Mol only. LibInvent and LinkInvent use a two-column SMILES format and TL only affects the learned part (R-groups or linker) rather than the constrained scaffold — making it of limited practical value in most cases.


## Output

| File | Description |
|------|-------------|
| `output_model_file` | Final trained agent — use as `model_file` in sampling or as `agent_file` in RL |
| Checkpoints (`*.chkpt`) | Intermediate snapshots saved every `save_every_n_epochs` |
| TensorBoard logs | Loss curves for training and validation (written to `tb_logdir`) |

## What to Check

- **Training and validation loss converge**: a widening gap means overfitting — stop earlier or use a checkpoint with lower validation loss.
- **Valid SMILES rate stays high**: sample from the trained agent; a drop below ~90% means the model drifted too far from the prior.
- **Sampled structures resemble training set**: if outputs are too dissimilar or degenerate, reduce `num_epochs`.
