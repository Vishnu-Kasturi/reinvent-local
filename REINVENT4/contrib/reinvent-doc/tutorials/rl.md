# Tutorial: Reinforcement Learning

Reinforcement Learning (RL) iteratively updates an agent to increase the likelihood of generating molecules that score well on a user-defined scoring function. In REINVENT4 all RL runs use `run_type = "staged_learning"`, even with a single stage.

This tutorial covers the **optimization setup** — how to configure the agent, learning strategy, diversity filter, and stages. Scoring function design (components, transforms, aggregation) is covered in the [Scoring tutorial](scoring.md).

See [Core Concepts](../core_concept/README.md) for background on priors, agents, scoring, diversity filter, inception, and curriculum learning.

## Key Parameters

### `[parameters]`

| Parameter | Description |
|-----------|-------------|
| `prior_file` | Fixed reference model — not updated during the run |
| `agent_file` | Starting model for the agent — typically the same as the prior, or a TL checkpoint |
| `summary_csv_prefix` | Prefix for output CSV files (one per stage) |
| `batch_size` | SMILES sampled per epoch; larger batches give more stable gradients |
| `unique_sequences` | Discard duplicate raw sequences within a batch |
| `randomize_smiles` | Shuffle atom order in input SMILES; improves diversity |

### `[learning_strategy]`

| Parameter | Description |
|-----------|-------------|
| `type` | Always `dap` (Difference between Augmented and Posterior) |
| `sigma` | Reward scaling. Default 128. Lower (e.g. 32) for conservative learning; higher (e.g. 256) for faster but riskier convergence |
| `rate` | Learning rate for Adam optimiser. Default `0.0001` |

### `[diversity_filter]`

| Parameter | Description |
|-----------|-------------|
| `type` | `IdenticalMurckoScaffold` (recommended), `IdenticalTopologicalScaffold`, `ScaffoldSimilarity`, `PenalizeSameSmiles` |
| `bucket_size` | Max molecules per scaffold bucket before penalisation kicks in |
| `minscore` | Only add molecules to memory if total score exceeds this threshold |
| `minsimilarity` | Minimum Tanimoto similarity for `ScaffoldSimilarity` type |
| `penalty_multiplier` | Score multiplier for penalised molecules (0–1); `PenalizeSameSmiles` only |

### `[[stage]]` 

| Parameter | Description |
|-----------|-------------|
| `chkpt_file` | Checkpoint written at end of stage; can be reused as `agent_file` |
| `termination` | Always `simple` |
| `max_score` | Advance to next stage (or stop) when mean batch score exceeds this |
| `min_steps` | Run at least this many epochs before checking `max_score` |
| `max_steps` | Hard limit; terminates the entire run if reached |

## Configuration Structure

```toml
run_type = "staged_learning"
device = "cuda:0"
tb_logdir = "tb_RL"

[parameters]          # generator and global settings
[learning_strategy]   # optimiser and sigma
[diversity_filter]    # optional: scaffold diversity
[inception]           # optional: experience replay (Reinvent only)

[[stage]]             # one or more stage blocks
[stage.scoring]
[[stage.scoring.component]]
...
```

## Single-Stage Example

```toml
run_type = "staged_learning"
device = "cuda:0"
tb_logdir = "tb_RL"

[parameters]
prior_file = "priors/reinvent.prior"
agent_file = "priors/reinvent.prior"   # or a TL checkpoint
summary_csv_prefix = "rl_run"
batch_size = 64
randomize_smiles = true
unique_sequences = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4

[[stage]]
chkpt_file = "stage1.chkpt"
termination = "simple"
max_score = 0.7
min_steps = 25
max_steps = 500

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Alerts"
params.smarts = [
    "[*;r{8-17}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]"
]

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 1.0

[[stage.scoring.component]]
[stage.scoring.component.MolecularWeight]
[[stage.scoring.component.MolecularWeight.endpoint]]
name = "MW"
weight = 1.0
transform.type = "double_sigmoid"
transform.low = 200.0
transform.high = 500.0
transform.coef_div = 500.0
transform.coef_si = 20.0
transform.coef_se = 20.0
```

## Multi-Stage Example (Curriculum Learning)

```toml
[[stage]]
chkpt_file = "stage1.chkpt"
termination = "simple"
max_score = 0.6
min_steps = 25
max_steps = 300

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Alerts"
params.smarts = ["[*;r{8-17}]", "[#8][#8]", "[#6;+]", "[#16][#16]"]

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 1.0


[[stage]]
chkpt_file = "stage2.chkpt"
termination = "simple"
max_score = 0.8
min_steps = 25
max_steps = 500

[stage.scoring]
type = "geometric_mean"
filename = "stage2_scoring.toml"  # load components from a separate file
filetype = "toml"
```

## Generator Variants

All four generators work with `staged_learning`. Conditional generators require `smiles_file` in `[parameters]`:

```toml
# LibInvent
prior_file = "priors/libinvent.prior"
agent_file = "priors/libinvent.prior"
smiles_file = "doc/data/scaffolds.smi"

# LinkInvent
prior_file = "priors/linkinvent.prior"
agent_file = "priors/linkinvent.prior"
smiles_file = "doc/data/warheads.smi"

# Mol2Mol
prior_file = "priors/mol2mol_scaffold_generic.prior"
agent_file = "priors/mol2mol_scaffold_generic.prior"
smiles_file = "doc/data/mol2mol.smi"
sample_strategy = "multinomial"
distance_threshold = 100
```

## Running

```bash
reinvent rl.toml
```

## What to Check

- **Average total score increases**: should rise over steps and plateau; if it stays flat, check component transforms and score ranges.
- **Prior and agent NLL stay close**: a large gap means the agent is drifting from the prior — lower `sigma` or reduce `batch_size`.
- **Unique scaffolds remain stable**: a sharp drop in scaffold count signals mode collapse — lower `bucket_size` or tighten `minscore`.
- **No component stuck at 0**: if a component score never rises, the transform range may not match the raw value range.

## Output

One CSV file is written per stage (prefixed with `summary_csv_prefix`). Each row is one generated molecule from one epoch.

| Column | Description |
|--------|-------------|
| `SMILES` | Generated molecule |
| `AGENT` | Agent log-likelihood |
| `PRIOR` | Prior log-likelihood |
| `AUGMENTED_NLL` | Augmented log-likelihood used in the DAP loss |
| `total_score` | Aggregated score |
| `<component>_raw` | Raw component value |
| `<component>` | Transformed component score (0–1) |
| `step` | Epoch number |
