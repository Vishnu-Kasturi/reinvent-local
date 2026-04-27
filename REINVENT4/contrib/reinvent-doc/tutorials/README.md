# Tutorials

Step-by-step guides for common REINVENT4 workflows. Each tutorial includes descriptions of the input files, configuration settings in TOML file, and expected outputs.

## Running REINVENT4

All workflows follow the same pattern: prepare a TOML config file, then run:

```bash
reinvent config.toml
```

Optional flags:

```bash
reinvent config.toml -l run.log -s 42
```

| Flag | Description |
|------|-------------|
| `-l FILE` | Write log output to a file instead of stderr |
| `-s N` | Set random seed for reproducibility |

## Tutorials

1. [Sampling](sampling.md) — generate molecules from a prior without any training or optimization.
2. [Transfer Learning](tl.md) — fine-tune a prior on a focused SMILES dataset to bias generation toward a target chemical series
3. [Reinforcement Learning](rl.md) — optimize an agent against a multi-component scoring function; includes staged/curriculum learning
4. [Scoring Function Design](scoring_function.md) — how to formulate objectives, choose transforms and weights, use built-in components, and write custom ones
5. [Scoring](scoring.md) — evaluate an existing SMILES list against a scoring function without running RL; useful for validating your scoring setup
6. [Common Workflows](workflows.md) — end-to-end strategies combining sampling, TL, and RL for different scenarios
7. [Monitoring and Analysis](monitoring.md) — TensorBoard metrics during TL/RL, CSV output columns, and DataWarrior visualisation

## Example Config Files

Ready-to-run TOML configs for each tutorial are provided in `example_cfgs/`:

```
example_cfgs/
├── sampling/   reinvent.toml, libinvent.toml, linkinvent.toml, mol2mol.toml
├── tl/         reinvent.toml, mol2mol.toml
└── rl/         single_stage.toml, multi_stage.toml
```

These configs reference prior model files in `prior/`. Download the prior models from [Zenodo (record 15641297)](https://zenodo.org/records/15641297) and place them in `prior/` before running.
