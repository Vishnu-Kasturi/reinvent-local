# REINVENT4 Documentation

Command-line tool for generative molecular design using RNNs and Transformers.


## Overview

REINVENT4 generates SMILES strings and optimizes them against user-defined scoring functions. All behavior is controlled via a TOML configuration file.


![Overview](asset/reinvent_overview.svg)
*Taken from Loeffler et al., J. Cheminformatics (2024) under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).*
### Generators

Four generators are available: **Reinvent** (de novo), **LibInvent** (scaffold decoration), **LinkInvent** (fragment linking), and **Mol2Mol** (molecule optimization). See [Core Concepts](core_concept/README.md) for details.

### Model Adaptation

Pre-trained prior models for all generators are available on [Zenodo](https://zenodo.org/records/15641297). These can be adapted to a specific task in two ways, which can be combined:

- **Transfer Learning (TL)**: fine-tune the prior on a focused set of SMILES (e.g. known actives for a target).
- **Reinforcement / Curriculum Learning (RL/CL)**: iteratively bias the agent toward molecules that score well on a user-defined scoring function. Multiple RL stages can be chained (curriculum learning).

### Configuration

All run modes (sampling, TL, RL/CL, scoring) are configured through a single TOML file. JSON is also accepted.

## Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:MolecularAI/REINVENT4.git --depth 1
   ```
2. **Create a environment** (Python 3.10+):
   ```bash
   conda create --name reinvent4 python=3.10
   conda activate reinvent4
   ```
3. **Install dependencies**:
   Run the installation script with your processor type (e.g., `cu126`, `rocm6.4`, `xpu`, `cpu`, or `mac`):
   ```bash
   python install.py cpu  # Replace 'cpu' with your target platform
   ```
4. **Verify**:
   ```bash
   reinvent --help
   ```


```{toctree}
:maxdepth: 1
:caption: Concepts

core_concept/README
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/README
tutorials/sampling
tutorials/tl
tutorials/rl
tutorials/scoring
tutorials/scoring_function
tutorials/workflows
tutorials/monitoring
```
