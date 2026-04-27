# Tutorial: Scoring Function Design

This tutorial covers how to design and configure scoring functions for RL runs, and how to write custom scoring components. For the mechanics of running RL, see the [RL tutorial](rl.md).

## Formulating a Scoring Function

A scoring function is a weighted combination of components. Each component computes one property per SMILES, maps the raw value to [0, 1] via a transform, and contributes to an aggregated total score.

### Step 1 — Identify your objectives

List the properties your molecules must satisfy. Separate them into:

- **Hard constraints** (must-pass): structural alerts, reactive groups, forbidden substructures → use `custom_alerts` or `MatchingSubstructure`
- **Soft objectives** (optimise toward): QED, LogP, MW, docking score → weighted components with transforms

Start with as few components as possible. Each additional component dilutes the signal from the others — especially with `geometric_mean` aggregation.

### Step 2 — Choose transforms

Transforms map raw values to [0, 1]. Choose based on the shape of your objective:

| Transform | Use when | Key params |
|-----------|----------|-----------|
| `sigmoid` | Higher is better (e.g. similarity, QED) | `low`, `high`, `k` |
| `reverse_sigmoid` | Lower is better (e.g. LogP, rotatable bonds) | `low`, `high`, `k` |
| `double_sigmoid` | Value should stay within a range (e.g. MW 200–500) | `low`, `high`, `coef_div`, `coef_si`, `coef_se` |
| `step` | Hard window — 1 inside, 0 outside (e.g. stereocenters ≤ 3) | `low`, `high` |

For `sigmoid` and `reverse_sigmoid`, `k` controls steepness: higher `k` = sharper transition. For `double_sigmoid`, `coef_si` and `coef_se` control the steepness of the left and right edges; `coef_div` is the normalisation divisor (typically set to `high`).

### Step 3 — Set weights

Weights are relative — they scale each component's contribution before aggregation. A component with `weight = 2.0` counts twice as much as one with `weight = 1.0`. With `geometric_mean` aggregation, a component scoring 0 pulls the total to 0 regardless of weight.

Recommended approach:
1. Set all weights to 1.0 initially.
2. Run a scoring-only job on a test set (see [Scoring tutorial](scoring.md)) to inspect the distribution of each component.
3. Adjust weights iteratively based on which objectives are being under- or over-optimised.

### Step 4 — Validate before RL

Always run a `scoring` job on a representative SMILES set before starting RL. This confirms transforms are correctly shaped and weights are sensible — without spending GPU time.

## Built-in Scoring Components

All built-in components live in `reinvent_plugins/components/`. The TOML name is the class name (case-insensitive).

### Physico-chemical (RDKit)

| TOML name | Property |
|-----------|----------|
| `QED` | Drug-likeness score (0–1) |
| `SlogP` | Crippen LogP |
| `MolecularWeight` | Molecular weight (Da) |
| `TPSA` | Topological polar surface area |
| `HBondAcceptors` | H-bond acceptors (Lipinski) |
| `HBondDonors` | H-bond donors (Lipinski) |
| `NumRotBond` | Rotatable bonds |
| `NumRings` | Total rings |
| `NumAromaticRings` | Aromatic rings |
| `NumAliphaticRings` | Aliphatic rings |
| `NumHeavyAtoms` | Heavy atom count |
| `Csp3` | Fraction of sp3 carbons |
| `SAScore` | Synthetic accessibility (1–10, lower = easier) |
| `PMI` | Principal moment of inertia (`npr1` or `npr2` via `params.property`) |

### Similarity and substructure

| TOML name | Description |
|-----------|-------------|
| `TanimotoDistance` | Tanimoto similarity to a reference SMILES (`params.smiles`, `params.radius`) |
| `GroupCount` | Count of a SMARTS pattern (`params.smarts`); filter applied before other components |
| `MatchingSubstructure` | Penalty multiplier if SMARTS is present (`params.smarts`) |
| `custom_alerts` | Global filter — zeros total score if any SMARTS matches (`params.smarts` list) |
| `MMP` | Matched molecular pair similarity to a reference |

### External

| TOML name | Description |
|-----------|-------------|
| `DockStream` | Docking via DockStream |
| `Maize` | Generic workflow runner (docking, solubility, etc.) |
| `ChemProp` | D-MPNN QSAR models |
| `ExternalProcess` | Run any external executable; communicates via JSON on stdin/stdout |
| `REST` | Generic REST API interface |

## Writing a Custom Scoring Component

### How components are discovered

REINVENT4 scans the `reinvent_plugins.components` namespace for all files whose name starts with `comp_`. No manual registration is needed — placing the file in the right location is sufficient.

**File location:** `reinvent_plugins/components/comp_<yourname>.py`

Subdirectories are also scanned, so `reinvent_plugins/components/MyTool/comp_mytool.py` works too.

### Interface

A component consists of two classes in the same file:

1. **A parameters dataclass** tagged with `@add_tag("__parameters")` — holds all user-configurable inputs. All fields must be `List` (even if only one endpoint is used), because the framework supports multiple endpoints per component.

2. **A component class** tagged with `@add_tag("__component")` — implements `__init__` and `__call__`.

The `__call__` method either receives a list of SMILES strings (default) or a list of `Chem.Mol` objects if decorated with `@molcache`. It must return a `ComponentResults` object.

**Component tags:**
- `@add_tag("__component")` — standard scoring component
- `@add_tag("__component", "filter")` — global filter; zeros total score if this component scores 0
- `@add_tag("__component", "penalty")` — penalty; multiplied against total score

**Failures:** use `np.nan` for molecules that could not be scored. Do not use 0.

### Minimal example

```python
"""Scores molecules by the number of nitrogen atoms (example)."""

__all__ = ["NitrogenCount"]
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass
from rdkit import Chem

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag


@add_tag("__parameters")
@dataclass
class Parameters:
    # All fields must be List — one entry per endpoint
    # No params needed for this example, but the class must exist
    pass


@add_tag("__component")
class NitrogenCount:
    def __init__(self, params: Parameters):
        pass  # no parameters to read

    @molcache  # converts SMILES list to Chem.Mol list before calling __call__
    def __call__(self, mols: List[Chem.Mol]) -> ComponentResults:
        scores = []
        for mol in mols:
            if mol is None:
                scores.append(np.nan)
            else:
                n = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
                scores.append(float(n))

        return ComponentResults([np.array(scores, dtype=float)])
```

### Using the component in TOML

Once the file is in place, use the class name as the component key:

```toml
[[scoring.component]]
[scoring.component.NitrogenCount]
[[scoring.component.NitrogenCount.endpoint]]
name = "N count"
weight = 1.0
transform.type = "reverse_sigmoid"
transform.low = 0
transform.high = 5
transform.k = 0.5
```

### Example with parameters

If your component needs user-supplied values (e.g. a reference SMILES or a file path):

```python
@add_tag("__parameters")
@dataclass
class Parameters:
    threshold: List[float]   # one per endpoint


@add_tag("__component")
class MyComponent:
    def __init__(self, params: Parameters):
        self.threshold = params.threshold[0]  # index 0 = first endpoint

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> ComponentResults:
        scores = [1.0 if mol and self._score(mol) > self.threshold else 0.0
                  for mol in mols]
        return ComponentResults([np.array(scores, dtype=float)])
```

In TOML:

```toml
[[scoring.component]]
[scoring.component.MyComponent]
[[scoring.component.MyComponent.endpoint]]
name = "My score"
weight = 1.0
params.threshold = [0.5]   # must be a list
```
