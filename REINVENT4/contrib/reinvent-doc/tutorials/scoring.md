# Tutorial: Scoring

The scoring run mode evaluates an existing SMILES list against a scoring function and writes the results to a CSV. No model or training is involved.

**Primary use case:** validate and iterate on your scoring function configuration before committing to a full RL run.

## Configuration

```toml
run_type = "scoring"

[parameters]
smiles_file = "molecules.smi"   # one SMILES per line
output_csv = "scores.csv"

[scoring]
type = "geometric_mean"         # aggregation across components
parallel = 4                    # number of parallel CPU workers

[[scoring.component]]
...
```

## Scoring Components

Each component computes one property per SMILES. Components are declared as `[[scoring.component]]` blocks. The component name (e.g. `QED`, `MolecularWeight`) must match the name registered in REINVENT4.

### Structure of a component block

```toml
[[scoring.component]]
[scoring.component.MolecularWeight]

[[scoring.component.MolecularWeight.endpoint]]
name = "MW"          # label used in the output CSV column
weight = 1.0         # relative weight in aggregation

transform.type = "double_sigmoid"
transform.low = 200.0
transform.high = 500.0
transform.coef_div = 500.0
transform.coef_si = 20.0
transform.coef_se = 20.0
```

Some components accept `params` for additional configuration (e.g. a reference SMILES, SMARTS pattern, or file path).

### Available components

| Component | Description |
|-----------|-------------|
| `QED` | Drug-likeness score (0–1, higher is better) |
| `SlogP` | Crippen LogP |
| `MolecularWeight` | Molecular weight in Da |
| `TPSA` | Topological polar surface area |
| `HBondAcceptors` | Number of H-bond acceptors |
| `HBondDonors` | Number of H-bond donors |
| `NumRotBond` | Number of rotatable bonds |
| `NumRings` | Total number of rings |
| `NumAromaticRings` | Number of aromatic rings |
| `NumAliphaticRings` | Number of aliphatic rings |
| `Csp3` | Fraction of sp3 carbons |
| `NumHeavyAtoms` | Number of heavy atoms |
| `SAScore` | Synthetic accessibility score |
| `TanimotoDistance` | Tanimoto similarity to a reference SMILES |
| `GroupCount` | Count of a SMARTS substructure (filter) |
| `MatchingSubstructure` | Penalty if a SMARTS substructure is present (multiplied against total score) |
| `custom_alerts` | Zero the total score if any SMARTS alert matches (global filter) |
| `PMI` | Principal moment of inertia — 3D shape descriptor (`npr1` or `npr2`) |

External components (DockStream, Maize, ChemProp, REST) require additional setup and are not covered here.

## Transforms

Transforms map the raw component value to [0, 1] before aggregation. All transforms are optional — without one, the raw value is passed directly (only appropriate if it is already in [0, 1], like QED).

### `sigmoid`

Scores rise from 0 to 1 as the value increases through the `[low, high]` range. Use when higher is better (e.g. QED, similarity).

```toml
transform.type = "sigmoid"
transform.low = 0.3
transform.high = 0.7
transform.k = 0.5      # steepness; larger = sharper transition
```

### `reverse_sigmoid`

Scores fall from 1 to 0 as the value increases. Use when lower is better (e.g. LogP, rotatable bonds).

```toml
transform.type = "reverse_sigmoid"
transform.low = 1.0
transform.high = 3.0
transform.k = 0.5
```

### `double_sigmoid`

Scores peak at 1 within the `[low, high]` window and fall to 0 outside it. Use for properties with a preferred range (e.g. MW 200–500 Da, TPSA 0–140 Å²).

```toml
transform.type = "double_sigmoid"
transform.low = 200.0
transform.high = 500.0
transform.coef_div = 500.0   # normalisation divisor, typically set to high
transform.coef_si = 20.0     # steepness of the left (rising) edge
transform.coef_se = 20.0     # steepness of the right (falling) edge
```

### `step`

Returns 1.0 if the value is within `[low, high]`, 0.0 otherwise. Hard cutoff, no gradient.

```toml
transform.type = "step"
transform.low = 0
transform.high = 3
```

## Aggregation

The `[scoring]` `type` controls how component scores are combined into a total score:

- `geometric_mean` (default and recommended): sensitive to low-scoring components — a single zero pulls the total to zero. Encourages balanced optimisation.
- `arithmetic_mean`: averages scores; a high score on one component can compensate for a low score on another.

## Filters vs. Components

Two special components operate differently from standard components:

- **`custom_alerts`**: a global filter — if any SMARTS pattern matches, the total score is set to 0 regardless of all other components. No weight is needed.
- **`MatchingSubstructure`**: a penalty multiplier — the total score is multiplied by the component score. Use to penalise molecules containing an unwanted substructure.

## Example: Drug-likeness Filter

```toml
run_type = "scoring"

[parameters]
smiles_file = "molecules.smi"
output_csv = "scores.csv"

[scoring]
type = "geometric_mean"

[[scoring.component]]
[scoring.component.custom_alerts]
[[scoring.component.custom_alerts.endpoint]]
name = "Alerts"
params.smarts = [
    "[*;r{8-17}]",   # macrocycles
    "[#8][#8]",       # peroxide
    "[#6;+]",         # charged carbon
    "[#16][#16]"      # disulfide
]

[[scoring.component]]
[scoring.component.QED]
[[scoring.component.QED.endpoint]]
name = "QED"
weight = 1.0

[[scoring.component]]
[scoring.component.MolecularWeight]
[[scoring.component.MolecularWeight.endpoint]]
name = "MW"
weight = 1.0
transform.type = "double_sigmoid"
transform.low = 200.0
transform.high = 500.0
transform.coef_div = 500.0
transform.coef_si = 20.0
transform.coef_se = 20.0

[[scoring.component]]
[scoring.component.SlogP]
[[scoring.component.SlogP.endpoint]]
name = "LogP"
weight = 1.0
transform.type = "reverse_sigmoid"
transform.low = 1.0
transform.high = 3.0
transform.k = 0.5
```

## Running

```bash
reinvent scoring.toml
```

## Output

The output CSV contains one row per input SMILES with columns for the total score and each component score (raw and transformed).

| Column | Description |
|--------|-------------|
| `SMILES` | Canonicalized input SMILES |
| `total_score` | Aggregated score across all components |
| `<name>_raw` | Raw value from the component |
| `<name>` | Transformed value (0–1) used in aggregation |
