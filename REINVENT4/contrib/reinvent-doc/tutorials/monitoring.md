# Monitoring and Analysis

REINVENT4 produces two complementary outputs during TL and RL runs: **TensorBoard event files** for live training curves, and **CSV files** with per-molecule records for post-hoc analysis.

---

## TensorBoard

### Enabling TensorBoard logging

Add `tb_logdir` to the `[parameters]` block of your TOML file:

```toml
[parameters]
tb_logdir = "tb_logs"   # writes event files to ./tb_logs/
```

For staged RL runs the log directory is suffixed with the stage index automatically (e.g. `tb_logs_0`, `tb_logs_1`).

### Launching the dashboard

```bash
tensorboard --logdir tb_logs
```

Then open `http://localhost:6006` in a browser.

### What is logged during Transfer Learning

| Scalar | Description |
|--------|-------------|
| `A_Mean NLL loss / Training Loss` | Mean NLL on the training batch (epoch − 1) |
| `A_Mean NLL loss / Sample Loss` | Mean NLL on molecules sampled from the agent |
| `A_Mean NLL loss / Validation Loss` | Mean NLL on the validation set (if provided) |
| `B_Fraction valid SMILES` | Fraction of sampled SMILES that parsed successfully |
| `C_Fraction duplicate SMILES` | Fraction of exact duplicates in the sample |
| `D_Internal Diversity of sample` | Internal diversity of sampled molecules |
| `E_Average iSIM similarity` | Mean iSIM similarity across the sample |

The **Sampled structures** image panel shows a grid of sampled molecules with their NLL values, updated each epoch.

When a reference SMILES set is provided, a **Tanimoto similarity** histogram (RDKit fingerprints) is written per epoch.

**What to watch:** Training Loss and Sample Loss should decrease and converge together. A growing gap (Sample Loss >> Training Loss) indicates overfitting — reduce the number of epochs or increase regularization.

### What is logged during Reinforcement Learning

| Scalar | Description |
|--------|-------------|
| `Loss` | RL training loss (DAP objective) |
| `Loss (likelihood averages) / prior NLL` | Mean NLL under the fixed prior |
| `Loss (likelihood averages) / agent NLL` | Mean NLL under the current agent |
| `Loss (likelihood averages) / augmented NLL` | Mean augmented NLL (target for DAP) |
| `Average total score` | Mean total score across the batch |
| `Fraction of valid SMILES` | Fraction of valid SMILES in the batch |
| `Fraction of duplicate SMILES` | Fraction of exact duplicates in the batch |
| `<component name>` | Mean transformed score for each scoring component |
| `<component name> (raw)` | Mean raw (untransformed) value for each scoring component |
| `Number of unique scaffolds` | Count of distinct Murcko scaffolds (when diversity filter is active) |
| `Number of scaffolds found more than N times` | Count of full buckets — high values indicate mode collapse |
| `iSIM: Average similarity` | Mean iSIM similarity across the batch |

The **First 30 Structures** image panel shows a grid of scored molecules updated each step.

**What to watch:**
- `Average total score` should increase over steps and plateau.
- `prior NLL` and `agent NLL` should stay close; a widening gap means the agent is drifting from the prior.
- `Number of unique scaffolds` should remain above a few dozen; a sharp drop signals scaffold collapse.
- If a component score is stuck at 0, check your transform range or component configuration.

---

## CSV Output

### RL output CSV

Each RL step appends one row per molecule to the output CSV. The column order is fixed as follows:

| Column | Description |
|--------|-------------|
| `Agent` | NLL assigned to the SMILES by the current agent |
| `Prior` | NLL assigned to the SMILES by the fixed prior |
| `Target` | Augmented NLL (the RL training target) |
| `Score` | Geometric mean of all component scores (0–1) |
| `SMILES` | Generated SMILES string |
| `SMILES_state` | Validity flag: `1` = valid, `2` = duplicate, `3` = invalid |

**Generator-specific columns** (appended after `SMILES_state` when applicable):

| Generator | Extra columns |
|-----------|---------------|
| LibInvent | `Input_Scaffold`, `R-groups` |
| LinkInvent | `Warheads`, `Linker` |
| Mol2Mol | `Input_SMILES` |

**Diversity filter column** (present when a diversity filter is configured):

| Column | Description |
|--------|-------------|
| `Scaffold` | Murcko scaffold of the molecule |

**Per-component columns** (one pair per scoring component):

| Column | Description |
|--------|-------------|
| `<component name>` | Transformed score (0–1, after sigmoid/step transform) |
| `<component name> (raw)` | Raw value before transformation (e.g. docking score in kcal/mol) |

**Metadata columns** (present when a component returns metadata):

| Column | Description |
|--------|-------------|
| `<metadata key> (<component name>)` | Auxiliary value returned by the component (e.g. pose RMSD, binding mode label) |

**Final column:**

| Column | Description |
|--------|-------------|
| `step` | Epoch/step index |

### Sampling output CSV

The sampling mode writes a simpler CSV with one row per molecule:

| Column | Description |
|--------|-------------|
| `SMILES` | Generated SMILES |
| `NLL` | Negative log-likelihood assigned by the model |

Generator-specific input columns (`Input_Scaffold`, `Warheads`, etc.) are prepended when using LibInvent, LinkInvent, or Mol2Mol.

---

## Visualising with DataWarrior

[DataWarrior](https://openmolecules.org/datawarrior/) is a free desktop tool that can render SMILES directly and is well-suited for browsing REINVENT output CSVs.

### Opening a CSV

1. Launch DataWarrior and choose **File → Open**.
2. Select your CSV file.
3. In the import dialog, DataWarrior will detect column types. Confirm that the `SMILES` column is recognised as **Structure** (it usually is automatically; if not, right-click the column header and set its type).

### Exploring the output

- The **Structure View** renders each molecule as a 2D structure grid. You can resize the grid and sort by any numeric column (e.g. `Score`, `<component> (raw)`).
- Use **Filter** panel on the left to restrict to valid molecules (`SMILES_state = 1`), high-scoring molecules (`Score > 0.5`), or specific scaffold families.
- Use **Analysis → New Bar Chart** or **New Scatter Plot** to visualise score distributions, component correlations, or score vs. step trends.
- Columns containing raw component values (e.g. docking scores, QED, MW) can be plotted directly — no preprocessing needed.

### Recommended views for RL output

| Goal | How |
|------|-----|
| Score progression over training | Scatter plot: X = `step`, Y = `Score` |
| Score vs. raw property | Scatter plot: X = `<component> (raw)`, Y = `Score` |
| Browse top compounds | Sort by `Score` descending, switch to Structure View |
| Diversity check | Use **Analysis → Scaffold Analysis** |
| Remove duplicates / invalids | Filter `SMILES_state` = `1` |
