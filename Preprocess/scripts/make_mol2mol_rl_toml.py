#!/usr/bin/env python3
"""
Generate a Mol2Mol staged-learning TOML (scaffold-hop + dock + tyr + physchem).

Optional in-RL QSAR: PD1PDL1pIC50 and PD1PDL1Sol (toggle USE_PIC50 / USE_SOL).
Post-RL: repredict pIC50 + Solubility offline (same as enrich_mol2mol_csv.py).

Edit CONFIG, then from repo root:
  python Preprocess/scripts/make_mol2mol_rl_toml.py
  python Preprocess/scripts/make_mol2mol_rl_toml.py --enrich-only
  python Preprocess/scripts/make_mol2mol_rl_toml.py --write-toml --enrich
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# ── EDIT CONFIG ───────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/genai/Vishnu/psearch-master/reinvent-local-main")

RUN_NAME = "mol2mol_sh2"
OUTPUT_TOML = BASE_DIR / "REINVENT4" / "configs" / f"pd1_pdl1_{RUN_NAME}_dock_tyr.toml"

DEVICE = "cuda:1"
PRIOR = "mol2mol_medium_similarity.prior"  # or mol2mol_scaffold_generic.prior
SMILES_FILE = "data/scafolds_5.smi"
SUMMARY_PREFIX = f"results/{RUN_NAME}"
CHKPT = f"models/{RUN_NAME}.chkpt"
TB_LOGDIR = f"tb_{RUN_NAME}"
JSON_OUT = f"json_{RUN_NAME}.json"

SCORING_AGG = "geometric_mean"
MIN_STEPS = 40
MAX_STEPS = 120

# ScaffoldHop (reverse_sigmoid on Tanimoto to leads)
SCAFFOLD_LOW = 0.40
SCAFFOLD_HIGH = 0.70
SCAFFOLD_K = 0.55
W_SCAFFOLD = 2.0

W_DOCK = 3.0
W_TYR = 2.5
W_CSP3 = 1.5
W_ROT = 1.5
W_AROMATIC = 1.5
W_MULTIRING = 1.0

# In-RL QSAR (off = use post-RL enrich for pIC50 / Solubility columns only)
USE_PIC50 = False
W_PIC50 = 2.0
PIC50_LOW = 8.5
PIC50_HIGH = 11.0
PIC50_K = 0.5

USE_SOL = False
W_SOL = 1.5
SOL_LOW = -5.5
SOL_HIGH = 2.0
SOL_K = 0.5

# Post-RL enrich (XGBoost repredict — columns pIC50, Solubility)
ENRICH_INPUT_CSV = BASE_DIR / "results" / f"{RUN_NAME}_1.csv"
ENRICH_OUTPUT_CSV = BASE_DIR / "iict_libinvent" / f"{RUN_NAME}_enriched.csv"
ENRICH_XGB_DEVICE = "cuda:0"

DOCKING_ROOT = "/home/genai/Vishnu/psearch-master/reinvent-local-main/docking_runs"
RECEPTOR = "/home/genai/navneet/iict/pdl1/docking_TL_dataset/receptor.pdb"
AUTOBOX = "/home/genai/navneet/iict/pdl1/docking_TL_dataset/ref_ligand.pdb"
GNINA = "/home/genai/Documents/gnina/gnina"

PIC50_MODEL = "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_model.ubj"
PIC50_SCALER = "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_scaler.pkl"
SOL_MODEL = "Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj"
SOL_SCALER = "Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl"
# ─────────────────────────────────────────────────────────────────────────────


def _p(base: Path, rel: str) -> str:
    return str(base / rel)


def _dock_block(name: str, weight: float, indent: str = "") -> str:
    return f"""{indent}[[stage.scoring.component.DockingScore.endpoint]]
{indent}name                    = "{name}"
{indent}weight                  = {weight}
{indent}params.receptor_path    = "{RECEPTOR}"
{indent}params.autobox_ligand   = "{AUTOBOX}"
{indent}params.gnina_executable = "{GNINA}"
{indent}params.output_root      = "{DOCKING_ROOT}"
{indent}params.cnn_scoring      = "none"
{indent}params.keep_outputs     = "true"
"""


def _tyr_block(name: str, weight: float, indent: str = "") -> str:
    return f"""{indent}[[stage.scoring.component.TyrosineInteraction.endpoint]]
{indent}name                    = "{name}"
{indent}weight                  = {weight}
{indent}params.receptor_path    = "{RECEPTOR}"
{indent}params.autobox_ligand   = "{AUTOBOX}"
{indent}params.gnina_executable = "{GNINA}"
{indent}params.output_root      = "{DOCKING_ROOT}"
{indent}params.cnn_scoring      = "none"
{indent}params.keep_outputs     = "true"
{indent}params.tyr_residue      = "56"
"""


def build_toml() -> str:
    smiles_path = _p(BASE_DIR, SMILES_FILE)
    prior_path = _p(BASE_DIR, f"REINVENT4/priors/{PRIOR}")
    summary = _p(BASE_DIR, SUMMARY_PREFIX)
    chkpt = _p(BASE_DIR, CHKPT)
    pic50_model = _p(BASE_DIR, PIC50_MODEL)
    pic50_scaler = _p(BASE_DIR, PIC50_SCALER)
    sol_model = _p(BASE_DIR, SOL_MODEL)
    sol_scaler = _p(BASE_DIR, SOL_SCALER)

    qsar_blocks = ""
    if USE_PIC50:
        qsar_blocks += f"""
# ── PD1-PDL1 pIC50 (in-RL QSAR) ───────────────────────────────────────────────
[[stage.scoring.component]]
[stage.scoring.component.PD1PDL1pIC50]
[[stage.scoring.component.PD1PDL1pIC50.endpoint]]
name               = "PD1PDL1pIC50"
weight             = {W_PIC50}
transform.type     = "sigmoid"
transform.high     = {PIC50_HIGH}
transform.low      = {PIC50_LOW}
transform.k        = {PIC50_K}
params.model_path  = "{pic50_model}"
params.scaler_path = "{pic50_scaler}"
"""
    if USE_SOL:
        qsar_blocks += f"""
# ── PD1-PDL1 Solubility (in-RL QSAR) ──────────────────────────────────────────
[[stage.scoring.component]]
[stage.scoring.component.PD1PDL1Sol]
[[stage.scoring.component.PD1PDL1Sol.endpoint]]
name               = "PD1PDL1Sol"
weight             = {W_SOL}
transform.type     = "sigmoid"
transform.high     = {SOL_HIGH}
transform.low      = {SOL_LOW}
transform.k        = {SOL_K}
params.model_path  = "{sol_model}"
params.scaler_path = "{sol_scaler}"
"""

    return f"""##############################################################################
# Generated by make_mol2mol_rl_toml.py — RUN_NAME={RUN_NAME}
# Post-RL enrich: python Preprocess/scripts/make_mol2mol_rl_toml.py --enrich-only
#   -> {ENRICH_OUTPUT_CSV}  (SMILES + TOML scores + pIC50 + Solubility)
##############################################################################

run_type        = "staged_learning"
device          = "{DEVICE}"
tb_logdir       = "{_p(BASE_DIR, TB_LOGDIR)}"
json_out_config = "{_p(BASE_DIR, JSON_OUT)}"

[parameters]
prior_file         = "{prior_path}"
agent_file         = "{prior_path}"
smiles_file        = "{smiles_path}"
sample_strategy    = "multinomial"
distance_threshold = 100
summary_csv_prefix = "{summary}"
batch_size         = 8
unique_sequences   = true
randomize_smiles   = true
use_checkpoint     = false

[learning_strategy]
type  = "dap"
sigma = 128
rate  = 0.0001

[diversity_filter]
type        = "IdenticalTopologicalScaffold"
bucket_size = 8
minscore    = 0.45

[[stage]]
chkpt_file  = "{chkpt}"
termination = "simple"
max_score   = 1.0
min_steps   = {MIN_STEPS}
max_steps   = {MAX_STEPS}

[stage.scoring]
type = "{SCORING_AGG}"

[[stage.scoring.component]]
[stage.scoring.component.TanimotoSimilarity]
[[stage.scoring.component.TanimotoSimilarity.endpoint]]
name                = "ScaffoldHop"
weight              = {W_SCAFFOLD}
transform.type      = "reverse_sigmoid"
transform.low       = {SCAFFOLD_LOW}
transform.high      = {SCAFFOLD_HIGH}
transform.k         = {SCAFFOLD_K}
params.smiles_file  = "{smiles_path}"
params.radius       = 2
params.use_counts   = false
params.use_features = false

[[stage.scoring.component]]
[stage.scoring.component.DockingScore]
{_dock_block("DockingReward", W_DOCK)}
{_dock_block("DockingAffinity_raw", 0.0)}

[[stage.scoring.component]]
[stage.scoring.component.TyrosineInteraction]
{_tyr_block("TyrInteractionReward", W_TYR)}
{_tyr_block("TyrInteractionCount_raw", 0.0)}

[[stage.scoring.component]]
[stage.scoring.component.Csp3]
[[stage.scoring.component.Csp3.endpoint]]
name           = "LowCsp3"
weight         = {W_CSP3}
transform.type = "reverse_sigmoid"
transform.low  = 0.05
transform.high = 0.45
transform.k    = 0.4

[[stage.scoring.component]]
[stage.scoring.component.NumRotBond]
[[stage.scoring.component.NumRotBond.endpoint]]
name           = "LowRotBonds"
weight         = {W_ROT}
transform.type = "reverse_sigmoid"
transform.low  = 0
transform.high = 6
transform.k    = 0.35

[[stage.scoring.component]]
[stage.scoring.component.NumAromaticRings]
[[stage.scoring.component.NumAromaticRings.endpoint]]
name           = "AromaticRings_2_4"
weight         = {W_AROMATIC}
transform.type = "step"
transform.low  = 2
transform.high = 4

[[stage.scoring.component]]
[stage.scoring.component.NumRings]
[[stage.scoring.component.NumRings.endpoint]]
name           = "MultiRing"
weight         = {W_MULTIRING}
transform.type = "sigmoid"
transform.low  = 2
transform.high = 5
transform.k    = 0.35
{qsar_blocks}
"""


def write_toml() -> Path:
    out_path = OUTPUT_TOML
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = build_toml()
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"  USE_PIC50={USE_PIC50}  USE_SOL={USE_SOL}")
    print(f"Then: reinvent {out_path}")
    print(f"After RL: python {Path(__file__).name} --enrich-only")
    print(f"  input:  {ENRICH_INPUT_CSV}")
    print(f"  output: {ENRICH_OUTPUT_CSV}")
    return out_path


def run_enrich() -> None:
    from mol2mol_enrich_core import enrich_mol2mol_csv_file

    if not ENRICH_INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"RL CSV not found: {ENRICH_INPUT_CSV}\n"
            f"Edit ENRICH_INPUT_CSV in make_mol2mol_rl_toml.py (RUN_NAME={RUN_NAME})."
        )

    print("=================================================================")
    print("  Mol2Mol post-RL enrich (pIC50 + Solubility repredict)          ")
    print("=================================================================\n")
    out = enrich_mol2mol_csv_file(
        ENRICH_INPUT_CSV,
        ENRICH_OUTPUT_CSV,
        base_dir=BASE_DIR,
        xgb_device=ENRICH_XGB_DEVICE,
    )
    print(f"\nWrote {len(out)} rows -> {ENRICH_OUTPUT_CSV}")
    print(f"Columns: {list(out.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mol2Mol RL TOML generator + CSV enrich")
    parser.add_argument(
        "--enrich-only",
        action="store_true",
        help="Repredict pIC50/Solubility on ENRICH_INPUT_CSV (no TOML write)",
    )
    parser.add_argument(
        "--write-toml",
        action="store_true",
        help="Write TOML (default unless --enrich-only alone)",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="After writing TOML, run enrich if ENRICH_INPUT_CSV exists",
    )
    args = parser.parse_args()

    do_toml = args.write_toml or not args.enrich_only
    if do_toml:
        write_toml()
    if args.enrich_only or args.enrich:
        run_enrich()


if __name__ == "__main__":
    main()
