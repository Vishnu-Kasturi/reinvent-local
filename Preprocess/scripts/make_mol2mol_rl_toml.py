#!/usr/bin/env python3
"""
Mol2Mol RL — one script: TOML generator + post-RL CSV enrich (pIC50 / Solubility).

Edit CONFIG toggles below, then:
  python Preprocess/scripts/make_mol2mol_rl_toml.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — turn steps on/off here (True / False)
# ══════════════════════════════════════════════════════════════════════════════
WRITE_TOML = True          # write REINVENT staged-learning TOML
RUN_ENRICH = False         # repredict pIC50 + Solubility on RL summary CSV

USE_PIC50_IN_RL = False    # PD1PDL1pIC50 component inside TOML (in-RL reward)
USE_SOL_IN_RL = False      # PD1PDL1Sol component inside TOML (in-RL reward)

# ── paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/genai/Vishnu/psearch-master/reinvent-local-main")

# RL_MODE: "scaffold_hop" (reverse_sigmoid Tanimoto) | "ring_decor" (stay near lead)
RL_MODE = "ring_decor"

RUN_NAME = "mol2mol_ring_decor" if RL_MODE == "ring_decor" else "mol2mol_sh2"
OUTPUT_TOML = BASE_DIR / "REINVENT4" / "configs" / (
    f"pd1_pdl1_{RUN_NAME}_dock_tyr.toml"
    if RL_MODE == "scaffold_hop"
    else "pd1_pdl1_mol2mol_ring_decor_dock_tyr.toml"
)

ENRICH_INPUT_CSV = BASE_DIR / "results" / f"{RUN_NAME}_1.csv"
ENRICH_OUTPUT_CSV = BASE_DIR / "iict_libinvent" / f"{RUN_NAME}_enriched.csv"
ENRICH_XGB_DEVICE = "cuda:0"
SMILES_COL = "SMILES"

DEVICE = "cuda:1"
PRIOR = (
    "mol2mol_medium_similarity.prior"
    if RL_MODE == "ring_decor"
    else "mol2mol_medium_similarity.prior"
)
SMILES_FILE = "data/scafolds_5.smi"
SUMMARY_PREFIX = f"results/{RUN_NAME}"
CHKPT = f"models/{RUN_NAME}.chkpt"
TB_LOGDIR = f"tb_{RUN_NAME}"
JSON_OUT = f"json_{RUN_NAME}.json"

SCORING_AGG = "geometric_mean"
MIN_STEPS = 40
MAX_STEPS = 120

SCAFFOLD_LOW = 0.40
SCAFFOLD_HIGH = 0.70
SCAFFOLD_K = 0.55
W_SCAFFOLD = 2.0

NEAR_LEAD_LOW = 0.58
NEAR_LEAD_HIGH = 0.88
NEAR_LEAD_K = 0.42
W_NEAR_LEAD = 2.5
DIV_MINSCORE = 0.42 if RL_MODE == "ring_decor" else 0.45
DIV_FILTER = "IdenticalMurckoScaffold" if RL_MODE == "ring_decor" else "IdenticalTopologicalScaffold"
W_DOCK = 3.0
W_TYR = 2.5
W_CSP3 = 1.5
W_ROT = 1.5
W_AROMATIC = 1.5
W_MULTIRING = 1.0

W_PIC50 = 2.0
PIC50_LOW = 8.5
PIC50_HIGH = 11.0
PIC50_K = 0.5
W_SOL = 1.5
SOL_LOW = -5.5
SOL_HIGH = 2.0
SOL_K = 0.5

DOCKING_ROOT = "/home/genai/Vishnu/psearch-master/reinvent-local-main/docking_runs"
RECEPTOR = "/home/genai/navneet/iict/pdl1/docking_TL_dataset/receptor.pdb"
AUTOBOX = "/home/genai/navneet/iict/pdl1/docking_TL_dataset/ref_ligand.pdb"
GNINA = "/home/genai/Documents/gnina/gnina"
# ══════════════════════════════════════════════════════════════════════════════

RL_SCORING_COLUMNS = [
    "NearLead",
    "NearLead (raw)",
    "MaxLeadTanimoto_raw",
    "MaxLeadTanimoto_raw (raw)",
    "ScaffoldHop",
    "ScaffoldHop (raw)",
    "DockingReward",
    "DockingReward (raw)",
    "DockingAffinity_raw",
    "DockingAffinity_raw (raw)",
    "TyrInteractionReward",
    "TyrInteractionReward (raw)",
    "TyrInteractionCount_raw",
    "TyrInteractionCount_raw (raw)",
    "tyr_pi_stacking (TyrInteractionReward)",
    "LowCsp3",
    "LowCsp3 (raw)",
    "LowRotBonds",
    "LowRotBonds (raw)",
    "AromaticRings_2_4",
    "AromaticRings_2_4 (raw)",
    "MultiRing",
    "MultiRing (raw)",
    "AliphaticRings_soft",
    "AliphaticRings_soft (raw)",
    "LargestRing_5_7",
    "LargestRing_5_7 (raw)",
    "HeteroAtoms_moderate",
    "HeteroAtoms_moderate (raw)",
    "AromaticN_in_ring",
    "AromaticN_in_ring (raw)",
    "RingNH",
    "RingNH (raw)",
    "AromaticO_in_ring",
    "AromaticO_in_ring (raw)",
    "AromaticRings_2_5",
    "AromaticRings_2_5 (raw)",
    "PD1PDL1pIC50",
    "PD1PDL1pIC50 (raw)",
    "PD1PDL1Sol",
    "PD1PDL1Sol (raw)",
]


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


def _tanimoto_block(smiles_path: str) -> str:
    if RL_MODE == "ring_decor":
        return f"""[[stage.scoring.component]]
[stage.scoring.component.TanimotoSimilarity]
[[stage.scoring.component.TanimotoSimilarity.endpoint]]
name                = "NearLead"
weight              = {W_NEAR_LEAD}
transform.type      = "sigmoid"
transform.low       = {NEAR_LEAD_LOW}
transform.high      = {NEAR_LEAD_HIGH}
transform.k         = {NEAR_LEAD_K}
params.smiles_file  = "{smiles_path}"
params.radius       = 2
params.use_counts   = false
params.use_features = false

[[stage.scoring.component.TanimotoSimilarity.endpoint]]
name                = "MaxLeadTanimoto_raw"
weight              = 0.0
params.smiles_file  = "{smiles_path}"
params.radius       = 2
params.use_counts   = false
params.use_features = false
"""
    return f"""[[stage.scoring.component]]
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
"""


def _ring_decor_physchem() -> str:
    if RL_MODE != "ring_decor":
        return f"""[[stage.scoring.component]]
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
"""
    return f"""[[stage.scoring.component]]
[stage.scoring.component.NumAromaticRings]
[[stage.scoring.component.NumAromaticRings.endpoint]]
name           = "AromaticRings_2_5"
weight         = {W_AROMATIC}
transform.type = "step"
transform.low  = 2
transform.high = 5

[[stage.scoring.component]]
[stage.scoring.component.NumRings]
[[stage.scoring.component.NumRings.endpoint]]
name           = "MultiRing"
weight         = {W_MULTIRING}
transform.type = "sigmoid"
transform.low  = 2
transform.high = 6
transform.k    = 0.32

[[stage.scoring.component]]
[stage.scoring.component.NumAliphaticRings]
[[stage.scoring.component.NumAliphaticRings.endpoint]]
name           = "AliphaticRings_soft"
weight         = 0.8
transform.type = "sigmoid"
transform.low  = 0
transform.high = 2
transform.k    = 0.35

[[stage.scoring.component]]
[stage.scoring.component.LargestRingSize]
[[stage.scoring.component.LargestRingSize.endpoint]]
name           = "LargestRing_5_7"
weight         = 0.9
transform.type = "step"
transform.low  = 5
transform.high = 7

[[stage.scoring.component]]
[stage.scoring.component.NumHeteroAtoms]
[[stage.scoring.component.NumHeteroAtoms.endpoint]]
name           = "HeteroAtoms_moderate"
weight         = 1.0
transform.type = "sigmoid"
transform.low  = 2
transform.high = 14
transform.k    = 0.18

[[stage.scoring.component]]
[stage.scoring.component.GroupCount]
[[stage.scoring.component.GroupCount.endpoint]]
name           = "AromaticN_in_ring"
weight         = 1.2
transform.type = "sigmoid"
transform.low  = 0
transform.high = 4
transform.k    = 0.35
params.smarts  = "[nR]"

[[stage.scoring.component]]
[stage.scoring.component.GroupCount]
[[stage.scoring.component.GroupCount.endpoint]]
name           = "RingNH"
weight         = 1.0
transform.type = "sigmoid"
transform.low  = 0
transform.high = 3
transform.k    = 0.4
params.smarts  = "[nR;H1]"

[[stage.scoring.component]]
[stage.scoring.component.GroupCount]
[[stage.scoring.component.GroupCount.endpoint]]
name           = "AromaticO_in_ring"
weight         = 1.0
transform.type = "sigmoid"
transform.low  = 0
transform.high = 3
transform.k    = 0.4
params.smarts  = "[oR]"
"""


def build_toml() -> str:
    smiles_path = _p(BASE_DIR, SMILES_FILE)
    prior_path = _p(BASE_DIR, f"REINVENT4/priors/{PRIOR}")
    summary = _p(BASE_DIR, SUMMARY_PREFIX)
    chkpt = _p(BASE_DIR, CHKPT)
    pic50_model = _p(BASE_DIR, "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_model.ubj")
    pic50_scaler = _p(BASE_DIR, "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_scaler.pkl")
    sol_model = _p(BASE_DIR, "Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj")
    sol_scaler = _p(BASE_DIR, "Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl")

    qsar_blocks = ""
    if USE_PIC50_IN_RL:
        qsar_blocks += f"""
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
    if USE_SOL_IN_RL:
        qsar_blocks += f"""
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
# {RUN_NAME} — generated by make_mol2mol_rl_toml.py
# Post-RL: set RUN_ENRICH=True in this script
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
type        = "{DIV_FILTER}"
bucket_size = 8
minscore    = {DIV_MINSCORE}

[[stage]]
chkpt_file  = "{chkpt}"
termination = "simple"
max_score   = 1.0
min_steps   = {MIN_STEPS}
max_steps   = {MAX_STEPS}

[stage.scoring]
type = "{SCORING_AGG}"

{_tanimoto_block(smiles_path)}
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

{_ring_decor_physchem()}
{qsar_blocks}
"""


def _resolve_smiles_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    matches = [c for c in df.columns if c.lower() == preferred.lower()]
    if not matches:
        raise ValueError(f"SMILES column not found. Columns: {list(df.columns)}")
    return matches[0]


def _load_booster(model_path: Path, device: str) -> xgb.Booster:
    booster = xgb.Booster({"device": device})
    booster.load_model(str(model_path))
    return booster


def _predict(smiles: list[str], model: xgb.Booster, scaler_path: Path, feature_fn) -> list[float]:
    features, mask = feature_fn(smiles, str(scaler_path))
    preds = model.predict(xgb.DMatrix(features, nthread=-1))
    return [
        round(float(preds[i]), 4) if mask[i] and np.isfinite(preds[i]) else np.nan
        for i in range(len(smiles))
    ]


def run_enrich() -> None:
    reinvent4_dir = BASE_DIR / "REINVENT4"
    sys.path.insert(0, str(reinvent4_dir))
    from reinvent_plugins.components.nophyschem_features import (
        compute_features as compute_pic50_features,
    )
    from reinvent_plugins.components.pd1_pdl1_features import (
        compute_features as compute_sol_features,
    )

    if not ENRICH_INPUT_CSV.is_file():
        raise FileNotFoundError(f"RL CSV not found: {ENRICH_INPUT_CSV}")

    print(f"--> Loading {ENRICH_INPUT_CSV}")
    df_in = pd.read_csv(ENRICH_INPUT_CSV, on_bad_lines="skip", engine="python")
    smi_col = _resolve_smiles_col(df_in, SMILES_COL)
    work = df_in.dropna(subset=[smi_col]).copy()
    smiles_list = work[smi_col].astype(str).tolist()

    pic50_model = _load_booster(
        BASE_DIR / "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_model.ubj",
        ENRICH_XGB_DEVICE,
    )
    sol_model = _load_booster(
        BASE_DIR / "Preprocess/final_acc/pd1_pdl1_sol_final_acc_model.ubj",
        ENRICH_XGB_DEVICE,
    )

    pic50_values = _predict(
        smiles_list,
        pic50_model,
        BASE_DIR / "Preprocess/final_acc/pd1_pdl1_pic50_final_acc_scaler.pkl",
        compute_pic50_features,
    )
    sol_values = _predict(
        smiles_list,
        sol_model,
        BASE_DIR / "Preprocess/final_acc/pd1_pdl1_sol_final_acc_scaler.pkl",
        compute_sol_features,
    )

    out = pd.DataFrame({"SMILES": smiles_list})
    for col in RL_SCORING_COLUMNS:
        if col in work.columns:
            out[col] = work[col].values
    out["pIC50"] = pic50_values
    out["Solubility"] = sol_values

    ENRICH_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ENRICH_OUTPUT_CSV, index=False)
    print(f"[+] Wrote {len(out)} rows -> {ENRICH_OUTPUT_CSV}")
    print(f"    Columns: {list(out.columns)}")


def main() -> None:
    if not WRITE_TOML and not RUN_ENRICH:
        raise SystemExit("Set WRITE_TOML=True or RUN_ENRICH=True in CONFIG.")

    if WRITE_TOML:
        OUTPUT_TOML.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_TOML.write_text(build_toml(), encoding="utf-8")
        print(f"[+] TOML -> {OUTPUT_TOML}")
        print(f"    reinvent {OUTPUT_TOML}")

    if RUN_ENRICH:
        run_enrich()


if __name__ == "__main__":
    main()
