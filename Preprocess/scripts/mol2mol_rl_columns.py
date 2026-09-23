"""Shared Mol2Mol RL CSV column lists (TOML endpoints + XGBoost inference)."""

MOL2MOL_RL_SCORING_COLUMNS = [
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
    "PD1PDL1pIC50",
    "PD1PDL1pIC50 (raw)",
    "PD1PDL1Sol",
    "PD1PDL1Sol (raw)",
]

# Repredicted offline (enrich_mol2mol_csv.py) — same models as PD1PDL1* components.
MOL2MOL_INFERENCE_COLUMNS = ["pIC50", "Solubility"]

MOL2MOL_OUTPUT_COLUMNS = ["SMILES"] + MOL2MOL_RL_SCORING_COLUMNS + MOL2MOL_INFERENCE_COLUMNS
