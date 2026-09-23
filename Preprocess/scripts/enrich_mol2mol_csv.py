#!/usr/bin/env python3
"""
Enrich a Mol2Mol REINVENT RL summary CSV with pIC50 and solubility predictions.

Writes TOML scoring-endpoint columns (dock/tyr/physchem/optional PD1PDL1*)
plus repredicted pIC50 and Solubility (XGBoost) — no Agent/Prior/Score metadata.

Edit paths below, then from repo root:
  python Preprocess/scripts/enrich_mol2mol_csv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# ── EDIT THESE ────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/genai/Vishnu/psearch-master/reinvent-local-main")

INPUT_CSV = BASE_DIR / "results" / "mol2mol_scaffold_hop_1.csv"
OUTPUT_CSV = BASE_DIR / "iict_libinvent" / "mol2mol_scaffold_hop_1_enriched.csv"

XGB_DEVICE = "cuda:0"  # use "cpu" if no GPU
SMILES_COL = "SMILES"
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from mol2mol_enrich_core import enrich_mol2mol_csv_file  # noqa: E402


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    print("=================================================================")
    print("  Mol2Mol RL CSV enrichment (pIC50 + Solubility inference)       ")
    print("=================================================================\n")
    print(f"--> Loading dataset from: {INPUT_CSV}")

    out = enrich_mol2mol_csv_file(
        INPUT_CSV,
        OUTPUT_CSV,
        base_dir=BASE_DIR,
        smiles_col=SMILES_COL,
        xgb_device=XGB_DEVICE,
    )

    print(f"\nSuccessfully wrote {len(out)} rows -> {OUTPUT_CSV}")
    print(f"Columns: {list(out.columns)}")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
