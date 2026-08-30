#!/usr/bin/env python3
"""MW summary + scaffold PNGs for iict_libinvent run6/7/8."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdDepictor

REPO = Path(__file__).resolve().parent.parent / "iict_libinvent"

RUNS = {
    "run6": {
        "scaffold": "*Cc1cc(-c2n[nH]c3cc(*)ncc23)ccn1",
    },
    "run7": {
        "scaffold": "*c1nccc(-c2ccc3nc(C4COc5ccc(*)cc5C4)[nH]c3c2)n1",
    },
    "run8": {
        "scaffold": "*c1noc(-c2cc3cc(-c4ccnn4*)ccc3[nH]2)n1",
    },
}

SETS = {
    "Composite": "top5_sol_pic50_dock.csv",
    "ASP": "top5_asp_interaction.csv",
    "Solubility": "top5_solubility.csv",
    "Docking": "top5_docking.csv",
}


def parse_smiles(smiles: str) -> Chem.Mol | None:
    for s in (smiles, smiles.replace("*", "[*]")):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            return mol
    return None


def render_scaffold(smiles: str, out_path: Path) -> None:
    mol = parse_smiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse scaffold: {smiles}")
    rdDepictor.Compute2DCoords(mol)
    opts = Draw.rdMolDraw2D.MolDrawOptions()
    opts.bondLineWidth = 2.2
    opts.padding = 0.12
    img = Draw.MolToImage(mol, size=(700, 500), options=opts)
    img.save(str(out_path))


def mw_for_run(run_dir: Path) -> pd.DataFrame:
    rows = []
    for cat, fname in SETS.items():
        path = run_dir / fname
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            mol = Chem.MolFromSmiles(str(r["SMILES"]))
            mw = round(Descriptors.MolWt(mol), 2) if mol else float("nan")
            rows.append({"category": cat, "molID": int(r["molID"]), "rank": int(r["rank"]), "MW": mw})
    return pd.DataFrame(rows)


def process_run(run_name: str, scaffold_smiles: str) -> None:
    run_dir = REPO / run_name
    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"{'='*60}")

    # Scaffold PNG
    scaffold_path = run_dir / "scaffold_molecule.png"
    render_scaffold(scaffold_smiles, scaffold_path)
    print(f"Scaffold PNG -> {scaffold_path}")

    all_df = mw_for_run(run_dir)
    if all_df.empty:
        print("  No top5 CSVs found, skipping MW.")
        return

    print("\nPer category (top 5 each):")
    print(f"{'Category':<14} {'Min':>8} {'Max':>8} {'Avg':>8}  n")
    print("-" * 44)
    summary_rows = []
    for cat in SETS:
        s = all_df[all_df["category"] == cat]["MW"]
        if s.empty:
            continue
        print(f"{cat:<14} {s.min():>8.2f} {s.max():>8.2f} {s.mean():>8.2f}  {len(s)}")
        summary_rows.append(
            {"group": cat, "MW_min": round(s.min(), 2), "MW_max": round(s.max(), 2), "MW_avg": round(s.mean(), 2), "n": len(s)}
        )

    uniq = all_df.drop_duplicates("molID").sort_values("molID")
    mw = uniq["MW"]
    print("\nALL COMBINED (unique across 4 lists):")
    print(f"  Unique molecules: {len(uniq)}  ({len(all_df)} entries, overlaps removed)")
    print(f"  MW Min:  {mw.min():.2f} Da")
    print(f"  MW Max:  {mw.max():.2f} Da")
    print(f"  MW Avg:  {mw.mean():.2f} Da")

    summary_rows.append(
        {
            "group": "ALL_COMBINED_UNIQUE",
            "MW_min": round(mw.min(), 2),
            "MW_max": round(mw.max(), 2),
            "MW_avg": round(mw.mean(), 2),
            "n": len(uniq),
        }
    )

    # Save CSVs
    all_df.to_csv(run_dir / "top5_all_combined_long.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(run_dir / "top5_mw_summary.csv", index=False)

    uniq_out = uniq.copy()
    uniq_out["categories"] = [
        ", ".join(sorted(all_df[all_df["molID"] == mid]["category"].unique())) for mid in uniq_out["molID"]
    ]
    uniq_out.to_csv(run_dir / "top5_all_combined_mw.csv", index=False)

    print(f"\nSaved: top5_mw_summary.csv, top5_all_combined_mw.csv, top5_all_combined_long.csv")


def main() -> None:
    for run_name, cfg in RUNS.items():
        process_run(run_name, cfg["scaffold"])
    print("\nDone.")


if __name__ == "__main__":
    main()
