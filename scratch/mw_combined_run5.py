import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

RUN5 = Path(__file__).resolve().parent.parent / "iict_libinvent" / "run5"
SETS = {
    "Composite": "top5_sol_pic50_dock.csv",
    "ASP": "top5_asp_interaction.csv",
    "Solubility": "top5_solubility.csv",
    "Docking": "top5_docking.csv",
}

rows = []
for cat, fname in SETS.items():
    df = pd.read_csv(RUN5 / fname)
    for _, r in df.iterrows():
        mol = Chem.MolFromSmiles(str(r["SMILES"]))
        mw = Descriptors.MolWt(mol) if mol else float("nan")
        rows.append(
            {
                "category": cat,
                "rank": int(r["rank"]),
                "molID": int(r["molID"]),
                "SMILES": r["SMILES"],
                "MW": round(mw, 2),
                "pIC50": r["pIC50"],
                "Solubility": r["Solubility"],
                "Docking_Score": r["Docking_Score"],
                "composite": r.get("composite", ""),
                "ASP122": int(r["ASP122_Interactions"]),
                "TYR56": int(r["TYR56_Interactions"]),
            }
        )

all_df = pd.DataFrame(rows)

print("=== Per category (top 5 each) ===")
print(f"{'Category':<14} {'Min':>8} {'Max':>8} {'Avg':>8}  n")
print("-" * 44)
for cat in SETS:
    s = all_df[all_df["category"] == cat]["MW"]
    print(f"{cat:<14} {s.min():>8.2f} {s.max():>8.2f} {s.mean():>8.2f}  {len(s)}")

uniq = all_df.drop_duplicates("molID").sort_values("molID")
mw = uniq["MW"]
print()
print("=== ALL COMBINED (unique molecules across 4 lists) ===")
print(f"Unique molecules: {len(uniq)}  (20 entries total, overlaps removed)")
print(f"MW Min:  {mw.min():.2f} Da")
print(f"MW Max:  {mw.max():.2f} Da")
print(f"MW Avg:  {mw.mean():.2f} Da")
print()
print(f"{'molID':>6} {'MW':>8}  categories")
print("-" * 40)
for mid in sorted(uniq["molID"]):
    mrow = uniq[uniq["molID"] == mid].iloc[0]
    cats = ", ".join(sorted(all_df[all_df["molID"] == mid]["category"].unique()))
    print(f"{mid:>6} {mrow['MW']:>8.2f}  {cats}")

all_df.to_csv(RUN5 / "top5_all_combined_long.csv", index=False)

uniq_out = uniq[
    ["molID", "SMILES", "MW", "pIC50", "Solubility", "Docking_Score", "TYR56", "ASP122", "composite"]
].copy()
uniq_out["categories"] = [
    ", ".join(sorted(all_df[all_df["molID"] == mid]["category"].unique())) for mid in uniq_out["molID"]
]
uniq_out.to_csv(RUN5 / "top5_all_combined_mw.csv", index=False)

summary_rows = []
for cat in SETS:
    s = all_df[all_df["category"] == cat]["MW"]
    summary_rows.append(
        {"group": cat, "MW_min": round(s.min(), 2), "MW_max": round(s.max(), 2), "MW_avg": round(s.mean(), 2), "n": len(s)}
    )
summary_rows.append(
    {
        "group": "ALL_COMBINED_UNIQUE",
        "MW_min": round(mw.min(), 2),
        "MW_max": round(mw.max(), 2),
        "MW_avg": round(mw.mean(), 2),
        "n": len(uniq),
    }
)
pd.DataFrame(summary_rows).to_csv(RUN5 / "top5_mw_summary.csv", index=False)

print()
print("Saved:")
print(" ", RUN5 / "top5_all_combined_long.csv")
print(" ", RUN5 / "top5_all_combined_mw.csv")
print(" ", RUN5 / "top5_mw_summary.csv")
