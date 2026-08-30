import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

RUN5 = Path(__file__).resolve().parent.parent / "iict_libinvent" / "run5"
SETS = {
    "Composite (total score)": "top5_sol_pic50_dock.csv",
    "ASP interaction": "top5_asp_interaction.csv",
    "Solubility": "top5_solubility.csv",
    "Docking": "top5_docking.csv",
}


def mw_stats(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    rows = []
    mws = []
    for _, r in df.iterrows():
        mol = Chem.MolFromSmiles(str(r["SMILES"]))
        mw = Descriptors.MolWt(mol) if mol else float("nan")
        mws.append(mw)
        rows.append({"molID": int(r["molID"]), "rank": int(r["rank"]), "MW": round(mw, 2)})
    s = pd.Series(mws)
    return {"min": round(s.min(), 2), "max": round(s.max(), 2), "avg": round(s.mean(), 2), "molecules": rows}


print("run5 — Molecular Weight (Da) for top 5 molecules\n")
print(f"{'Category':<26} {'Min':>8} {'Max':>8} {'Avg':>8}")
print("-" * 54)
for label, fname in SETS.items():
    st = mw_stats(RUN5 / fname)
    print(f"{label:<26} {st['min']:>8.2f} {st['max']:>8.2f} {st['avg']:>8.2f}")
    print("  per mol:", ", ".join(f"mol{m['molID']}={m['MW']:.1f}" for m in st["molecules"]))
