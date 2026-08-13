import os
import sys
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from rdkit import Chem

import MDAnalysis as mda
import prolif as plf


# ============================================================
# CONFIGURATION
# ============================================================

DOCK_PY = (
    "/home/genai/Vishnu/psearch-master/reinvent-local-main/"
    "Preprocess/scripts/dock.py"
)

RECEPTOR_PDB = (
    "/home/genai/navneet/iict/pdl1/docking_TL_dataset/"
    "receptor.pdb"
)

OUTPUT_DIR = "pipeline_output"


# ============================================================
# RUN COMMAND
# ============================================================

def run(cmd):
    print("\n>> " + " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


# ============================================================
# CLEAN / CANONICALIZE SMILES
# ============================================================

def clean_smiles(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            "Invalid SMILES. RDKit could not parse the input."
        )

    clean = Chem.MolToSmiles(
        mol,
        canonical=True
    )

    return clean


# ============================================================
# CREATE INPUT CSV
# ============================================================

def create_input_csv(smiles, output_dir):

    input_csv = Path(output_dir) / "input.csv"

    df = pd.DataFrame({
        "SMILES": [smiles]
    })

    df.to_csv(
        input_csv,
        index=False
    )

    return str(input_csv)


# ============================================================
# RUN DOCKING
# ============================================================

def run_docking(smiles, output_dir):

    output_dir = Path(output_dir)

    input_csv = create_input_csv(
        smiles,
        output_dir
    )

    run([
        "python",
        DOCK_PY,
        str(input_csv),
        str(output_dir) + "/"
    ])

    log_file = output_dir / "mol0_log.txt"

    output_sdf = output_dir / "mol0_out.sdf"

    if not log_file.exists():

        raise FileNotFoundError(
            f"Docking log not found: {log_file}"
        )

    if not output_sdf.exists():

        raise FileNotFoundError(
            f"Docked SDF not found: {output_sdf}"
        )

    return str(log_file), str(output_sdf)


# ============================================================
# EXTRACT BEST DOCKING SCORE
# ============================================================

def extract_best_docking_score(log_file):

    with open(log_file, "r") as f:

        text = f.read()

    pattern = (
        r"^\s*1\s+"
        r"(-?\d+(?:\.\d+)?)"
    )

    match = re.search(
        pattern,
        text,
        re.MULTILINE
    )

    if match:

        return float(
            match.group(1)
        )

    # Backup method
    pattern = (
        r"Docking Score:\s*"
        r"(-?\d+(?:\.\d+)?)"
    )

    match = re.search(
        pattern,
        text
    )

    if match:

        return float(
            match.group(1)
        )

    raise ValueError(
        "Could not extract docking score "
        "from GNINA log."
    )


# ============================================================
# EXTRACT BEST GNINA POSE
# GNINA stores poses in order:
# molecule 0 = pose 1 = best pose
# ============================================================

def get_best_pose(output_sdf):

    supplier = Chem.SDMolSupplier(
        output_sdf,
        removeHs=False
    )

    for mol in supplier:

        if mol is not None:

            return mol

    raise ValueError(
        "Could not read any valid pose "
        "from docked SDF."
    )


# ============================================================
# LOAD PROTEIN FOR PROLIF  (fixes AtomValenceException)
# ============================================================

def load_protein_for_prolif(receptor_pdb):
    """
    Load receptor PDB for ProLIF.

    Partial CONECT records in crystal-structure PDBs prevent MDAnalysis
    from guessing bonds, which causes:
        AtomValenceException: Explicit valence for atom # N, 5, ...
    Fix: strip CONECT records, then force guess_bonds().
    See: https://github.com/chemosim-lab/ProLIF/issues/196
    """
    import tempfile

    # Write a temp PDB with all CONECT records removed
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    with open(receptor_pdb) as f:
        for line in f:
            if not line.startswith("CONECT"):
                tmp.write(line)
    tmp.close()

    try:
        u = mda.Universe(tmp.name)
        ag = u.select_atoms("protein and not resname HOH WAT TIP3 SOL")
        if len(ag) == 0:
            ag = u.atoms
        ag.guess_bonds()
        return plf.Molecule.from_mda(ag)
    except Exception:
        # Fallback: RDKit assigns bonds from residue templates
        rdmol = Chem.MolFromPDBFile(receptor_pdb, removeHs=False, sanitize=False)
        if rdmol is None:
            raise
        return plf.Molecule.from_rdkit(rdmol)
    finally:
        os.unlink(tmp.name)


# ============================================================
# PROLIF INTERACTION ANALYSIS
# ============================================================

def analyze_interactions(
    receptor_pdb,
    docked_sdf
):

    print(
        "\nLoading receptor into MDAnalysis..."
    )

    print(
        "Loading best GNINA pose..."
    )

    ligand_rdkit = get_best_pose(
        docked_sdf
    )

    print(
        "Using GNINA pose 1 (best docking pose)."
    )

    ligand = plf.Molecule.from_rdkit(
        ligand_rdkit
    )

    protein = load_protein_for_prolif(
        receptor_pdb
    )

    fp = plf.Fingerprint()

    print(
        "Running ProLIF interaction analysis..."
    )

    fp.run(
        ligand,
        protein
    )

    return fp


# ============================================================
# EXTRACT TYR INTERACTIONS
# ============================================================

def get_tyr_interactions(fp):

    tyr_interactions = []

    interactions = fp.ifp

    for residue_pair, interaction_dict in interactions.items():

        ligand_residue = residue_pair[0]

        protein_residue = residue_pair[1]

        protein_residue_str = str(
            protein_residue
        )

        if "TYR" not in protein_residue_str:

            continue

        for interaction_name, metadata in interaction_dict.items():

            if metadata:

                tyr_interactions.append({
                    "ligand": str(
                        ligand_residue
                    ),

                    "protein": str(
                        protein_residue
                    ),

                    "interaction": interaction_name,

                    "metadata": metadata
                })

    return tyr_interactions


# ============================================================
# GET PI-PI STACKING INTERACTIONS
# ============================================================

def get_pi_stacking_interactions(
    tyr_interactions
):

    pi_pi = []

    for interaction in tyr_interactions:

        name = str(
            interaction["interaction"]
        ).lower()

        if (
            "pistacking" in name
            or "pi_stack" in name
            or "pi-stacking" in name
        ):

            pi_pi.append(
                interaction
            )

    return pi_pi


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    docking_score,
    tyr_interactions,
    pi_pi_interactions
):

    print("\n" + "=" * 60)

    print(
        "DOCKING RESULT"
    )

    print("=" * 60)

    print(
        f"Best docking score: "
        f"{docking_score:.2f} kcal/mol"
    )

    print("\n" + "=" * 60)

    print(
        "TYROSINE INTERACTIONS"
    )

    print("=" * 60)

    if not tyr_interactions:

        print(
            "No TYR interactions detected."
        )

    else:

        print(
            f"Total TYR interactions: "
            f"{len(tyr_interactions)}"
        )

        print()

        for i, interaction in enumerate(
            tyr_interactions,
            start=1
        ):

            print(
                f"{i}. "
                f"Protein: "
                f"{interaction['protein']}"
            )

            print(
                f"   Interaction: "
                f"{interaction['interaction']}"
            )

    print("\n" + "=" * 60)

    print(
        "TYR PI-PI STACKING"
    )

    print("=" * 60)

    if not pi_pi_interactions:

        print(
            "No TYR Pi-Pi stacking detected."
        )

    else:

        print(
            f"TYR Pi-Pi stacking interactions: "
            f"{len(pi_pi_interactions)}"
        )

        print()

        for i, interaction in enumerate(
            pi_pi_interactions,
            start=1
        ):

            print(
                f"{i}. "
                f"Protein: "
                f"{interaction['protein']}"
            )

            print(
                f"   Interaction: "
                f"{interaction['interaction']}"
            )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    output_dir,
    smiles,
    docking_score,
    tyr_interactions,
    pi_pi_interactions
):

    result_file = Path(
        output_dir
    ) / "interaction_results.txt"

    with open(result_file, "w") as f:

        f.write(
            "=" * 60 + "\n"
        )

        f.write(
            "INPUT SMILES\n"
        )

        f.write(
            "=" * 60 + "\n"
        )

        f.write(
            smiles + "\n\n"
        )

        f.write(
            "=" * 60 + "\n"
        )

        f.write(
            "DOCKING RESULT\n"
        )

        f.write(
            "=" * 60 + "\n"
        )

        f.write(
            f"Best docking score: "
            f"{docking_score:.2f} kcal/mol\n\n"
        )

        f.write(
            "=" * 60 + "\n"
        )

        f.write(
            "ALL TYR INTERACTIONS\n"
        )

        f.write(
            "=" * 60 + "\n"
        )

        if not tyr_interactions:

            f.write(
                "No TYR interactions detected.\n"
            )

        else:

            for interaction in tyr_interactions:

                f.write(
                    f"Protein: "
                    f"{interaction['protein']}\n"
                )

                f.write(
                    f"Interaction: "
                    f"{interaction['interaction']}\n"
                )

                f.write(
                    "\n"
                )

        f.write(
            "=" * 60 + "\n"
        )

        f.write(
            "TYR PI-PI STACKING\n"
        )

        f.write(
            "=" * 60 + "\n"
        )

        if not pi_pi_interactions:

            f.write(
                "No TYR Pi-Pi stacking detected.\n"
            )

        else:

            for interaction in pi_pi_interactions:

                f.write(
                    f"Protein: "
                    f"{interaction['protein']}\n"
                )

                f.write(
                    f"Interaction: "
                    f"{interaction['interaction']}\n\n"
                )

    print(
        f"\nResults saved to: "
        f"{result_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "python test_pipeline.py "
            "'<SMILES>'"
        )

        sys.exit(1)

    raw_smiles = sys.argv[1]

    print("\n" + "=" * 60)

    print(
        "INPUT SMILES"
    )

    print("=" * 60)

    print(
        raw_smiles
    )

    print("\n" + "=" * 60)

    print(
        "CLEANING / CANONICALIZING SMILES"
    )

    print("=" * 60)

    smiles = clean_smiles(
        raw_smiles
    )

    print(
        smiles
    )

    output_dir = Path(
        OUTPUT_DIR
    )

    if output_dir.exists():

        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_file, output_sdf = run_docking(
        smiles,
        str(output_dir)
    )

    docking_score = (
        extract_best_docking_score(
            log_file
        )
    )

    print("\n" + "=" * 60)

    print(
        "DOCKING RESULT"
    )

    print("=" * 60)

    print(
        f"Best docking score: "
        f"{docking_score:.2f} kcal/mol"
    )

    print("\n" + "=" * 60)

    print(
        "RUNNING PROLIF INTERACTION ANALYSIS"
    )

    print("=" * 60)

    fp = analyze_interactions(
        RECEPTOR_PDB,
        output_sdf
    )

    tyr_interactions = (
        get_tyr_interactions(fp)
    )

    pi_pi_interactions = (
        get_pi_stacking_interactions(
            tyr_interactions
        )
    )

    print_results(
        docking_score,
        tyr_interactions,
        pi_pi_interactions
    )

    save_results(
        str(output_dir),
        smiles,
        docking_score,
        tyr_interactions,
        pi_pi_interactions
    )

    print("\n" + "=" * 60)

    print(
        "PIPELINE FINISHED SUCCESSFULLY"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()
