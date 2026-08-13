"""
prolif_utils.py — Robust protein loading and interaction analysis for ProLIF.

Fixes the common AtomValenceException when converting receptor PDB → RDKit,
which happens when PDB files have partial CONECT records (only on HETATM
residues) or missing explicit hydrogens.
"""
from __future__ import annotations

import os
import tempfile
from typing import List, Optional

import MDAnalysis as mda
import prolif as plf
from rdkit import Chem


def strip_conect_records(pdb_path: str) -> str:
    """Write a temp PDB with all CONECT records removed.

    Partial CONECT records (common in crystal structures) prevent MDAnalysis
    from running its bond-guessing algorithm, causing RDKit valence errors.
    See: https://github.com/chemosim-lab/ProLIF/issues/196
    """
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("CONECT"):
                tmp.write(line)
    tmp.close()
    return tmp.name


def load_protein_for_prolif(
    receptor_pdb: str,
    selection: str = "protein and not resname HOH WAT TIP3 SOL CL NA K MG CA ZN",
) -> plf.Molecule:
    """Load a receptor PDB for ProLIF interaction fingerprinting.

    Tries multiple strategies in order:
      1. MDAnalysis + forced bond guessing (strips partial CONECT first)
      2. MDAnalysis with tighter VdW radii
      3. RDKit MolFromPDBFile (handles standard amino acids by residue name)

    Raises RuntimeError if all strategies fail.
    """
    errors: List[str] = []

    # --- Strategy 1: strip CONECT, guess all bonds ---
    cleaned_pdb = None
    try:
        cleaned_pdb = strip_conect_records(receptor_pdb)
        u = mda.Universe(cleaned_pdb)
        ag = u.select_atoms(selection)
        if len(ag) == 0:
            ag = u.atoms
        ag.guess_bonds()
        mol = plf.Molecule.from_mda(ag)
        return mol
    except Exception as exc:
        errors.append(f"strategy-1 (guess_bonds): {exc}")
    finally:
        if cleaned_pdb and os.path.exists(cleaned_pdb):
            os.unlink(cleaned_pdb)

    # --- Strategy 2: tighter VdW radii to avoid false bonds ---
    cleaned_pdb = None
    try:
        cleaned_pdb = strip_conect_records(receptor_pdb)
        u = mda.Universe(cleaned_pdb)
        ag = u.select_atoms(selection)
        if len(ag) == 0:
            ag = u.atoms
        ag.guess_bonds(vdwradii={"H": 1.05, "O": 1.48, "C": 1.70, "N": 1.55, "S": 1.80})
        mol = plf.Molecule.from_mda(ag)
        return mol
    except Exception as exc:
        errors.append(f"strategy-2 (tight vdwradii): {exc}")
    finally:
        if cleaned_pdb and os.path.exists(cleaned_pdb):
            os.unlink(cleaned_pdb)

    # --- Strategy 3: RDKit PDB reader (residue-template bonds) ---
    try:
        rdmol = Chem.MolFromPDBFile(receptor_pdb, removeHs=False, sanitize=False)
        if rdmol is not None:
            return plf.Molecule.from_rdkit(rdmol)
    except Exception as exc:
        errors.append(f"strategy-3 (rdkit): {exc}")

    raise RuntimeError(
        f"Could not load protein for ProLIF from {receptor_pdb}.\n"
        + "\n".join(f"  - {e}" for e in errors)
        + "\n\nTip: prepare the receptor with explicit hydrogens (e.g. pdb2pqr, "
        "OpenBabel: obabel receptor.pdb -O receptor_h.pdb -h) and retry."
    )


def load_ligand_for_prolif(docked_sdf: str) -> plf.Molecule:
    """Load the best (first) gnina pose from an output SDF."""
    supplier = Chem.SDMolSupplier(docked_sdf, removeHs=False)
    for mol in supplier:
        if mol is not None:
            return plf.Molecule.from_rdkit(mol)
    raise ValueError(f"No valid pose in {docked_sdf}")


def run_prolif(
    receptor_pdb: str,
    docked_sdf: str,
) -> plf.Fingerprint:
    """Run ProLIF fingerprint: ligand (best pose) vs protein."""
    protein = load_protein_for_prolif(receptor_pdb)
    ligand = load_ligand_for_prolif(docked_sdf)
    fp = plf.Fingerprint()
    fp.run(ligand, protein)
    return fp


def get_tyr_interactions(fp: plf.Fingerprint) -> List[dict]:
    """Extract all TYR residue interactions from a ProLIF fingerprint."""
    interactions = []
    for (lig_res, prot_res), interaction_dict in fp.ifp.items():
        if "TYR" not in str(prot_res):
            continue
        for name, metadata in interaction_dict.items():
            if metadata:
                interactions.append({
                    "ligand": str(lig_res),
                    "protein": str(prot_res),
                    "interaction": name,
                    "metadata": metadata,
                })
    return interactions


def get_pi_stacking(interactions: List[dict]) -> List[dict]:
    """Filter TYR interactions for pi-stacking types."""
    pi_types = ("pistacking", "pi_stack", "pi-stacking", "pication")
    return [
        ix for ix in interactions
        if any(t in str(ix["interaction"]).lower() for t in pi_types)
    ]
