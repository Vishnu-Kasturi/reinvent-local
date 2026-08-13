"""
reinvent_gnina_backend.py — Shared GNINA docking + ProLIF analysis backend.

Used by DockingScore and TyrosineInteraction REINVENT4 scoring components.
Results are cached per scoring batch so both components share one dock run
per molecule instead of docking twice.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import MDAnalysis as mda
import numpy as np
import prolif as plf
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from prolif_compat import (
    count_interactions,
    iter_ifp_pairs,
    make_fingerprint,
    run_fingerprint,
    tyr56_residue_ids,
)

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger("reinvent")

_AFFINITY_RE = re.compile(
    r"^\s*1\s+(-?\d+(?:\.\d+)?)", re.MULTILINE
)


@dataclass
class GninaProlifConfig:
    receptor_path: str
    autobox_ligand: str
    gnina_executable: str = "gnina"
    output_root: str = "docking_runs"
    cnn_scoring: str = "none"
    timeout_sec: int = 300
    keep_outputs: bool = True
    tyr_residue: int = 56  # only count pi-pi stacking at this TYR residue

    def validate(self) -> None:
        if not os.path.exists(self.receptor_path):
            raise FileNotFoundError(f"Receptor not found: {self.receptor_path}")
        if not os.path.exists(self.autobox_ligand):
            raise FileNotFoundError(f"Autobox ligand not found: {self.autobox_ligand}")
        os.makedirs(self.output_root, exist_ok=True)


def build_config_from_params(params) -> GninaProlifConfig:
    """Build GninaProlifConfig from a REINVENT Parameters dataclass."""
    def _bool(val, default=True):
        if val is None:
            return default
        return str(val).lower() in ("true", "1", "yes")

    return GninaProlifConfig(
        receptor_path=params.receptor_path[0],
        autobox_ligand=params.autobox_ligand[0],
        gnina_executable=(params.gnina_executable or ["gnina"])[0],
        output_root=(params.output_root or ["docking_runs"])[0],
        cnn_scoring=(params.cnn_scoring or ["none"])[0],
        timeout_sec=int((params.timeout_sec or ["300"])[0]),
        keep_outputs=_bool((params.keep_outputs or ["true"])[0]),
        tyr_residue=int((getattr(params, "tyr_residue", None) or ["56"])[0]),
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MoleculeResult:
    smiles: str
    canonical_smiles: str = ""
    affinity: float = float("nan")
    docking_reward: float = 0.0
    tyr_interaction_count: int = 0
    tyr_pi_stacking_count: int = 0
    tyr_interaction_reward: float = 0.0
    tyr_interactions: List[dict] = field(default_factory=list)
    work_dir: str = ""
    out_sdf: str = ""
    log_file: str = ""
    docking_ok: bool = False
    prolif_ok: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Reward tier functions
# ---------------------------------------------------------------------------

def affinity_to_reward(affinity: float) -> float:
    """Map GNINA affinity (kcal/mol) to tiered reward in [0, 1]."""
    if not np.isfinite(affinity):
        return 0.0
    if affinity <= -12.0:
        return 1.0
    if affinity <= -10.0:
        return 0.5
    return 0.0


def tyr_count_to_reward(count: int) -> float:
    """Map TYR interaction count to tiered reward in [0, 1]."""
    if count >= 2:
        return 1.0
    if count == 1:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# SMILES / 3D embedding
# ---------------------------------------------------------------------------

def canonicalize(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def embed_to_sdf(smiles: str, sdf_path: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(mol, params) != 0:
        return False
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()
    return True


# ---------------------------------------------------------------------------
# GNINA
# ---------------------------------------------------------------------------

def parse_affinity_from_log(log_path: str) -> float:
    text = open(log_path).read()
    m = _AFFINITY_RE.search(text)
    if m:
        return float(m.group(1))
    m = re.search(r"Docking Score:\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    return float("nan")


def run_gnina(
    ligand_sdf: str,
    out_sdf: str,
    log_path: str,
    config: GninaProlifConfig,
) -> bool:
    cmd = [
        config.gnina_executable,
        "-r", os.path.abspath(config.receptor_path),
        "--ligand", ligand_sdf,
        "--autobox_ligand", os.path.abspath(config.autobox_ligand),
        "--cnn_scoring", config.cnn_scoring,
        "--out", out_sdf,
        "--log", log_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=config.timeout_sec,
                       capture_output=True, text=True)
        return os.path.exists(out_sdf) and os.path.exists(log_path)
    except Exception as exc:
        logger.warning(f"[GninaBackend] gnina failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# ProLIF
# ---------------------------------------------------------------------------

def _strip_conect(pdb_path: str) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("CONECT"):
                tmp.write(line)
    tmp.close()
    return tmp.name


def load_protein_for_prolif(receptor_pdb: str) -> plf.Molecule:
    """Load receptor for ProLIF, handling partial CONECT records."""
    cleaned = _strip_conect(receptor_pdb)
    try:
        u = mda.Universe(cleaned)
        ag = u.select_atoms("protein and not resname HOH WAT TIP3 SOL CL NA K MG CA ZN")
        if len(ag) == 0:
            ag = u.atoms
        ag.guess_bonds()
        return plf.Molecule.from_mda(ag)
    except Exception:
        rdmol = Chem.MolFromPDBFile(receptor_pdb, removeHs=False, sanitize=False)
        if rdmol is None:
            raise RuntimeError(f"Cannot load receptor: {receptor_pdb}")
        return plf.Molecule.from_rdkit(rdmol)
    finally:
        if os.path.exists(cleaned):
            os.unlink(cleaned)


def prepare_docked_ligand(pose_mol: Chem.Mol, smiles: str) -> Chem.Mol:
    """Assign bond orders/aromaticity from SMILES onto GNINA docked coordinates."""
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    pose = Chem.Mol(pose_mol)
    try:
        fixed = AllChem.AssignBondOrdersFromTemplate(template, pose)
    except (ValueError, RuntimeError):
        pose_h = Chem.RemoveHs(pose)
        template_h = Chem.RemoveHs(template)
        fixed_h = AllChem.AssignBondOrdersFromTemplate(template_h, pose_h)
        fixed = Chem.AddHs(fixed_h, addCoords=True)
    Chem.SanitizeMol(fixed)
    return fixed


def load_best_pose(docked_sdf: str, smiles: str) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(docked_sdf, removeHs=False)
    for mol in supplier:
        if mol is not None:
            return prepare_docked_ligand(mol, smiles)
    raise ValueError(f"No valid pose in {docked_sdf}")


def _interaction_name(name) -> str:
    if hasattr(name, "__name__"):
        return name.__name__
    return str(name)


def _residue_info(prot_res) -> dict:
    info = {"str": str(prot_res)}
    for attr in ("resname", "resid", "chain", "segid", "icode"):
        if hasattr(prot_res, attr):
            info[attr] = getattr(prot_res, attr)
    return info


def _is_pi_pi_stacking(interaction_name) -> bool:
    """True only for pi-pi stacking (excludes cation-pi, H-bond, hydrophobic, etc.)."""
    n = _interaction_name(interaction_name).lower().replace("-", "").replace("_", "")
    if "pication" in n or "cationpi" in n:
        return False
    return n == "pistacking" or ("pi" in n and "stack" in n)


def _is_tyr_residue(prot_res, resid: int) -> bool:
    """True if protein residue is TYR with the given residue number."""
    info = _residue_info(prot_res)
    if str(info.get("resname", "")).upper() == "TYR":
        try:
            if int(info["resid"]) == resid:
                return True
        except (ValueError, TypeError):
            pass
    s = info["str"].upper()
    if "TYR" not in s:
        return False
    if re.search(rf"TYR[^\d]*{resid}(?:[^\d]|$)", s):
        return True
    if re.search(rf"(?:^|[^\d]){resid}[^\d]*TYR", s):
        return True
    return f"TYR{resid}" in s.replace(" ", "").replace(".", "").replace(":", "")


def analyze_tyr_interactions(
    receptor_pdb: str,
    docked_sdf: str,
    smiles: str,
    tyr_residue: int = 56,
) -> Tuple[int, int, List[dict]]:
    """
    Run ProLIF and count pi-pi stacking at a specific TYR residue only.

    Returns (pi_pi_stacking_count, pi_pi_stacking_count, details_list).
    Both count values are identical (kept for backward compatibility).
    """
    protein = load_protein_for_prolif(receptor_pdb)
    ligand_mol = load_best_pose(docked_sdf, smiles)
    ligand = plf.Molecule.from_rdkit(ligand_mol)

    fp = make_fingerprint(plf, count=True)
    residues = tyr56_residue_ids(tyr_residue)
    ifp = run_fingerprint(fp, ligand, protein, residues=residues)

    interactions: List[dict] = []
    total = 0

    for lig_res, prot_res, interaction_dict in iter_ifp_pairs(ifp):
        if not _is_tyr_residue(prot_res, tyr_residue):
            continue
        n = count_interactions(interaction_dict, _is_pi_pi_stacking)
        if n > 0:
            total += n
            interactions.append({
                "ligand": str(lig_res),
                "protein": str(prot_res),
                "interaction": "PiStacking",
                "count": n,
            })

    return total, total, interactions


# ---------------------------------------------------------------------------
# Per-molecule pipeline
# ---------------------------------------------------------------------------

def _smiles_hash(smiles: str) -> str:
    return hashlib.md5(smiles.encode()).hexdigest()[:8]


def run_molecule_pipeline(
    smiles: str,
    mol_index: int,
    batch_id: int,
    config: GninaProlifConfig,
) -> MoleculeResult:
    """Dock one SMILES, analyze with ProLIF, return tiered rewards."""
    result = MoleculeResult(smiles=smiles)

    can = canonicalize(smiles)
    if can is None:
        result.error = "invalid_smiles"
        return result
    result.canonical_smiles = can

    work_dir = os.path.join(
        config.output_root,
        f"batch_{batch_id:06d}",
        f"mol_{mol_index:04d}_{_smiles_hash(can)}",
    )
    os.makedirs(work_dir, exist_ok=True)
    result.work_dir = work_dir

    lig_sdf = os.path.join(work_dir, "ligand_input.sdf")
    out_sdf = os.path.join(work_dir, "mol0_out.sdf")
    log_txt = os.path.join(work_dir, "mol0_log.txt")
    result.out_sdf = out_sdf
    result.log_file = log_txt

    if not embed_to_sdf(can, lig_sdf):
        result.error = "embedding_failed"
        return result

    if not run_gnina(lig_sdf, out_sdf, log_txt, config):
        result.error = "gnina_failed"
        if os.path.exists(lig_sdf):
            os.remove(lig_sdf)
        return result

    # Remove intermediate ligand file; keep mol0_out.sdf + mol0_log.txt
    if os.path.exists(lig_sdf):
        os.remove(lig_sdf)
    mol_file = lig_sdf.replace(".sdf", ".mol")
    if os.path.exists(mol_file):
        os.remove(mol_file)

    affinity = parse_affinity_from_log(log_txt)
    result.affinity = affinity
    result.docking_ok = np.isfinite(affinity)
    result.docking_reward = affinity_to_reward(affinity)

    try:
        tyr_count, pi_count, details = analyze_tyr_interactions(
            config.receptor_path, out_sdf, can, tyr_residue=config.tyr_residue
        )
        result.tyr_interaction_count = tyr_count
        result.tyr_pi_stacking_count = pi_count
        result.tyr_interactions = details
        result.tyr_interaction_reward = tyr_count_to_reward(tyr_count)
        result.prolif_ok = True
    except Exception as exc:
        logger.warning(f"[GninaBackend] ProLIF failed for {can[:40]}: {exc}")
        result.error = f"prolif_failed: {exc}"
        result.tyr_interaction_reward = 0.0

    if not config.keep_outputs:
        shutil.rmtree(work_dir, ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# Batch cache (shared across DockingScore + TyrosineInteraction)
# ---------------------------------------------------------------------------

class BatchCache:
    """Thread-safe per-batch result cache."""

    _lock = threading.Lock()
    _batch_sig: Optional[int] = None
    _batch_id: int = 0
    _results: Dict[str, MoleculeResult] = {}

    @classmethod
    def get_batch_id(cls, smiles_list: List[str]) -> int:
        sig = hash(tuple(smiles_list))
        with cls._lock:
            if sig != cls._batch_sig:
                cls._batch_sig = sig
                cls._batch_id += 1
                cls._results = {}
            return cls._batch_id

    @classmethod
    def get_or_run(
        cls,
        smiles_list: List[str],
        config: GninaProlifConfig,
    ) -> List[MoleculeResult]:
        batch_id = cls.get_batch_id(smiles_list)
        output: List[MoleculeResult] = []

        with cls._lock:
            for i, smi in enumerate(smiles_list):
                can = canonicalize(smi) or smi
                if can in cls._results:
                    output.append(cls._results[can])
                else:
                    res = run_molecule_pipeline(smi, i, batch_id, config)
                    cls._results[can] = res
                    output.append(res)

        return output
