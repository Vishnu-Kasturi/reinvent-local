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
    residue_ids,
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
    """Load receptor for ProLIF, handling partial CONECT / valence issues."""
    cleaned = _strip_conect(receptor_pdb)
    errors = []
    try:
        for guess_bonds in (True, False):
            try:
                u = mda.Universe(cleaned)
                ag = u.select_atoms("protein and not resname HOH WAT TIP3 SOL CL NA K MG CA ZN")
                if len(ag) == 0:
                    ag = u.atoms
                if guess_bonds:
                    ag.guess_bonds()
                return plf.Molecule.from_mda(ag)
            except Exception as exc:
                errors.append(f"MDAnalysis(guess_bonds={guess_bonds}): {exc}")
    except Exception as exc:
        errors.append(f"MDAnalysis setup: {exc}")
    finally:
        if os.path.exists(cleaned):
            os.unlink(cleaned)

    try:
        rdmol = Chem.MolFromPDBFile(receptor_pdb, removeHs=False, sanitize=False)
        if rdmol is not None:
            try:
                Chem.SanitizeMol(
                    rdmol,
                    sanitizeOps=Chem.SANITIZE_SETAROMATICITY | Chem.SANITIZE_SYMMRINGS,
                )
            except Exception:
                pass
            return plf.Molecule.from_rdkit(rdmol)
    except Exception as exc:
        errors.append(f"RDKit unsanitized: {exc}")

    raise RuntimeError(
        f"Cannot load receptor: {receptor_pdb}\n  " + "\n  ".join(errors)
    )


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
    if not os.path.isfile(docked_sdf):
        raise ValueError(f"Pose file not found: {docked_sdf}")
    if os.path.getsize(docked_sdf) < 20:
        raise ValueError(f"Pose file empty or truncated: {docked_sdf}")
    try:
        supplier = Chem.SDMolSupplier(docked_sdf, removeHs=False)
    except OSError as exc:
        raise ValueError(f"Invalid SDF file {docked_sdf}: {exc}") from exc
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
    return n in ("pistacking", "facetoface", "edgetoface") or (
        "pi" in n and "stack" in n
    )


def _is_residue(prot_res, resname: str, resid: int) -> bool:
    """True if protein residue matches resname + residue number."""
    info = _residue_info(prot_res)
    if str(info.get("resname", "")).upper() == resname.upper():
        try:
            if int(info["resid"]) == resid:
                return True
        except (ValueError, TypeError):
            pass
    s = info["str"].upper()
    rn = resname.upper()
    if rn not in s:
        return False
    if re.search(rf"{rn}[^\d]*{resid}(?:[^\d]|$)", s):
        return True
    if re.search(rf"(?:^|[^\d]){resid}[^\d]*{rn}", s):
        return True
    return f"{rn}{resid}" in s.replace(" ", "").replace(".", "").replace(":", "")


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


def _is_asp_polar_interaction(interaction_name) -> bool:
    """
    ProLIF v2 polar contacts at ASP: HBDonor, HBAcceptor, Anionic, Cationic, etc.
    (ProLIF v1 used HBond / SaltBridge class names.)
    """
    n = _interaction_name(interaction_name).lower().replace("-", "").replace("_", "")
    if "hydrophobic" in n or "vdw" in n:
        return False
    if "pistacking" in n or "facetoface" in n or "edgetoface" in n:
        return False
    if "pi" in n and "stack" in n:
        return False
    return any(
        k in n
        for k in (
            "hbond",
            "hbdonor",
            "hbacceptor",
            "hdonor",
            "hacceptor",
            "hydrogenbond",
            "implicit",
            "saltbridge",
            "salt",
            "ionic",
            "anionic",
            "cationic",
            "cationpi",
            "pication",
        )
    )


# Backward-compatible alias
_is_asp_small_mol_interaction = _is_asp_polar_interaction


def _format_prolif_residue(prot_res) -> str:
    """ProLIF protein residue label, e.g. ASP122.A."""
    info = _residue_info(prot_res)
    rn = str(info.get("resname", "") or "").upper()
    resid = info.get("resid", "")
    chain = info.get("chain") or info.get("segid") or ""
    if rn and resid != "":
        label = f"{rn}{resid}"
        if chain:
            label += f".{chain}"
        return label
    return str(info["str"])


def classify_asp_interaction_bucket(interaction_name) -> str:
    """
    Map ProLIF interaction class → ASP122 report bucket.

    ProLIF v2: HBDonor, HBAcceptor, Anionic, Cationic, CationPi, PiCation, …
    ProLIF v1: HBond, SaltBridge, PiStacking, Hydrophobic, …
    """
    n = _interaction_name(interaction_name).lower().replace("-", "").replace("_", "")
    if "saltbridge" in n or n == "saltbridge":
        return "salt_bridge"
    if "cationpi" in n:
        return "cation_pi"
    if "pication" in n:
        return "pi_cation"
    if "anionic" in n:
        return "anionic"
    if any(k in n for k in ("hbdonor", "hbacceptor", "hbond", "hdonor", "hacceptor", "implicithb")):
        return "h_bond"
    if "cationic" in n:
        return "cationic"
    if "hydrophobic" in n:
        return "hydrophobic"
    if "pistacking" in n or "facetoface" in n or "edgetoface" in n or ("pi" in n and "stack" in n):
        return "pi_stacking"
    if "vdw" in n:
        return "vdw"
    return "other"


@dataclass
class AspInteractionResult:
    """Typed ProLIF interaction counts at ASP122."""

    h_bond: int = 0
    salt_bridge: int = 0
    anionic: int = 0
    cation_pi: int = 0
    pi_cation: int = 0
    hydrophobic: int = 0
    pi_stacking: int = 0
    vdw: int = 0
    cationic: int = 0
    other: int = 0
    total_interactions: int = 0
    all_contacts: int = 0
    residues: List[str] = field(default_factory=list)
    interaction_rows: List[dict] = field(default_factory=list)
    prolif_table: str = ""

    @property
    def residues_str(self) -> str:
        return "; ".join(self.residues)

    def as_dict(self) -> dict:
        return {
            "h_bond": self.h_bond,
            "salt_bridge": self.salt_bridge,
            "anionic": self.anionic,
            "cation_pi": self.cation_pi,
            "pi_cation": self.pi_cation,
            "total_interactions": self.total_interactions,
            "all_contacts": self.all_contacts,
            "residues": self.residues_str,
        }


def _asp_bucket_field(bucket: str) -> Optional[str]:
    mapping = {
        "h_bond": "h_bond",
        "salt_bridge": "salt_bridge",
        "anionic": "anionic",
        "cation_pi": "cation_pi",
        "pi_cation": "pi_cation",
        "hydrophobic": "hydrophobic",
        "pi_stacking": "pi_stacking",
        "vdw": "vdw",
        "cationic": "cationic",
        "other": "other",
    }
    return mapping.get(bucket)


def _count_interaction_metadata(metadata) -> int:
    if metadata is None:
        return 0
    if isinstance(metadata, (list, tuple)):
        return len(metadata)
    if isinstance(metadata, dict):
        return max(len(metadata), 1) if metadata else 0
    return 1


def _asp_contact_category(interaction_name) -> str:
    """Short label for verbose ASP122 breakdown."""
    n = _interaction_name(interaction_name).lower().replace("-", "").replace("_", "")
    if _is_asp_polar_interaction(interaction_name):
        return "polar"
    if "hydrophobic" in n:
        return "hydrophobic"
    if "pistacking" in n or "facetoface" in n or "edgetoface" in n or ("pi" in n and "stack" in n):
        return "pistacking"
    if "vdw" in n:
        return "vdw"
    return _interaction_name(interaction_name)


def _count_by_category(interaction_dict: dict) -> dict[str, int]:
    """Sum ProLIF contacts at one ligand–ASP pair by category."""
    totals: dict[str, int] = {}
    for name, metadata in interaction_dict.items():
        if metadata is None:
            continue
        cnt = len(metadata) if isinstance(metadata, (list, tuple)) else 1
        if cnt <= 0:
            continue
        cat = _asp_contact_category(name)
        totals[cat] = totals.get(cat, 0) + cnt
    return totals


def _format_asp_breakdown(breakdown: dict[str, int], polar: int) -> str:
    parts = [f"{k}={v}" for k, v in sorted(breakdown.items()) if k != "polar"]
    if polar:
        parts.append(f"polar={polar}")
    return ", ".join(parts) if parts else "none"


def format_asp_prolif_summary(
    mol_id: int,
    smiles: str,
    sdf_path: str,
    asp_residue: int,
    result: AspInteractionResult,
    prolif_table: str = "",
) -> str:
    """Human-readable ProLIF summary block for one molecule."""
    lines = [
        "=" * 72,
        f"molID {mol_id}",
        f"SMILES: {smiles}",
        f"Pose SDF: {sdf_path or '(none)'}",
        f"Target: ASP{asp_residue} (homodimer chains A/B)",
        f"Residues with contacts: {result.residues_str or 'none'}",
        "",
        "ASP122 interaction counts:",
        f"  HBond:        {result.h_bond}",
        f"  SaltBridge:   {result.salt_bridge}",
        f"  Anionic:      {result.anionic}",
        f"  CationPi:     {result.cation_pi}",
        f"  PiCation:     {result.pi_cation}",
        f"  TOTAL:        {result.total_interactions}",
        "",
        "Other ProLIF contacts at ASP122:",
        f"  Hydrophobic:  {result.hydrophobic}",
        f"  PiStacking:   {result.pi_stacking}",
        f"  VdW:          {result.vdw}",
        f"  Cationic:     {result.cationic}",
        f"  Other:        {result.other}",
        f"  All contacts: {result.all_contacts}",
        "",
        "Detailed interactions:",
    ]
    if result.interaction_rows:
        for row in result.interaction_rows:
            lines.append(
                f"  {row['protein']:16s} | {row['interaction']:14s} | "
                f"n={row['count']:2d} | ligand={row['ligand']}"
            )
    else:
        lines.append("  (none)")
    if prolif_table:
        lines.extend(["", "ProLIF dataframe:", prolif_table])
    lines.append("")
    return "\n".join(lines)


def _prolif_ifp_to_text(plf, fp, ifp) -> str:
    from prolif_compat import ifp_to_dataframe
    try:
        df = ifp_to_dataframe(plf, fp, ifp)
        return df.to_string()
    except Exception as exc:
        return f"(dataframe export unavailable: {exc})"


def analyze_asp_interactions(
    receptor_pdb: str,
    docked_sdf: str,
    smiles: str,
    asp_residue: int = 122,
    chains: str = "AB",
) -> AspInteractionResult:
    """
    ProLIF contacts at ASP122 with per-type counts.

    total_interactions = HBond + SaltBridge + Anionic + CationPi + PiCation
    all_contacts       = every ProLIF contact type at ASP122
    """
    protein = load_protein_for_prolif(receptor_pdb)
    ligand_mol = load_best_pose(docked_sdf, smiles)
    ligand = plf.Molecule.from_rdkit(ligand_mol)

    fp = make_fingerprint(plf, count=True)
    residues = residue_ids("ASP", asp_residue, chains=chains)
    ifp = run_fingerprint(fp, ligand, protein, residues=residues)

    result = AspInteractionResult()
    residue_set: set[str] = set()

    for lig_res, prot_res, interaction_dict in iter_ifp_pairs(ifp):
        if not _is_residue(prot_res, "ASP", asp_residue):
            continue
        prot_label = _format_prolif_residue(prot_res)
        for name, metadata in interaction_dict.items():
            cnt = _count_interaction_metadata(metadata)
            if cnt <= 0:
                continue
            bucket = classify_asp_interaction_bucket(name)
            field = _asp_bucket_field(bucket)
            if field:
                setattr(result, field, getattr(result, field) + cnt)
            result.all_contacts += cnt
            residue_set.add(prot_label)
            result.interaction_rows.append({
                "ligand": str(lig_res),
                "protein": prot_label,
                "interaction": _interaction_name(name),
                "bucket": bucket,
                "count": cnt,
            })

    result.residues = sorted(residue_set)
    result.total_interactions = (
        result.h_bond
        + result.salt_bridge
        + result.anionic
        + result.cation_pi
        + result.pi_cation
    )
    result.prolif_table = _prolif_ifp_to_text(plf, fp, ifp)
    return result


def analyze_asp_interactions_legacy(
    receptor_pdb: str,
    docked_sdf: str,
    smiles: str,
    asp_residue: int = 122,
    chains: str = "AB",
) -> Tuple[int, int, List[dict], dict[str, int]]:
    """Backward-compatible tuple return for older callers."""
    result = analyze_asp_interactions(
        receptor_pdb, docked_sdf, smiles, asp_residue=asp_residue, chains=chains
    )
    breakdown: dict[str, int] = {}
    for key in ("hydrophobic", "pi_stacking", "vdw", "other"):
        val = getattr(result, key, 0)
        if val:
            breakdown[key] = val
    polar = result.total_interactions
    details = [
        row for row in result.interaction_rows
        if row["bucket"] in {"h_bond", "salt_bridge", "anionic", "cation_pi", "pi_cation"}
    ]
    return result.all_contacts, polar, details, breakdown


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
