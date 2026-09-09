"""
dock_prolif_backend.py — Shared smina/GNINA docking + ProLIF TYR56 analysis.

One dock run per molecule supplies both tiered docking and tyrosine rewards
(REINVENT iict_new_reward.toml style). Results are cached per RL batch.
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
import inspect
from typing import Any, Dict, Iterator, List, Optional, Tuple

import MDAnalysis as mda
import numpy as np
import prolif as plf
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from reinvent_transforms import transform_docking, transform_tyrosine


# --- prolif_compat (inlined so only this file is needed beside vis_rl.py) ---

def make_fingerprint(plf_mod, count: bool = True):
    try:
        return plf_mod.Fingerprint(count=count)
    except TypeError:
        return plf_mod.Fingerprint()


def _generate_v2(fp, ligand, protein, residues: Optional[List[str]]):
    sig = inspect.signature(fp.generate)
    kwargs: dict[str, Any] = {}
    if residues is not None and "residues" in sig.parameters:
        kwargs["residues"] = residues
    if "metadata" in sig.parameters:
        kwargs["metadata"] = True
    return fp.generate(ligand, protein, **kwargs)


def _run_from_iterable_v2(fp, ligand, protein, residues: Optional[List[str]], frame: int = 0):
    if residues is not None:
        fp.run_from_iterable([ligand], protein, residues=residues)
    else:
        fp.run_from_iterable([ligand], protein)
    return get_ifp_from_fingerprint(fp, frame=frame)


def get_ifp_from_fingerprint(fp, frame: int = 0):
    ifp = fp.ifp
    if isinstance(ifp, dict) and ifp and all(isinstance(k, int) for k in ifp):
        return ifp[frame]
    return ifp


def run_fingerprint(fp, ligand, protein, residues: Optional[List[str]] = None, frame: int = 0):
    if hasattr(fp, "generate"):
        try:
            return _generate_v2(fp, ligand, protein, residues)
        except TypeError:
            pass
    if hasattr(fp, "run_from_iterable"):
        try:
            return _run_from_iterable_v2(fp, ligand, protein, residues, frame=frame)
        except TypeError:
            return _run_from_iterable_v2(fp, ligand, protein, None, frame=frame)
    if residues is not None:
        try:
            fp.run(ligand, protein, residues=residues)
        except TypeError:
            fp.run(ligand, protein)
    else:
        fp.run(ligand, protein)
    return fp.ifp


def iter_ifp_pairs(ifp) -> Iterator[Tuple[Any, Any, dict]]:
    for key, ix_dict in ifp.items():
        if isinstance(key, tuple) and len(key) == 2:
            lig_res, prot_res = key
            yield lig_res, prot_res, ix_dict


def ifp_to_dataframe(plf_mod, fp, ifp):
    if hasattr(plf_mod, "to_dataframe"):
        try:
            interactions = getattr(fp, "interactions", None)
            if interactions is not None:
                return plf_mod.to_dataframe({0: ifp}, interactions)
            return plf_mod.to_dataframe({0: ifp})
        except TypeError:
            pass
        try:
            return plf_mod.to_dataframe(ifp)
        except TypeError:
            pass
    if hasattr(fp, "to_dataframe"):
        return fp.to_dataframe(ifp)
    raise TypeError("No compatible to_dataframe API found")


def count_interactions(ix_dict, predicate) -> int:
    total = 0
    for name, metadata in ix_dict.items():
        if not predicate(name):
            continue
        if metadata is None:
            continue
        if isinstance(metadata, (list, tuple)):
            total += len(metadata)
        elif isinstance(metadata, dict):
            total += max(len(metadata), 1) if metadata else 0
        else:
            total += 1
    return total


def residue_ids(resname: str, resid: int, chains: str = "AB") -> List[str]:
    return [f"{resname.upper()}{resid}.{c}" for c in chains]


def tyr56_residue_ids(tyr_resid: int, chains: str = "AB") -> List[str]:
    return residue_ids("TYR", tyr_resid, chains=chains)

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger("reinvent")

_AFFINITY_RE = re.compile(
    r"^\s*1\s+(-?\d+(?:\.\d+)?)", re.MULTILINE
)


@dataclass
class GninaProlifConfig:
    receptor_path: str
    autobox_ligand: str
    smina_executable: Optional[str] = None
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
        if not (
            (self.smina_executable and os.path.isfile(self.smina_executable))
            or self.gnina_executable
        ):
            raise FileNotFoundError(
                "No docking executable: set smina.static or gnina in docking_path"
            )
        os.makedirs(self.output_root, exist_ok=True)


def _find_docking_file(base_dir: str, candidates: Tuple[str, ...]) -> Optional[str]:
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            return path
    return None


def build_config_from_docking_path(
    docking_path: str,
    output_root: Optional[str] = None,
    tyr_residue: int = 56,
) -> GninaProlifConfig:
    """Build config from vis_rl docking_path folder (smina + receptor + autobox)."""
    if not docking_path.endswith("/"):
        docking_path = docking_path + "/"

    receptor = _find_docking_file(
        docking_path,
        ("receptor.pdb", "receptor.pdbqt", "protein.pdb", "protein.pdbqt"),
    )
    autobox = _find_docking_file(
        docking_path,
        (
            "autobox_ligand.sdf",
            "crystal_ligand.sdf",
            "ligand.sdf",
            "ref_ligand.sdf",
            "ref_ligand.pdb",
            "docked_ligand.sdf",
        ),
    )
    smina = os.path.join(docking_path, "smina.static")
    gnina = _find_docking_file(docking_path, ("gnina",)) or "gnina"

    if receptor is None:
        raise FileNotFoundError(f"receptor not found in {docking_path}")
    if autobox is None:
        raise FileNotFoundError(f"autobox reference ligand not found in {docking_path}")

    return GninaProlifConfig(
        receptor_path=receptor,
        autobox_ligand=autobox,
        smina_executable=smina if os.path.isfile(smina) else None,
        gnina_executable=gnina,
        output_root=output_root or os.path.join(docking_path, "dock_prolif_runs"),
        tyr_residue=tyr_residue,
    )


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


def run_smina(
    ligand_sdf: str,
    out_sdf: str,
    log_path: str,
    config: GninaProlifConfig,
) -> bool:
    cmd = [
        config.smina_executable,
        "--receptor", os.path.abspath(config.receptor_path),
        "--ligand", ligand_sdf,
        "--autobox_ligand", os.path.abspath(config.autobox_ligand),
        "--out", out_sdf,
        "--log", log_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=config.timeout_sec,
                       capture_output=True, text=True)
        return os.path.exists(out_sdf) and os.path.exists(log_path)
    except Exception as exc:
        logger.warning(f"[DockProlif] smina failed: {exc}")
        return False


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
        logger.warning(f"[DockProlif] gnina failed: {exc}")
        return False


def run_docking(
    ligand_sdf: str,
    out_sdf: str,
    log_path: str,
    config: GninaProlifConfig,
) -> bool:
    if config.smina_executable and os.path.isfile(config.smina_executable):
        return run_smina(ligand_sdf, out_sdf, log_path, config)
    return run_gnina(ligand_sdf, out_sdf, log_path, config)


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

    if not run_docking(lig_sdf, out_sdf, log_txt, config):
        result.error = "docking_failed"
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
    result.docking_reward = transform_docking(affinity)

    try:
        tyr_count, pi_count, details = analyze_tyr_interactions(
            config.receptor_path, out_sdf, can, tyr_residue=config.tyr_residue
        )
        result.tyr_interaction_count = tyr_count
        result.tyr_pi_stacking_count = pi_count
        result.tyr_interactions = details
        result.tyr_interaction_reward = transform_tyrosine(tyr_count)
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


class SingleMoleculeCache:
    """Per-RL-batch cache: dock once per canonical SMILES for docking + tyrosine."""

    _config: Optional[GninaProlifConfig] = None
    _batch_id: int = 0
    _results: Dict[str, MoleculeResult] = {}

    @classmethod
    def configure(cls, config: GninaProlifConfig) -> None:
        cls._config = config

    @classmethod
    def clear(cls) -> None:
        cls._results = {}
        cls._batch_id += 1

    @classmethod
    def get_or_run(cls, smiles: str, mol_index: int, output_root: str) -> MoleculeResult:
        if cls._config is None:
            raise RuntimeError("SingleMoleculeCache.configure() must be called first")

        can = canonicalize(smiles)
        if can is None:
            return MoleculeResult(smiles=smiles, error="invalid_smiles")

        if can in cls._results:
            return cls._results[can]

        config = cls._config
        if output_root:
            config = GninaProlifConfig(
                receptor_path=config.receptor_path,
                autobox_ligand=config.autobox_ligand,
                smina_executable=config.smina_executable,
                gnina_executable=config.gnina_executable,
                output_root=output_root,
                cnn_scoring=config.cnn_scoring,
                timeout_sec=config.timeout_sec,
                keep_outputs=config.keep_outputs,
                tyr_residue=config.tyr_residue,
            )

        result = run_molecule_pipeline(smiles, mol_index, cls._batch_id, config)
        cls._results[can] = result
        return result


class SingleMoleculeCache:
    """Per-RL-batch cache: dock once per canonical SMILES for docking + tyrosine."""

    _config: Optional[GninaProlifConfig] = None
    _batch_id: int = 0
    _results: Dict[str, MoleculeResult] = {}

    @classmethod
    def configure(cls, config: GninaProlifConfig) -> None:
        cls._config = config

    @classmethod
    def clear(cls) -> None:
        cls._results = {}
        cls._batch_id += 1

    @classmethod
    def get_or_run(cls, smiles: str, mol_index: int, output_root: str) -> MoleculeResult:
        if cls._config is None:
            raise RuntimeError("SingleMoleculeCache.configure() must be called first")

        can = canonicalize(smiles)
        if can is None:
            bad = MoleculeResult(smiles=smiles, error="invalid_smiles")
            return bad

        if can in cls._results:
            return cls._results[can]

        config = cls._config
        if output_root:
            config = GninaProlifConfig(
                receptor_path=config.receptor_path,
                autobox_ligand=config.autobox_ligand,
                smina_executable=config.smina_executable,
                gnina_executable=config.gnina_executable,
                output_root=output_root,
                cnn_scoring=config.cnn_scoring,
                timeout_sec=config.timeout_sec,
                keep_outputs=config.keep_outputs,
                tyr_residue=config.tyr_residue,
            )

        result = run_molecule_pipeline(smiles, mol_index, cls._batch_id, config)
        cls._results[can] = result
        return result
