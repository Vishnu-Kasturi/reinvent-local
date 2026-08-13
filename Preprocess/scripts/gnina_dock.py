#!/usr/bin/env python3
"""
gnina_dock.py — Batch Gnina docking for REINVENT4 on-the-fly scoring.

Embeds SMILES to 3D with RDKit, docks each ligand with Gnina against a
fixed receptor, and returns binding affinity scores (kcal/mol).

Can be used standalone (stdin/stdout JSON for ExternalProcess) or imported
by the GninaDock REINVENT scoring component.

Usage (standalone / ExternalProcess):
    echo -e "CCO\\nc1ccccc1" | python gnina_dock.py \\
        --receptor docking/receptor.pdbqt \\
        --center_x 10 --center_y 20 --center_z 30 \\
        --size_x 20 --size_y 20 --size_z 20

    # Or with autobox from a reference ligand:
    python gnina_dock.py --receptor docking/receptor.pdbqt \\
        --autobox_ligand docking/ref_ligand.sdf --autobox_add 4 < smiles.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

# Gnina table output: mode | affinity | intramol | CNN | CNNaffinity
_GNINA_TABLE_RE = re.compile(
    r"^\s*1\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)",
    re.MULTILINE,
)
_AFFINITY_LINE_RE = re.compile(r"Affinity:\s*(-?\d+\.?\d*)", re.IGNORECASE)


@dataclass
class GninaConfig:
    receptor_path: str
    gnina_executable: str = "gnina"
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    center_z: Optional[float] = None
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    autobox_ligand: Optional[str] = None
    autobox_add: float = 4.0
    exhaustiveness: int = 4
    num_modes: int = 1
    cnn_scoring: str = "rescore"
    docking_mode: str = "full"  # full | minimize | score_only
    seed: int = 42
    n_workers: int = 4
    use_gpu: bool = True
    cache_dir: Optional[str] = None
    timeout_sec: int = 120

    def validate(self) -> None:
        if not os.path.exists(self.receptor_path):
            raise FileNotFoundError(f"Receptor not found: {self.receptor_path}")
        has_center = all(v is not None for v in (self.center_x, self.center_y, self.center_z))
        if not has_center and not self.autobox_ligand:
            raise ValueError(
                "Provide either --center_x/y/z or --autobox_ligand for the search box."
            )
        if self.autobox_ligand and not os.path.exists(self.autobox_ligand):
            raise FileNotFoundError(f"Autobox ligand not found: {self.autobox_ligand}")


@dataclass
class DockResult:
    smiles: str
    affinity: float = float("nan")
    cnn_score: float = float("nan")
    cnn_affinity: float = float("nan")
    success: bool = False
    error: str = ""


def embed_smiles_to_sdf(smiles: str, sdf_path: str) -> bool:
    """Generate a 3D conformer and write to SDF. Returns True on success."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            return False
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        return True
    except Exception:
        return False


def _parse_gnina_output(stdout: str, stderr: str) -> Tuple[float, float, float]:
    """Extract (affinity, cnn_score, cnn_affinity) from gnina output."""
    combined = stdout + "\n" + stderr

    m = _GNINA_TABLE_RE.search(combined)
    if m:
        return float(m.group(1)), float(m.group(3)), float(m.group(4))

    m = _AFFINITY_LINE_RE.search(combined)
    if m:
        return float(m.group(1)), float("nan"), float("nan")

    return float("nan"), float("nan"), float("nan")


def _dock_single(args: Tuple[str, str, GninaConfig, str]) -> DockResult:
    """Worker function for parallel docking. args = (smiles, work_dir, config, idx)."""
    smiles, work_dir, config, idx = args
    result = DockResult(smiles=smiles)

    ligand_sdf = os.path.join(work_dir, f"lig_{idx}.sdf")
    out_sdf = os.path.join(work_dir, f"out_{idx}.sdf")
    log_path = os.path.join(work_dir, f"lig_{idx}_log.txt")

    if not embed_smiles_to_sdf(smiles, ligand_sdf):
        result.error = "3D embedding failed"
        return result

    cmd = [
        config.gnina_executable,
        "-r", os.path.abspath(config.receptor_path),
        "--ligand", ligand_sdf,
        "-o", out_sdf,
        "--log", log_path,
        "--num_modes", str(config.num_modes),
        "--exhaustiveness", str(config.exhaustiveness),
        "--seed", str(config.seed),
        "--cnn_scoring", config.cnn_scoring,
    ]

    if config.autobox_ligand:
        cmd += ["--autobox_ligand", os.path.abspath(config.autobox_ligand),
                "--autobox_add", str(config.autobox_add)]
    else:
        cmd += [
            "--center_x", str(config.center_x),
            "--center_y", str(config.center_y),
            "--center_z", str(config.center_z),
            "--size_x", str(config.size_x),
            "--size_y", str(config.size_y),
            "--size_z", str(config.size_z),
        ]

    if config.docking_mode == "minimize":
        cmd.append("--minimize")
    elif config.docking_mode == "score_only":
        cmd.append("--score_only")

    if not config.use_gpu:
        cmd.append("--cpu")
        cmd += ["--cpu", "4"]

    env = os.environ.copy()
    if not config.use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
            env=env,
        )
        # Parse from log file if written, else stdout/stderr
        log_text = ""
        if os.path.exists(log_path):
            with open(log_path) as lf:
                log_text = lf.read()
        affinity, cnn_score, cnn_affinity = _parse_gnina_output(
            log_text or proc.stdout, proc.stderr
        )

        if not np.isfinite(affinity):
            result.error = f"Could not parse gnina output (rc={proc.returncode})"
            return result

        result.affinity = affinity
        result.cnn_score = cnn_score
        result.cnn_affinity = cnn_affinity
        result.success = True
        return result

    except subprocess.TimeoutExpired:
        result.error = f"gnina timed out after {config.timeout_sec}s"
        return result
    except FileNotFoundError:
        result.error = f"gnina executable not found: {config.gnina_executable}"
        return result
    except Exception as exc:
        result.error = str(exc)
        return result


class GninaDocker:
    """Batch Gnina docker with optional score caching."""

    def __init__(self, config: GninaConfig):
        config.validate()
        self.config = config
        self._cache: Dict[str, DockResult] = {}
        if config.cache_dir:
            os.makedirs(config.cache_dir, exist_ok=True)

    def _canonical(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, canonical=True)

    def dock_batch(self, smiles_list: List[str]) -> List[DockResult]:
        """Dock a batch of SMILES, using cache where possible."""
        results: List[Optional[DockResult]] = [None] * len(smiles_list)
        to_dock: List[Tuple[int, str, str]] = []

        for i, smi in enumerate(smiles_list):
            can = self._canonical(smi)
            if can in self._cache:
                cached = self._cache[can]
                results[i] = DockResult(
                    smiles=smi,
                    affinity=cached.affinity,
                    cnn_score=cached.cnn_score,
                    cnn_affinity=cached.cnn_affinity,
                    success=cached.success,
                    error=cached.error,
                )
            else:
                to_dock.append((i, smi, can))

        if not to_dock:
            return results  # type: ignore[return-value]

        work_dir = tempfile.mkdtemp(prefix="gnina_batch_")
        try:
            n_workers = min(self.config.n_workers, len(to_dock))
            tasks = [
                (smi, work_dir, self.config, f"{idx}")
                for idx, (_, smi, _) in enumerate(to_dock)
            ]

            if n_workers <= 1:
                docked = [_dock_single(t) for t in tasks]
            else:
                docked_map = {}
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(_dock_single, t): i for i, t in enumerate(tasks)}
                    for fut in as_completed(futures):
                        docked_map[futures[fut]] = fut.result()
                docked = [docked_map[i] for i in range(len(tasks))]

            for (orig_idx, orig_smi, can), res in zip(to_dock, docked):
                res.smiles = orig_smi
                results[orig_idx] = res
                if res.success:
                    self._cache[can] = res

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        return results  # type: ignore[return-value]


def dock_smiles_batch(smiles_list: List[str], config: GninaConfig) -> List[DockResult]:
    """Convenience function for one-shot batch docking."""
    docker = GninaDocker(config)
    return docker.dock_batch(smiles_list)


def main():
    parser = argparse.ArgumentParser(description="Batch Gnina docking for REINVENT4")
    parser.add_argument("--receptor", required=True, help="Receptor PDBQT file")
    parser.add_argument("--gnina", default="gnina", help="Path to gnina executable")
    parser.add_argument("--center_x", type=float, default=None)
    parser.add_argument("--center_y", type=float, default=None)
    parser.add_argument("--center_z", type=float, default=None)
    parser.add_argument("--size_x", type=float, default=20.0)
    parser.add_argument("--size_y", type=float, default=20.0)
    parser.add_argument("--size_z", type=float, default=20.0)
    parser.add_argument("--autobox_ligand", default=None, help="Reference ligand for autobox")
    parser.add_argument("--autobox_add", type=float, default=4.0)
    parser.add_argument("--exhaustiveness", type=int, default=4)
    parser.add_argument("--num_modes", type=int, default=1)
    parser.add_argument("--cnn_scoring", default="rescore", choices=["none", "rescore", "refinement", "all"])
    parser.add_argument("--docking_mode", default="full", choices=["full", "minimize", "score_only"])
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    smiles_list = [line.strip() for line in sys.stdin if line.strip()]
    if not smiles_list:
        print(json.dumps({"version": 1, "payload": {"affinity": [], "cnn_score": [], "cnn_affinity": []}}))
        return

    config = GninaConfig(
        receptor_path=args.receptor,
        gnina_executable=args.gnina,
        center_x=args.center_x,
        center_y=args.center_y,
        center_z=args.center_z,
        size_x=args.size_x,
        size_y=args.size_y,
        size_z=args.size_z,
        autobox_ligand=args.autobox_ligand,
        autobox_add=args.autobox_add,
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        cnn_scoring=args.cnn_scoring,
        docking_mode=args.docking_mode,
        n_workers=args.n_workers,
        use_gpu=not args.no_gpu,
        cache_dir=args.cache_dir,
        timeout_sec=args.timeout,
    )

    results = dock_smiles_batch(smiles_list, config)

    affinities = [r.affinity if r.success else None for r in results]
    cnn_scores = [r.cnn_score if r.success else None for r in results]
    cnn_affinities = [r.cnn_affinity if r.success else None for r in results]

    payload = {
        "affinity": affinities,
        "cnn_score": cnn_scores,
        "cnn_affinity": cnn_affinities,
        "GninaAffinity": affinities,
    }
    print(json.dumps({"version": 1, "payload": payload}))


if __name__ == "__main__":
    main()
