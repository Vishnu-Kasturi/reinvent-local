"""
prolif_compat.py — ProLIF v1/v2 API compatibility helpers.
"""
from __future__ import annotations

import inspect
from typing import Any, List, Optional


def prolif_version(plf) -> str:
    return getattr(plf, "__version__", "unknown")


def make_fingerprint(plf, count: bool = True):
    """count=True required to detect multiple pi-pi stacks on same residue."""
    try:
        return plf.Fingerprint(count=count)
    except TypeError:
        return plf.Fingerprint()


def run_fingerprint(fp, ligand, protein, residues: Optional[List[str]] = None):
    """
    Run ProLIF across v1 and v2 APIs.

    v2: fp.generate(lig, prot) or fp.run_from_iterable([lig], prot)
    v1: fp.run(lig, prot)
    """
    kwargs = {}
    if residues is not None:
        kwargs["residues"] = residues

    # ProLIF v2: generate() for single structure pair
    if hasattr(fp, "generate"):
        try:
            sig = inspect.signature(fp.generate)
            if residues is not None and "residues" in sig.parameters:
                return fp.generate(ligand, protein, residues=residues)
            return fp.generate(ligand, protein)
        except TypeError:
            pass

    # ProLIF v2: run_from_iterable
    if hasattr(fp, "run_from_iterable"):
        try:
            if residues is not None:
                return fp.run_from_iterable([ligand], protein, residues=residues)
            return fp.run_from_iterable([ligand], protein)
        except TypeError:
            return fp.run_from_iterable([ligand], protein)

    # ProLIF v1
    return fp.run(ligand, protein)


def count_interactions(ix_dict, predicate) -> int:
    """Count interactions; handles count=True metadata (lists/tuples)."""
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


def tyr56_residue_ids(tyr_resid: int, chains: str = "AB") -> List[str]:
    """ProLIF residue ID strings for TYR on multiple chains."""
    return [f"TYR{tyr_resid}.{c}" for c in chains]
