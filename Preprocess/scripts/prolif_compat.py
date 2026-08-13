"""
prolif_compat.py — ProLIF v1/v2 API compatibility helpers.
"""
from __future__ import annotations

import inspect
from typing import Any, Iterator, List, Optional, Tuple


def prolif_version(plf) -> str:
    return getattr(plf, "__version__", "unknown")


def make_fingerprint(plf, count: bool = True):
    """count=True required to detect multiple pi-pi stacks on same residue."""
    try:
        return plf.Fingerprint(count=count)
    except TypeError:
        return plf.Fingerprint()


def _generate_v2(fp, ligand, protein, residues: Optional[List[str]]):
    """ProLIF v2: generate() returns IFP; metadata=True for interaction details."""
    sig = inspect.signature(fp.generate)
    kwargs: dict[str, Any] = {}
    if residues is not None and "residues" in sig.parameters:
        kwargs["residues"] = residues
    if "metadata" in sig.parameters:
        kwargs["metadata"] = True
    return fp.generate(ligand, protein, **kwargs)


def _run_from_iterable_v2(fp, ligand, protein, residues: Optional[List[str]], frame: int = 0):
    """ProLIF v2: run_from_iterable() stores frame-indexed results on fp.ifp."""
    if residues is not None:
        fp.run_from_iterable([ligand], protein, residues=residues)
    else:
        fp.run_from_iterable([ligand], protein)
    return get_ifp_from_fingerprint(fp, frame=frame)


def get_ifp_from_fingerprint(fp, frame: int = 0):
    """Extract IFP from a Fingerprint object (v1 direct, v2 frame-indexed)."""
    ifp = fp.ifp
    if isinstance(ifp, dict) and ifp and all(isinstance(k, int) for k in ifp):
        return ifp[frame]
    return ifp


def run_fingerprint(
    fp,
    ligand,
    protein,
    residues: Optional[List[str]] = None,
    frame: int = 0,
):
    """
    Run ProLIF across v1 and v2 APIs and return the interaction fingerprint.

    v2: fp.generate(lig, prot, metadata=True) — returns IFP directly
    v2: fp.run_from_iterable([lig], prot) — stores {frame: IFP} on fp.ifp
    v1: fp.run(lig, prot) — stores IFP on fp.ifp
    """
    # ProLIF v2: generate() for single structure pair
    if hasattr(fp, "generate"):
        try:
            return _generate_v2(fp, ligand, protein, residues)
        except TypeError:
            pass

    # ProLIF v2: run_from_iterable
    if hasattr(fp, "run_from_iterable"):
        try:
            return _run_from_iterable_v2(fp, ligand, protein, residues, frame=frame)
        except TypeError:
            return _run_from_iterable_v2(fp, ligand, protein, None, frame=frame)

    # ProLIF v1
    if residues is not None:
        try:
            fp.run(ligand, protein, residues=residues)
        except TypeError:
            fp.run(ligand, protein)
    else:
        fp.run(ligand, protein)
    return fp.ifp


def iter_ifp_pairs(ifp) -> Iterator[Tuple[Any, Any, dict]]:
    """Iterate (lig_res, prot_res, interaction_dict) from any IFP format."""
    for key, ix_dict in ifp.items():
        if isinstance(key, tuple) and len(key) == 2:
            lig_res, prot_res = key
            yield lig_res, prot_res, ix_dict


def ifp_to_dataframe(plf, fp, ifp):
    """Export IFP to pandas DataFrame across ProLIF versions."""
    # v2 module-level helper
    if hasattr(plf, "to_dataframe"):
        try:
            interactions = getattr(fp, "interactions", None)
            if interactions is not None:
                return plf.to_dataframe({0: ifp}, interactions)
            return plf.to_dataframe({0: ifp})
        except TypeError:
            pass
        try:
            return plf.to_dataframe(ifp)
        except TypeError:
            pass

    # v1 method on Fingerprint
    if hasattr(fp, "to_dataframe"):
        return fp.to_dataframe(ifp)

    raise TypeError("No compatible to_dataframe API found")


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
