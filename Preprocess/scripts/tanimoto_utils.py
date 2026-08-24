#!/usr/bin/env python3
"""Shared Tanimoto helpers — SMILES-only CSV reads, JAK2 6–11 reference set."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

DEFAULT_JAK2_PREPROCESS = (
    _REPO_ROOT / "Preprocess" / "Data_jak2" / "data_csvs" / "jak2_preprocess_all.csv"
)
DEFAULT_JAK2_TRAIN_SMI = _REPO_ROOT / "data" / "jak2_TL_train.smi"

SMILES_COLUMN_NAMES = (
    "smiles",
    "canonical_smiles",
    "input_smiles",
    "generated_smiles",
    "optimized_smiles",
    "lead_smiles",
)


def _detect_sep(path: Path) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
    if header.count(";") > header.count(",") and header.count(";") > header.count("\t"):
        return ";"
    if header.count("\t") > header.count(","):
        return "\t"
    return ","


def find_smiles_column(columns: Iterable[str]) -> Optional[str]:
    lower_map = {c.strip().lower(): c for c in columns}
    for name in SMILES_COLUMN_NAMES:
        if name in lower_map:
            return lower_map[name]
    return None


def read_smiles_from_csv(path: str | Path) -> list[str]:
    """
    Read only the SMILES column from a CSV/TSV (ignores all other columns).

    Works when the file has many REINVENT reward columns — only SMILES is used.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    sep = _detect_sep(path)
    header = pd.read_csv(path, sep=sep, nrows=0)
    smi_col = find_smiles_column(header.columns)
    if smi_col is None:
        raise ValueError(
            f"No SMILES column in {path}. "
            f"Expected one of {SMILES_COLUMN_NAMES}; got {list(header.columns)}"
        )

    series = pd.read_csv(path, sep=sep, usecols=[smi_col], dtype=str)[smi_col]
    smiles = series.dropna().astype(str).str.strip()
    smiles = smiles[smiles != ""].tolist()
    return smiles


def read_smiles_from_smi(path: str | Path) -> list[str]:
    path = Path(path).expanduser().resolve()
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [line.strip() for line in fh if line.strip()]


def load_jak2_reference_smiles(
    min_pic50: float = 6.0,
    max_pic50: float = 11.0,
    preprocess_csv: str | Path | None = None,
    train_smi: str | Path | None = None,
) -> list[str]:
    """
    JAK2 reference molecules for Tanimoto (pIC50 in [6, 11] by default).

    Prefers jak2_preprocess_all.csv when available; falls back to jak2_TL_train.smi.
    """
    csv_path = Path(preprocess_csv or DEFAULT_JAK2_PREPROCESS).expanduser().resolve()
    if csv_path.exists():
        sep = _detect_sep(csv_path)
        df = pd.read_csv(csv_path, sep=sep, usecols=lambda c: c.strip().lower() in {"smiles", "pic50"})
        df.columns = [c.strip().lower() for c in df.columns]
        if "smiles" not in df.columns or "pic50" not in df.columns:
            raise ValueError(f"{csv_path} must contain smiles and pic50 columns")
        df = df.dropna(subset=["smiles", "pic50"])
        df["pic50"] = pd.to_numeric(df["pic50"], errors="coerce")
        df = df[(df["pic50"] >= min_pic50) & (df["pic50"] <= max_pic50)]
        return df["smiles"].astype(str).str.strip().tolist()

    smi_path = Path(train_smi or DEFAULT_JAK2_TRAIN_SMI).expanduser().resolve()
    if smi_path.exists():
        return read_smiles_from_smi(smi_path)

    raise FileNotFoundError(
        "JAK2 reference not found. Run prepare_tl_smiles.py or provide --reference."
    )


def build_morgan_fps(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[list, list[str]]:
    fps: list = []
    valid_smiles: list[str] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
        valid_smiles.append(str(smi))
    return fps, valid_smiles


def max_tanimoto_per_molecule(query_fps: list, ref_fps: list) -> list[float]:
    if not ref_fps:
        return [0.0] * len(query_fps)
    scores: list[float] = []
    for fp in query_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        scores.append(float(max(sims)) if sims else 0.0)
    return scores


def mean_tanimoto_per_molecule(query_fps: list, ref_fps: list) -> list[float]:
    if not ref_fps:
        return [0.0] * len(query_fps)
    scores: list[float] = []
    for fp in query_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        scores.append(float(np.mean(sims)) if sims else 0.0)
    return scores


def load_reference_fps(
    target: str = "jak2",
    reference_path: str | Path | None = None,
    min_pic50: float = 6.0,
    max_pic50: float = 11.0,
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[list, list[str], str]:
    """Load reference fingerprints and return (fps, smiles, label)."""
    if reference_path:
        ref_path = Path(reference_path).expanduser().resolve()
        if ref_path.suffix.lower() == ".smi":
            ref_smiles = read_smiles_from_smi(ref_path)
            label = str(ref_path)
        elif "jak2_preprocess" in ref_path.name.lower() or ref_path.name == "jak2_preprocess_all.csv":
            ref_smiles = load_jak2_reference_smiles(
                min_pic50=min_pic50,
                max_pic50=max_pic50,
                preprocess_csv=ref_path,
            )
            label = f"JAK2 pIC50 [{min_pic50}, {max_pic50}] ({ref_path.name})"
        else:
            ref_smiles = read_smiles_from_csv(ref_path)
            label = str(ref_path)
    elif target.lower() == "jak2":
        ref_smiles = load_jak2_reference_smiles(min_pic50=min_pic50, max_pic50=max_pic50)
        label = f"JAK2 pIC50 [{min_pic50}, {max_pic50}] ({DEFAULT_JAK2_PREPROCESS.name})"
    else:
        raise ValueError(f"Unknown target {target!r}; pass --reference explicitly")

    ref_fps, ref_valid = build_morgan_fps(ref_smiles, radius=radius, n_bits=n_bits)
    return ref_fps, ref_valid, label
