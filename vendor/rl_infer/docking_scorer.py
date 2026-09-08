"""Docking affinity reward — exp(-dockscore / 3.0), same as original RL.py."""

from __future__ import annotations

import math
import re
from pathlib import Path

DOCKING_SCALE = 3.0

# Regex patterns tried in order (GNINA / AutoDock Vina style logs)
_SCORE_PATTERNS = [
    re.compile(r"Affinity:\s*([-\d.]+)"),
    re.compile(r"REMARK VINA RESULT:\s*([-\d.]+)"),
    re.compile(r"minimizedAffinity\s+([-\d.]+)"),
    re.compile(r"^([-\d.]+)\s", re.MULTILINE),
]


def extract_docking_score(docking_outfile: str | Path) -> float:
    """
    Parse best docking score (kcal/mol) from a GNINA/Vina output file.
    Edit _SCORE_PATTERNS if your docking log format differs.
    """
    path = Path(docking_outfile)
    if not path.is_file():
        raise FileNotFoundError(f"Docking output not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in _SCORE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return float(matches[0])

    raise ValueError(f"Could not parse docking score from: {path}")


def calculateScore(docking_outfile: str | Path | None) -> float | None:
    """
    Docking reward. Returns None if docking_outfile is missing
    (term skipped in geometric mean for non-docking batch runs).
    """
    if not docking_outfile:
        return None
    dockscore = extract_docking_score(docking_outfile)
    return float(math.exp(-dockscore / DOCKING_SCALE))
