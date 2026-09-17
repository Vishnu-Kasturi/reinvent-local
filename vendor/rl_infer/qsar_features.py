"""Feature helper for pIC50 / solubility XGBoost models."""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parent.parent


def get_compute_features():
    sys.path.insert(0, str(_VENDOR_DIR))
    try:
        from pd1_pdl1_features import compute_features  # noqa: WPS433

        return compute_features
    except ImportError:
        repo_root = _VENDOR_DIR.parent
        reinvent4 = repo_root / "REINVENT4"
        if reinvent4.is_dir():
            sys.path.insert(0, str(reinvent4))
            from reinvent_plugins.components.pd1_pdl1_features import compute_features  # noqa: WPS433

            return compute_features
    raise ImportError(
        f"Could not import compute_features. Expected {_VENDOR_DIR / 'pd1_pdl1_features.py'}"
    )
