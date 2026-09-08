"""Optional argv helpers for vis_rl.py (same 9 args as the main script)."""

import sys
from dataclasses import dataclass
from pathlib import Path

_ARG_NAMES = (
    "indexfile",
    "char_to_int_file",
    "int_to_char_file",
    "graphpath",
    "svae_gvae_combined_cptfile",
    "predictor_cptfile",
    "docking_path",
    "savepath",
    "retraining_flag",
)


@dataclass(frozen=True)
class VisRLConfig:
    indexfile: str
    char_to_int_file: str
    int_to_char_file: str
    graphpath: str
    svae_gvae_combined_cptfile: str
    predictor_cptfile: str
    docking_path: str
    savepath: str
    retraining_flag: str


def parse_vis_rl_argv(argv=None) -> VisRLConfig:
    if argv is None:
        argv = sys.argv
    n_got = len(argv) - 1
    if n_got != 9:
        print(f"ERROR: expected 9 arguments, got {n_got}")
        sys.exit(1)
    return VisRLConfig(*argv[1:])


def boot_scorers():
    from RL import init_scorers
    init_scorers()
