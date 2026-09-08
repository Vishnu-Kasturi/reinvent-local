"""Argv validation helpers for vendor/rl_infer/vis_rl.py."""

import sys
from dataclasses import dataclass
from pathlib import Path

_EXPECTED_ARGS = 9
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


def print_usage(script_name="vis_rl.py"):
    print(
        f"Usage: python {script_name} "
        + " ".join(f"<{n}>" for n in _ARG_NAMES)
    )
    print(f"\nExpected {_EXPECTED_ARGS} arguments after the script name.")


def parse_vis_rl_argv(argv=None) -> VisRLConfig:
    if argv is None:
        argv = sys.argv

    n_got = len(argv) - 1
    if n_got != _EXPECTED_ARGS:
        print(f"ERROR: expected {_EXPECTED_ARGS} arguments, got {n_got}")
        if n_got:
            for i, val in enumerate(argv[1:], start=1):
                name = _ARG_NAMES[i - 1] if i <= len(_ARG_NAMES) else "?"
                print(f"  [{i}] {name}: {val!r}")
        print()
        print_usage(Path(argv[0]).name if argv else "vis_rl.py")
        sys.exit(1)

    return VisRLConfig(*argv[1:])


def boot_scorers():
    from RL import init_scorers

    print("Initializing reward scorers (SA, pIC50, solubility)...")
    init_scorers()
    print("Reward scorers ready.")
