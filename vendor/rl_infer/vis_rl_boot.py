"""
Drop-in helpers for CVAE_RL_finetuning/vis_rl.py.

If vis_rl.py prints "Device available: cuda" then exits, the usual causes are:
  1. len(sys.argv) != 10  (script name + 9 args) — often fails silently
  2. No code after function definitions (missing if __name__ block)
  3. init_scorers() never called before get_reward()

Add at the bottom of vis_rl.py (after all function defs):

    from vis_rl_boot import parse_vis_rl_argv, boot_scorers, print_usage

    if __name__ == "__main__":
        args = parse_vis_rl_argv()
        boot_scorers()
        indexfile, output_path, svae_cpt, gvae_cpt, graph_file, x_file, batch_size, n_iters, sigma_val = args
        # ... load models, read index, then:
        # predictor = "yes"   # or your GBT flag — kept for get_reward signature
        # rl_losses, avg_rewards = rltrainingloop(...)
"""

import sys
from pathlib import Path

_EXPECTED_ARGS = 9
_ARG_NAMES = (
    "indexfile",
    "output_path",
    "svae_cpt",
    "gvae_cpt",
    "graph_file",
    "x_file",
    "batch_size",
    "n_iters",
    "sigma_val",
)


def print_usage(script_name="vis_rl.py"):
    print(
        f"Usage: python {script_name} "
        + " ".join(f"<{n}>" for n in _ARG_NAMES)
    )
    print(f"\nExpected {_EXPECTED_ARGS} arguments after the script name.")
    print("Example:")
    print(
        f"  python {script_name} index.txt ./out/ prior.cpt graph.cpt "
        "9iow.graph 9iow.x 32 100 3"
    )


def parse_vis_rl_argv(argv=None):
    """Validate argv and return the 9 positional args. Exits with a clear message if wrong."""
    if argv is None:
        argv = sys.argv

    n_got = len(argv) - 1
    if n_got != _EXPECTED_ARGS:
        print(f"ERROR: expected {_EXPECTED_ARGS} arguments, got {n_got}")
        if n_got:
            print("Arguments received:")
            for i, val in enumerate(argv[1:], start=1):
                name = _ARG_NAMES[i - 1] if i <= len(_ARG_NAMES) else "?"
                print(f"  [{i}] {name}: {val!r}")
        print()
        print_usage(Path(argv[0]).name if argv else "vis_rl.py")
        sys.exit(1)

    args = argv[1:]
    for path_arg in (args[0], args[1], args[2], args[3], args[4], args[5]):
        if not Path(path_arg).expanduser().exists():
            print(f"WARNING: path does not exist: {path_arg}")
    return args


def boot_scorers():
    """Load SA fragments + pIC50/sol XGBoost models. Call once before RL."""
    from RL import init_scorers

    print("Initializing reward scorers (SA, pIC50, solubility)...")
    init_scorers()
    print("Reward scorers ready.")
