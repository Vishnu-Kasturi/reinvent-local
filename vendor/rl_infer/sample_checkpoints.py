#!/usr/bin/env python3
"""
Sample SMILES from every checkpoint in a directory using sample.py.

For each *.cpt checkpoint, runs the CVAE sampler and writes a CSV with a SMILES column.

Usage:
  python sample_checkpoints.py CHECKPOINT_DIR RECEPTOR CHAR_TO_INT INT_TO_CHAR GRAPHPATH \\
      --out-dir sampled_smiles/ --n-samples 500

Example:
  python sample_checkpoints.py vis_cpt1_files/ 9iow char_to_int.pkl int_to_char.pkl \\
      /path/to/graphs/ --out-dir results/checkpoint_samples/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _resolve_graphpath(graph_path: str) -> tuple[str, str | None]:
    path = graph_path.rstrip("/")
    receptor_hint = None
    base = Path(path).name
    if base.endswith(".graph"):
        receptor_hint = base[: -len(".graph")]
        path = str(Path(path).parent)
    elif base.endswith(".x"):
        receptor_hint = base[: -len(".x")]
        path = str(Path(path).parent)
    if path and not path.endswith("/"):
        path = path + "/"
    return path, receptor_hint


def _find_checkpoints(checkpoint_dir: Path, pattern: str) -> list[Path]:
    ckpts = sorted(checkpoint_dir.glob(pattern))
    if not ckpts and checkpoint_dir.is_file() and checkpoint_dir.suffix == ".cpt":
        return [checkpoint_dir]
    return [p for p in ckpts if p.is_file()]


def main() -> None:
    p = argparse.ArgumentParser(description="Sample SMILES from all checkpoints in a folder")
    p.add_argument("checkpoint_dir", help="Directory containing *.cpt checkpoints")
    p.add_argument("receptor", nargs="?", default=None, help="Receptor id, e.g. 9iow (optional if graphpath ends in .graph)")
    p.add_argument("char_to_int", help="char_to_int.pkl")
    p.add_argument("int_to_char", help="int_to_char.pkl")
    p.add_argument("graphpath", help="Graph folder or path like .../9iow.graph")
    p.add_argument("--out-dir", required=True, help="Output directory for per-checkpoint CSV files")
    p.add_argument("--n-samples", type=int, default=500, help="Valid SMILES to collect per checkpoint")
    p.add_argument("--pattern", default="*.cpt", help="Checkpoint glob pattern (default: *.cpt)")
    p.add_argument(
        "--sample-script",
        default=None,
        help="Path to sample.py (default: sample.py next to this script)",
    )
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    sample_script = Path(args.sample_script) if args.sample_script else script_dir / "sample.py"
    if not sample_script.is_file():
        raise SystemExit(f"sample.py not found: {sample_script}")

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    graphpath, receptor_hint = _resolve_graphpath(args.graphpath)
    receptor = args.receptor or receptor_hint
    if not receptor:
        raise SystemExit("Provide RECEPTOR or pass graphpath ending in .graph / .x")

    ckpts = _find_checkpoints(checkpoint_dir, args.pattern)
    if not ckpts:
        raise SystemExit(f"No checkpoints matching {args.pattern!r} in {checkpoint_dir}")

    print(f"Found {len(ckpts)} checkpoint(s) in {checkpoint_dir}")
    print(f"Sampling {args.n_samples} valid SMILES each -> {out_dir}")

    for i, ckpt in enumerate(ckpts, start=1):
        out_csv = out_dir / f"{ckpt.stem}.csv"
        print(f"[{i}/{len(ckpts)}] {ckpt.name} -> {out_csv.name}")
        cmd = [
            sys.executable,
            str(sample_script),
            receptor,
            str(Path(args.char_to_int).expanduser().resolve()),
            str(Path(args.int_to_char).expanduser().resolve()),
            graphpath,
            str(ckpt),
            str(out_csv),
            str(args.n_samples),
        ]
        subprocess.run(cmd, check=True)

    manifest = out_dir / "checkpoint_manifest.txt"
    manifest.write_text("\n".join(p.name for p in ckpts) + "\n", encoding="utf-8")
    print(f"[+] Done. {len(ckpts)} CSV files in {out_dir}")
    print(f"[+] Manifest -> {manifest}")


if __name__ == "__main__":
    main()
