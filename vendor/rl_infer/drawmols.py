#!/usr/bin/env python3
"""
Draw / analyze RL docking outputs (same as analyze_rl_outputs.py).

Parses docking score from mol*_log.txt (pose 1 line) — no docking_scorer module.

Usage:
  python drawmols.py vis_cpt1_files/ --out-dir results/run1/
"""

from analyze_rl_outputs import main

if __name__ == "__main__":
    main()
