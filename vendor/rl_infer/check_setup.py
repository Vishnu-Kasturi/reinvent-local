#!/usr/bin/env python3
"""Verify rl_infer reward dependencies before running vis_rl.py."""

import sys
from pathlib import Path

_INFER_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _INFER_DIR.parent
sys.path.insert(0, str(_INFER_DIR))

OK = "[OK]"
FAIL = "[FAIL]"


def _check(label, path):
    p = Path(path)
    if p.is_file():
        print(f"{OK} {label}: {p}")
        return True
    print(f"{FAIL} {label}: missing {p}")
    return False


def main():
    ok = True
    ok &= _check("SA fpscores", _INFER_DIR / "fpscores.pkl.gz")

    model_root = _VENDOR_DIR / "Preprocess/final_acc"
    ok &= _check("pIC50 model", model_root / "pd1_pdl1_pic50_final_acc_model.ubj")
    ok &= _check("pIC50 scaler", model_root / "pd1_pdl1_pic50_final_acc_scaler.pkl")
    ok &= _check("sol model", model_root / "pd1_pdl1_sol_final_acc_model.ubj")
    ok &= _check("sol scaler", model_root / "pd1_pdl1_sol_final_acc_scaler.pkl")

    alt_root = _INFER_DIR / "Preprocess/final_acc"
    if not (model_root / "pd1_pdl1_pic50_final_acc_model.ubj").is_file():
        ok &= _check("pIC50 model (alt)", alt_root / "pd1_pdl1_pic50_final_acc_model.ubj")

    print()
    try:
        from RL import init_scorers, get_reward

        init_scorers()
        print(f"{OK} init_scorers()")
    except Exception as exc:
        print(f"{FAIL} init_scorers(): {exc}")
        ok = False

    print()
    print("Expected args: indexfile char_to_int_file int_to_char_file graphpath "
          "svae_gvae_combined_cptfile predictor_cptfile docking_path savepath retraining_flag")
    if len(sys.argv) > 1:
        print("argv:", sys.argv[1:])

    if not ok:
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
