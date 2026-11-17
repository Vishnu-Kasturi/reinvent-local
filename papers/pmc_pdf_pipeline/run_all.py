#!/usr/bin/env python3
"""Download PDFs only, then LLM read + judge verification."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf-dir", type=Path, default=Path("."))
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-llm", action="store_true")
    args = p.parse_args()

    pdf_dir = args.pdf_dir.resolve()
    os.environ["PMC_PDF_DIR"] = str(pdf_dir)

    if not args.skip_download:
        sys.path.insert(0, str(_HERE))
        import download_pmc_pdfs as dl

        dl.OUTPUT_DIR = str(pdf_dir)
        dl.main()

    if not args.skip_llm:
        if not os.environ.get("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY to run LLM verification.")
            sys.exit(1)
        subprocess.check_call(
            [
                sys.executable,
                str(_HERE / "llm_read_and_judge.py"),
                "--pdf-dir",
                str(pdf_dir),
                "--out-dir",
                str(pdf_dir / "llm_outputs"),
            ]
        )


if __name__ == "__main__":
    main()