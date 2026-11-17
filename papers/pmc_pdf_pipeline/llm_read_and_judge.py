#!/usr/bin/env python3
"""
Read PMC PDFs with an LLM (page-marked full text) and verify with a judge pass.

Verification (proves the model used supplied PDF text, not hallucination):
  1. Ground-truth text extracted locally with PyMuPDF (not XML).
  2. Reader LLM gets FULL text with --- PAGE n --- markers.
  3. Judge step: reader must return verbatim quotes for seeded snippets;
     programmatic overlap score must exceed threshold.
  4. Optional second LLM "judge" rates whether quotes match snippets.

Env:
  OPENAI_API_KEY          — required for API calls
  OPENAI_BASE_URL         — optional (OpenAI-compatible)
  OPENAI_MODEL            — reader model (default gpt-4o-mini)
  OPENAI_JUDGE_MODEL      — judge model (default same as reader)
  PMC_PDF_DIR             — directory of {pmid}.pdf files

Usage:
  python papers/pmc_pdf_pipeline/llm_read_and_judge.py --pdf-dir ./papers/pdfs
  python papers/pmc_pdf_pipeline/llm_read_and_judge.py --pdf ./papers/pdfs/40147006.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from pdf_ground_truth import (
    PdfGroundTruth,
    extract_pdf,
    quote_overlap_score,
    sample_verification_snippets,
)

# Allow running as script from repo root or from this folder
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

READER_SYSTEM = """You are a scientific literature analyst.
You receive the FULL text of a paper with page markers --- PAGE n ---.
You MUST base every claim only on that text.
For verification tasks, return EXACT substrings copied from the document (verbatim quotes).
Do not invent citations or results not present in the text."""

VERIFICATION_USER = """The paper text is below.

{full_text}

---

Task A — Complete read-through:
1. Title (if visible), journal cues, year if stated
2. Structured summary (background, methods, key results, conclusions) — cite PAGE numbers
3. List limitations mentioned in the paper (with page refs)

Task B — Verification quotes (required):
For each item below, return a JSON object in a single ```json``` block with key "verification_quotes"
mapping id -> {{"page": int, "quote": "verbatim substring from that page"}}.

Snippets to anchor (find the passage on the given page and quote ~15-30 words overlapping it):
{snippet_list}

After the JSON block, add a short "coverage_note" stating how many pages contained substantive text you used.
"""


def _get_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("pip install openai") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY")

    base = os.environ.get("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base) if base else OpenAI(api_key=api_key)


def _chat(client, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def _parse_verification_json(text: str) -> dict:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
        return data.get("verification_quotes", data)
    except json.JSONDecodeError:
        return {}


def score_verification(
    gt: PdfGroundTruth,
    snippets: list[dict],
    reader_quotes: dict,
    min_overlap: float = 0.5,
) -> tuple[bool, list[dict]]:
    """Programmatic judge: quotes must overlap ground-truth snippets."""
    details = []
    passed = 0
    for i, snip in enumerate(snippets):
        key = str(i)
        alt_keys = [snip.get("id", key), key, f"snippet_{i}"]
        entry = None
        for k in alt_keys:
            if k in reader_quotes:
                entry = reader_quotes[k]
                break
        if entry is None and isinstance(reader_quotes, list) and i < len(reader_quotes):
            entry = reader_quotes[i]

        quote = ""
        if isinstance(entry, dict):
            quote = str(entry.get("quote", ""))
        elif isinstance(entry, str):
            quote = entry

        overlap = quote_overlap_score(quote, snip["snippet_normalized"])
        ok = overlap >= min_overlap
        if ok:
            passed += 1
        details.append(
            {
                "snippet_id": i,
                "page": snip["page"],
                "reference": snip["snippet_normalized"][:120],
                "reader_quote": quote[:200],
                "overlap": round(overlap, 3),
                "passed": ok,
            }
        )

    need = max(1, len(snippets) - 1)  # allow one miss
    overall = passed >= need
    return overall, details


def judge_llm(
    client,
    model: str,
    snippets: list[dict],
    reader_response: str,
    verification_details: list[dict],
) -> str:
    """Optional LLM judge: confirms quotes align with reference snippets."""
    payload = {
        "reference_snippets": snippets,
        "programmatic_scores": verification_details,
        "reader_full_response_excerpt": reader_response[:12000],
    }
    user = (
        "You are a strict verification judge.\n"
        "Given reference text snippets from a PDF and the reader model's output, "
        "decide if the reader demonstrably used the document (not hallucination).\n"
        "Respond with JSON: {\"verdict\": \"PASS\"|\"FAIL\", \"reason\": \"...\", "
        "\"pages_cited_plausibly\": true|false}\n\n"
        + json.dumps(payload, indent=2)
    )
    return _chat(client, model, "You output only valid JSON.", user)


def process_one_pdf(pdf_path: Path, out_dir: Path, seed: int = 42) -> dict:
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    judge_model = os.environ.get("OPENAI_JUDGE_MODEL", model)
    client = _get_client()

    gt = extract_pdf(pdf_path)
    snippets = sample_verification_snippets(gt, n=5, seed=seed)
    snippet_list = "\n".join(
        f"- id={i} page={s['page']}: ...{s['snippet_normalized'][:80]}..."
        for i, s in enumerate(snippets)
    )

    full_text = gt.full_text_with_markers()
    # Cap extremely long docs with windowed note (still pass full if fits ~100k chars)
    if len(full_text) > 900_000:
        full_text = full_text[:900_000] + "\n\n[TRUNCATED FOR API — use chunked rerun if needed]"

    user_prompt = VERIFICATION_USER.format(
        full_text=full_text,
        snippet_list=snippet_list,
    )

    print(f"[READER] {pdf_path.name} ({gt.page_count} pages, {gt.total_chars} chars)")
    reader_out = _chat(client, model, READER_SYSTEM, user_prompt)

    quotes = _parse_verification_json(reader_out)
    ok_prog, details = score_verification(gt, snippets, quotes)

    judge_out = ""
    judge_json: dict = {}
    if ok_prog:
        judge_out = judge_llm(client, judge_model, snippets, reader_out, details)
        try:
            judge_json = json.loads(judge_out)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", judge_out, re.DOTALL)
            if m:
                try:
                    judge_json = json.loads(m.group(0))
                except json.JSONDecodeError:
                    judge_json = {"raw": judge_out}

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    manifest = {
        "pdf": str(pdf_path),
        "sha256": gt.sha256,
        "page_count": gt.page_count,
        "total_chars": gt.total_chars,
        "programmatic_verification_passed": ok_prog,
        "verification_details": details,
        "judge": judge_json,
    }

    (out_dir / f"{stem}_reader.md").write_text(reader_out, encoding="utf-8")
    (out_dir / f"{stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    verdict = "PASS" if ok_prog and judge_json.get("verdict") != "FAIL" else "FAIL"
    if ok_prog and not judge_json:
        verdict = "PASS (programmatic only)"

    print(f"[VERIFY] {stem} -> {verdict}")
    return {"pdf": stem, "verdict": verdict, "manifest": manifest}


def main() -> None:
    p = argparse.ArgumentParser(description="LLM read PDF + judge verification")
    p.add_argument("--pdf-dir", type=Path, default=None)
    p.add_argument("--pdf", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pdf_dir = args.pdf_dir or Path(os.environ.get("PMC_PDF_DIR", "."))
    out_dir = args.out_dir or (pdf_dir / "llm_outputs")

    paths: list[Path] = []
    if args.pdf:
        paths = [args.pdf.resolve()]
    else:
        paths = sorted(pdf_dir.glob("*.pdf"))

    if not paths:
        raise SystemExit(f"No PDFs in {pdf_dir}")

    summary = []
    for path in paths:
        try:
            summary.append(process_one_pdf(path, out_dir, seed=args.seed))
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}")
            summary.append({"pdf": path.stem, "verdict": f"ERROR: {exc}"})

    (out_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nWrote:", out_dir / "batch_summary.json")


if __name__ == "__main__":
    main()