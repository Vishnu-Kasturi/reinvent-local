"""
Extract page-bounded text from PDFs (ground truth for LLM verification).
No XML — PyMuPDF on .pdf files only.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageText:
    page: int  # 1-based
    text: str


@dataclass
class PdfGroundTruth:
    path: Path
    sha256: str
    page_count: int
    total_chars: int
    pages: list[PageText] = field(default_factory=list)

    def full_text_with_markers(self) -> str:
        blocks = []
        for p in self.pages:
            blocks.append(f"\n--- PAGE {p.page} ---\n{p.text.strip()}\n")
        return "".join(blocks)


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_pdf(path: Path) -> PdfGroundTruth:
    path = path.resolve()
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()

    doc = fitz.open(path)
    try:
        pages: list[PageText] = []
        total = 0
        for i in range(doc.page_count):
            text = doc.load_page(i).get_text("text") or ""
            text = text.strip()
            total += len(text)
            pages.append(PageText(page=i + 1, text=text))
    finally:
        doc.close()

    if not pages or total < 200:
        raise ValueError(
            f"PDF has too little extractable text ({total} chars). "
            "Scanned PDFs may need OCR — not supported here."
        )

    return PdfGroundTruth(
        path=path,
        sha256=sha,
        page_count=len(pages),
        total_chars=total,
        pages=pages,
    )


def sample_verification_snippets(
    gt: PdfGroundTruth, n: int = 5, seed: int = 0
) -> list[dict]:
    """
    Pick short anchored snippets from random pages (deterministic seed).
    Judge checks that reader LLM quotes overlap these strings.
    """
    import random

    rng = random.Random(seed)
    candidates: list[tuple[int, str]] = []
    for p in gt.pages:
        if len(p.text) < 80:
            continue
        # mid-page sentence-ish chunk
        words = _normalize_whitespace(p.text).split()
        if len(words) < 12:
            continue
        start = rng.randint(0, max(0, len(words) - 12))
        chunk = " ".join(words[start : start + 12])
        candidates.append((p.page, chunk))

    rng.shuffle(candidates)
    picked = candidates[:n]
    return [
        {"page": page, "snippet": snippet, "snippet_normalized": _normalize_whitespace(snippet)}
        for page, snippet in picked
    ]


def quote_overlap_score(reader_quote: str, reference: str) -> float:
    """Fraction of reference word 3-grams found in reader quote."""
    def ngrams(words: list[str], n: int = 3):
        return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

    rq = _normalize_whitespace(reader_quote).lower()
    ref = _normalize_whitespace(reference).lower()
    rw = rq.split()
    refw = ref.split()
    if len(refw) < 3:
        return 1.0 if ref in rq else 0.0
    ref_ng = ngrams(refw, 3)
    if not ref_ng:
        return 0.0
    hit = sum(1 for g in ref_ng if g in rq)
    return hit / len(ref_ng)