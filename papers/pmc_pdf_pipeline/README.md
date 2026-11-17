# PMC PDF-only pipeline + verified LLM reading

No XML. Only `.pdf` objects from PMC OA S3 (or `pdf_url` from JSON metadata).

## Install

```bash
pip install -r papers/pmc_pdf_pipeline/requirements.txt
```

## 1. Download PDFs

```bash
cd papers/pdfs   # or any output folder
export PMC_PDF_DIR=.
python ../pmc_pdf_pipeline/download_pmc_pdfs.py
```

Edit `ARTICLES` in `download_pmc_pdfs.py` for your PMID → PMCID map.

## 2. LLM read + verification

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o        # optional; use a model with large context for long PDFs
python papers/pmc_pdf_pipeline/llm_read_and_judge.py --pdf-dir papers/pdfs
```

### How verification works

| Step | What |
|------|------|
| **Ground truth** | PyMuPDF extracts text per page (`pdf_ground_truth.py`) — proves the file is a readable PDF |
| **Reader LLM** | Gets **full text** with `--- PAGE n ---` markers + structured summary task |
| **Snippet test** | 5 random 12-word anchors (seeded); reader must return **verbatim quotes** in JSON |
| **Programmatic judge** | 3-gram overlap between quote and anchor (≥50%); need 4/5 hits |
| **LLM judge** | Second call checks plausibility; `PASS`/`FAIL` in `{pmid}_manifest.json` |

Outputs per PDF:

- `{pmid}_reader.md` — summary + quotes
- `{pmid}_manifest.json` — sha256, page count, verification scores
- `batch_summary.json`

## One command

```bash
python papers/pmc_pdf_pipeline/run_all.py --pdf-dir papers/pdfs
```

## Scanned PDFs

If text extraction &lt; 200 characters, the pipeline stops (image-only PDF). Use OCR upstream if needed.