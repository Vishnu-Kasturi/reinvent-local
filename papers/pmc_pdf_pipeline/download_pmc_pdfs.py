#!/usr/bin/env python3
"""
PMC Open Access bulk downloader — PDF objects only (no XML, no other formats).

Uses NIH PMC OA S3 bucket + optional metadata pdf_url fallback.
Outputs: {OUTPUT_DIR}/{pmid}.pdf and pmc_download_report.csv
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Any

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

# ── CONFIG ───────────────────────────────────────────────────────────────────
ARTICLES: dict[str, str] = {
    "40147006": "PMC11987020",
    "36232394": "PMC9570041",
    "37959782": "PMC10647610",
    "36771096": "PMC9920234",
    "36144624": "PMC9505602",
}

OUTPUT_DIR = os.environ.get("PMC_PDF_DIR", ".")
BUCKET = "pmc-oa-opendata"
# ─────────────────────────────────────────────────────────────────────────────

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
session = requests.Session()
session.headers.update({"User-Agent": "PMCResearchDownloader/2.0-pdf-only"})

s3 = boto3.client(
    "s3",
    region_name="us-east-1",
    config=Config(
        signature_version=UNSIGNED,
        retries={"max_attempts": 5, "mode": "standard"},
    ),
)


def check_ncbi(pmid: str) -> tuple[bool | None, list[str]]:
    query = (
        f"{pmid}[pmid] AND has_pdf[filter] "
        "AND (open_access[filter] OR author_manuscript[filter])"
    )
    try:
        response = session.get(
            EUTILS + "esearch.fcgi",
            params={"db": "pmc", "term": query, "retmode": "json"},
            timeout=40,
        )
        response.raise_for_status()
        ids = response.json()["esearchresult"]["idlist"]
        return bool(ids), ids
    except Exception as exc:
        print("[NCBI ERROR]", exc)
        return None, []


def find_versions(pmcid: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    versions: list[str] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=pmcid + ".", Delimiter="/"):
        for item in page.get("CommonPrefixes", []):
            versions.append(item["Prefix"])
    return versions


def list_pdfs_only(prefix: str) -> list[str]:
    """List object keys under prefix that end with .pdf (case-insensitive)."""
    paginator = s3.get_paginator("list_objects_v2")
    pdfs: list[str] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.lower().endswith(".pdf"):
                pdfs.append(key)
    return pdfs


def read_metadata_pdf_url(prefix: str) -> str | None:
    """Optional: read json metadata only to get pdf_url — never downloads XML."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.lower().endswith(".json"):
                continue
            try:
                body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
                meta = json.loads(body)
                url = meta.get("pdf_url")
                if url:
                    return str(url)
            except Exception:
                continue
    return None


def is_valid_pdf(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as f:
        return f.read(5).startswith(b"%PDF")


def download_pdf_s3(key: str, output_path: str) -> bool:
    temp_path = output_path + ".tmp"
    try:
        print("[DOWNLOADING PDF]", key)
        s3.download_file(BUCKET, key, temp_path)
        if not is_valid_pdf(temp_path):
            os.remove(temp_path)
            print("[INVALID PDF]", key)
            return False
        os.replace(temp_path, output_path)
        print("[SUCCESS]", output_path)
        return True
    except Exception as exc:
        print("[DOWNLOAD ERROR]", exc)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def download_pdf_url(url: str, output_path: str) -> bool:
    try:
        print("[DOWNLOADING URL]", url)
        response = session.get(url, timeout=120)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            print("[INVALID PDF FROM URL]")
            return False
        with open(output_path, "wb") as f:
            f.write(response.content)
        print("[SUCCESS]", output_path)
        return True
    except Exception as exc:
        print("[URL ERROR]", exc)
        return False


def process_article(pmid: str, pmcid: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{pmid}.pdf")

    if is_valid_pdf(output_path):
        print("[SKIP] Already downloaded:", pmid)
        return "Already downloaded"
    if os.path.exists(output_path):
        os.remove(output_path)

    print("\n" + "=" * 60)
    print("PMID:", pmid, "PMCID:", pmcid)

    available, _ = check_ncbi(pmid)
    if available is True:
        print("[NCBI] In has_pdf OA/author manuscript subset")
    elif available is False:
        print("[NCBI] Not in filter — still trying AWS PDF objects")
    else:
        print("[NCBI] Check failed — still trying AWS PDF objects")

    try:
        versions = find_versions(pmcid)
        if not versions:
            return "No AWS article version"

        for version in versions:
            print("\nVersion:", version)
            pdf_keys = list_pdfs_only(version)
            for key in pdf_keys:
                if download_pdf_s3(key, output_path):
                    return "Downloaded PDF from AWS"

            pdf_url = read_metadata_pdf_url(version)
            if pdf_url and download_pdf_url(pdf_url, output_path):
                return "Downloaded PDF from metadata URL"

        return "No PDF object found (PDF-only mode; XML ignored)"

    except ClientError as exc:
        print("[AWS ERROR]", exc)
        return "AWS error"


def main() -> None:
    report: list[dict[str, Any]] = []
    print("\nPMC PDF-ONLY DOWNLOADER")
    print("=" * 60)

    for pmid, pmcid in ARTICLES.items():
        status = process_article(pmid, pmcid)
        report.append({"PMID": pmid, "PMCID": pmcid, "Status": status})
        time.sleep(0.5)

    report_path = os.path.join(OUTPUT_DIR, "pmc_download_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["PMID", "PMCID", "Status"])
        writer.writeheader()
        writer.writerows(report)

    ok = sum(
        1
        for row in report
        if row["Status"].startswith("Downloaded") or row["Status"] == "Already downloaded"
    )
    print("\nSuccessful:", ok, "/", len(ARTICLES))
    print("PDF dir:", OUTPUT_DIR)
    print("Report:", report_path)


if __name__ == "__main__":
    main()