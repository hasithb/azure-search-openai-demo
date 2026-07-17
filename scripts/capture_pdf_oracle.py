"""Capture an immutable PDF oracle snapshot for a canonical legal source."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pypdf
import requests

from audit_source_documents import load_web_sources
from html_schema_oracle import write_html_snapshot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "html_oracle_snapshots"


def snapshot_filename(identity: str) -> str:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{digest}.json"


def capture_pdf_snapshot(session: requests.Session, source: object, timeout: int = 30) -> dict:
    response = session.get(source.url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    body = bytes(response.content)
    content_type = str(response.headers.get("Content-Type", ""))
    if "application/pdf" not in content_type.casefold() and not source.url.casefold().split("?", 1)[0].endswith(".pdf"):
        raise ValueError(f"Expected PDF response, got {content_type or 'unknown content type'}")

    reader = pypdf.PdfReader(io.BytesIO(body))
    page_texts = [page.extract_text() or "" for page in reader.pages]
    extracted_text = "\n\n".join(text for text in page_texts if text.strip())
    if len(extracted_text.strip()) < 200:
        raise ValueError("PDF extraction quality guard failed: fewer than 200 characters")
    paragraphs = "".join(
        f"<p>{escape(paragraph.strip())}</p>"
        for paragraph in extracted_text.split("\n\n")
        if paragraph.strip()
    )
    title = escape(source.sourcefile)
    html = f"<html><body><article><h1>{title}</h1><div>{paragraphs}</div></article></body></html>"
    return {
        "identity": source.identity,
        "source_type": source.source_type,
        "sourcefile": source.sourcefile,
        "category": source.category,
        "manifest_key": source.manifest_key,
        "requested_url": source.url,
        "final_url": str(response.url),
        "redirect_count": len(getattr(response, "history", [])),
        "status_code": int(response.status_code),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "content_type": content_type,
        "source_sha256": hashlib.sha256(body).hexdigest(),
        "extracted_text": extracted_text,
        "page_count": len(reader.pages),
        "html": html,
        "status": "ok",
    }


def run(output_dir: Path, source_filter: str | None, timeout: int) -> dict:
    sources = [
        source for source in load_web_sources()
        if source.source_type == "pdf"
        and (not source_filter or source_filter.casefold() in source.identity.casefold() or source_filter.casefold() in source.sourcefile.casefold())
    ]
    if not sources:
        raise ValueError("No matching canonical PDF source")
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "legal-rag-pdf-oracle/1.0"})
    results = []
    for source in sources:
        snapshot = capture_pdf_snapshot(session, source, timeout)
        path = output_dir / snapshot_filename(source.identity)
        write_html_snapshot(snapshot, path)
        results.append({"identity": source.identity, "status": "ok", "path": str(path)})
    return {"source_count": len(sources), "ok_count": len(results), "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source", dest="source_filter", help="Capture one PDF source by identity or sourcefile")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.source_filter, args.timeout), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
