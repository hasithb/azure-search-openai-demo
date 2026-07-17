"""Capture canonical HTML responses for the independent schema oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from audit_source_documents import load_web_sources
from html_schema_oracle import ORACLE_VERSION, capture_html_snapshot, write_html_snapshot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "html_oracle_snapshots"


def snapshot_filename(identity: str) -> str:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{digest}.json"


def source_matches(source: Any, source_filter: str | None) -> bool:
    if not source_filter:
        return True
    value = source_filter.casefold()
    return value in source.identity.casefold() or value in source.sourcefile.casefold() or value in source.url.casefold()


def capture_source(session: requests.Session, source: Any, output_dir: Path, timeout: int, refresh: bool = False) -> dict[str, Any]:
    output_path = output_dir / snapshot_filename(source.identity)

    if source.source_type == "pdf" or source.url.casefold().split("?", 1)[0].endswith(".pdf"):
        not_applicable = {
            "identity": source.identity,
            "source_type": source.source_type,
            "sourcefile": source.sourcefile,
            "category": source.category,
            "manifest_key": source.manifest_key,
            "requested_url": source.url,
            "status": "not_applicable",
            "reason": "PDF source is covered by PDF completeness verification, not the HTML DOM oracle",
        }
        write_html_snapshot(not_applicable, output_path)
        return not_applicable

    if output_path.exists() and not refresh:
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
        if existing.get("status") == "ok":
            return {"identity": source.identity, "status": "skipped", "path": str(output_path)}

    parsed_url = urlparse(source.url)
    if not source.url or parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        failure = {
            "identity": source.identity,
            "source_type": source.source_type,
            "sourcefile": source.sourcefile,
            "category": source.category,
            "manifest_key": source.manifest_key,
            "requested_url": source.url,
            "status": "unavailable",
            "error": "canonical source has no usable HTTP URL",
        }
        write_html_snapshot(failure, output_path)
        return failure

    try:
        snapshot = capture_html_snapshot(session, source.url, timeout=timeout)
    except Exception as error:  # noqa: BLE001 - capture must record per-source failures and continue
        failure = {
            "identity": source.identity,
            "source_type": source.source_type,
            "sourcefile": source.sourcefile,
            "category": source.category,
            "manifest_key": source.manifest_key,
            "requested_url": source.url,
            "status": "unavailable",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_html_snapshot(failure, output_path)
        return failure

    snapshot.update(
        {
            "identity": source.identity,
            "source_type": source.source_type,
            "sourcefile": source.sourcefile,
            "category": source.category,
            "manifest_key": source.manifest_key,
            "status": "ok",
        }
    )
    write_html_snapshot(snapshot, output_path)
    return {"identity": source.identity, "status": "ok", "path": str(output_path)}


def run(output_dir: Path, source_filter: str | None, limit: int | None, timeout: int, refresh: bool = False) -> dict[str, Any]:
    sources = [source for source in load_web_sources() if source_matches(source, source_filter)]
    if limit is not None:
        sources = sources[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "legal-rag-html-oracle/1.0"})
    results = [capture_source(session, source, output_dir, timeout, refresh) for source in sources]
    summary = {
        "schema_version": 1,
        "oracle_version": ORACLE_VERSION,
        "source_count": len(sources),
        "ok_count": sum(result.get("status") == "ok" for result in results),
        "skipped_count": sum(result.get("status") == "skipped" for result in results),
        "unavailable_count": sum(result.get("status") == "unavailable" for result in results),
        "not_applicable_count": sum(result.get("status") == "not_applicable" for result in results),
        "results": results,
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source", dest="source_filter", help="Capture only sources matching identity, filename, or URL")
    parser.add_argument("--limit", type=int, help="Capture at most this many selected sources")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--refresh", action="store_true", help="Re-fetch and rewrite existing successful snapshots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args.output_dir, args.source_filter, args.limit, args.timeout, args.refresh)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, sort_keys=True))
    return 0 if summary["unavailable_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())