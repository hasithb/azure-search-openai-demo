"""Validate exhaustive citation-to-Search-to-UI evidence for a v4 release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class CitationCoverageError(ValueError):
    """Raised when citation evidence does not reconcile exactly."""


IDENTITY_FIELDS = ("source_revision", "source_id", "document_id", "subsection_id", "canonical_text_sha256")
EVIDENCE_FIELDS = ("rendered", "clicked", "supporting_content_count", "primary_source_count", "highlighted_text_sha256")


def _identity(record: dict[str, Any]) -> tuple[str, ...]:
    values = tuple(str(record.get(field) or "").strip() for field in IDENTITY_FIELDS)
    if any(not value for value in values):
        raise CitationCoverageError("Citation record has incomplete immutable identity")
    return values


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != 1:
        raise CitationCoverageError("Citation coverage report must use schema version 1")
    manifest = report.get("manifest")
    search_documents = report.get("search_documents")
    if not isinstance(manifest, list) or not manifest:
        raise CitationCoverageError("Citation coverage manifest is empty")
    if not isinstance(search_documents, list):
        raise CitationCoverageError("Citation coverage is missing Search document evidence")

    search_by_id: dict[str, dict[str, Any]] = {}
    for document in search_documents:
        if not isinstance(document, dict) or not str(document.get("id") or "").strip():
            raise CitationCoverageError("Search evidence contains a document without an id")
        document_id = str(document["id"])
        if document_id in search_by_id:
            raise CitationCoverageError(f"Duplicate Search document id: {document_id}")
        search_by_id[document_id] = document

    identities: set[tuple[str, ...]] = set()
    failures: list[str] = []
    for record in manifest:
        if not isinstance(record, dict):
            raise CitationCoverageError("Citation manifest contains a non-object record")
        identity = _identity(record)
        if identity in identities:
            raise CitationCoverageError(f"Duplicate immutable citation identity: {identity}")
        identities.add(identity)
        document = search_by_id.get(identity[2])
        if document is None:
            failures.append(f"missing Search document: {identity[2]}")
            continue
        if str(document.get("source_revision") or record.get("source_revision") or "") != identity[0]:
            failures.append(f"source revision mismatch: {identity[2]}")
        document_hash = str(document.get("canonical_text_sha256") or "")
        if document_hash and document_hash != identity[4]:
            failures.append(f"canonical text hash mismatch: {identity[2]}")
        for field in EVIDENCE_FIELDS:
            if field not in record:
                raise CitationCoverageError(f"Citation record is missing {field}: {identity[2]}")
        if record["rendered"] is not True or record["clicked"] is not True:
            failures.append(f"citation was not rendered and clicked exactly once: {identity[2]}")
        if record["supporting_content_count"] != 1 or record["primary_source_count"] != 1:
            failures.append(f"UI view count mismatch: {identity[2]}")
        if record["highlighted_text_sha256"] != identity[4]:
            failures.append(f"highlight hash mismatch: {identity[2]}")

    expected_count = len(manifest)
    counts = {
        "manifest": expected_count,
        "search_documents": len(search_documents),
        "rendered": sum(record.get("rendered") is True for record in manifest),
        "clicked": sum(record.get("clicked") is True for record in manifest),
        "supporting_content": sum(record.get("supporting_content_count") == 1 for record in manifest),
        "primary_source": sum(record.get("primary_source_count") == 1 for record in manifest),
    }
    report_out = {"schema_version": 1, "status": "PASS", "case_count": expected_count, "counts": counts, "failures": failures}
    if counts["search_documents"] != expected_count or any(value != expected_count for key, value in counts.items() if key != "search_documents") or failures:
        report_out["status"] = "FAIL"
        raise CitationCoverageError(json.dumps(report_out, sort_keys=True))
    report_out["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return report_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise CitationCoverageError("Citation coverage input must be a JSON object")
        result = validate_report(payload)
    except (OSError, json.JSONDecodeError, CitationCoverageError) as error:
        result = {"schema_version": 1, "status": "FAIL", "error": str(error)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())