"""Fail-closed structural validation for a verified v4 Search snapshot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

from audit_source_documents import load_index_snapshot

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "app" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from customizations.subsection_extractor import SubsectionExtractor


REQUIRED_FIELDS = ("id", "content", "category", "sourcefile", "sourcepage")


def _subsection_mismatches(documents: list[dict[str, Any]]) -> list[str]:
    mismatches: list[str] = []
    for document in documents:
        document_id = str(document.get("id") or "<unknown>")
        content = str(document.get("content") or "")
        extracted = {value.casefold() for value in SubsectionExtractor.extract_all_subsections(content)}
        subsection_id = str(document.get("subsection_id") or "").strip()
        if subsection_id and subsection_id.casefold() not in extracted:
            mismatches.append(f"{document_id}: subsection_id={subsection_id}")
        subsections = document.get("subsections")
        if isinstance(subsections, list):
            for subsection in subsections:
                value = str(subsection or "").strip()
                if value and value.casefold() not in extracted:
                    mismatches.append(f"{document_id}: subsections={value}")
    return mismatches


def validate_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(document.get("id") or "") for document in documents]
    duplicate_ids = sorted(document_id for document_id, count in Counter(ids).items() if document_id and count > 1)
    missing_fields = [
        f"{document.get('id', '<unknown>')}: {field}"
        for document in documents
        for field in REQUIRED_FIELDS
        if not str(document.get(field) or "").strip()
    ]
    uncategorized = sorted({str(document.get("id") or "") for document in documents if not str(document.get("category") or "").strip()})
    empty_content = sorted({str(document.get("id") or "") for document in documents if not str(document.get("content") or "").strip()})
    subsection_mismatches = _subsection_mismatches(documents)
    result = {
        "document_count": len(documents),
        "empty_index_count": int(not documents),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids[:100],
        "missing_field_count": len(missing_fields),
        "missing_fields": missing_fields[:100],
        "uncategorized_count": len(uncategorized),
        "uncategorized_ids": uncategorized[:100],
        "empty_content_count": len(empty_content),
        "empty_content_ids": empty_content[:100],
        "subsection_mismatch_count": len(subsection_mismatches),
        "subsection_mismatches": subsection_mismatches[:100],
    }
    if any(
        result[key]
        for key in (
            "empty_index_count",
            "duplicate_id_count",
            "missing_field_count",
            "uncategorized_count",
            "empty_content_count",
            "subsection_mismatch_count",
        )
    ):
        return {"status": "FAIL", **result}
    return {"status": "PASS", **result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    documents, provenance = load_index_snapshot(args.snapshot)
    result = validate_documents(documents)
    report = {"schema_version": 1, "provenance": provenance, "candidate": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "document_count": result["document_count"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())