"""Build the release-bound canonical subsection manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_exclusions(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Exclusion allowlist must be a JSON array")
    exclusions = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("Every exclusion must be an object")
        if not str(entry.get("identity") or "").strip() or not str(entry.get("reason") or "").strip():
            raise ValueError("Every exclusion needs identity and reason")
        exclusions.append({
            "identity": str(entry["identity"]).strip(),
            "reason": str(entry["reason"]).strip(),
            "reviewed_by": str(entry.get("reviewed_by") or "").strip(),
        })
    return sorted(exclusions, key=lambda entry: (entry["identity"], entry["reason"]))


def build_manifest(oracle_path: Path, exclusions_path: Path | None = None) -> dict[str, Any]:
    oracle = load_json(oracle_path)
    cases = oracle.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Oracle has no cases")
    exclusions = load_exclusions(exclusions_path)
    excluded_identities = {entry["identity"] for entry in exclusions}
    sources: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    subsections = []
    seen_cases: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Oracle contains a non-object case")
        required = ("case_id", "identity", "sourcefile", "sourcepage", "subsection_id", "expected_heading", "heading_locator", "body_sha256", "body_length", "snapshot_file", "snapshot_content_sha256")
        if any(not str(case.get(field) or "").strip() for field in required):
            raise ValueError("Oracle contains an incomplete subsection case")
        case_id = str(case["case_id"])
        identity = str(case["identity"])
        if case_id in seen_cases:
            raise ValueError(f"Duplicate subsection case: {case_id}")
        seen_cases.add(case_id)
        snapshot_file = str(case["snapshot_file"])
        document_id = stable_id("document", f"{identity}|{snapshot_file}")
        sources.setdefault(identity, {
            "identity": identity,
            "category": str(case.get("category") or ""),
            "sourcefile": str(case["sourcefile"]),
            "snapshot_file": snapshot_file,
            "snapshot_content_sha256": str(case["snapshot_content_sha256"]),
            "status": "excluded" if identity in excluded_identities else "included",
        })
        documents.setdefault(document_id, {
            "document_id": document_id,
            "search_document_id": str(case.get("search_document_id") or ""),
            "identity": identity,
            "snapshot_file": snapshot_file,
            "sourcefile": str(case["sourcefile"]),
            "sourcepage_count": 0,
        })
        documents[document_id]["sourcepage_count"] += 1
        subsections.append({
            "case_id": case_id,
            "document_id": document_id,
            "identity": identity,
            "sourcefile": str(case["sourcefile"]),
            "sourcepage": str(case["sourcepage"]),
            "subsection_id": str(case["subsection_id"]),
            "expected_heading": str(case["expected_heading"]),
            "heading_locator": str(case["heading_locator"]),
            "next_heading": case.get("next_heading"),
            "next_heading_locator": case.get("next_heading_locator"),
            "body_sha256": str(case["body_sha256"]),
            "body_length": int(case["body_length"]),
            "primary_source": str(case["sourcefile"]),
            "expected_citation": str(case["sourcepage"]),
            "status": "excluded" if identity in excluded_identities else "included",
        })
    return {
        "schema_version": 2,
        "manifest_type": "v4_canonical_subsections",
        "oracle_sha256": sha256_file(oracle_path),
        "oracle_version": str(oracle.get("oracle_version") or ""),
        "snapshot_manifest_sha256": str(oracle.get("snapshot_manifest_sha256") or ""),
        "source_count": len(sources),
        "document_count": len(documents),
        "subsection_count": len(subsections),
        "included_subsection_count": sum(item["status"] == "included" for item in subsections),
        "excluded_source_count": len(excluded_identities),
        "sources": sorted(sources.values(), key=lambda item: item["identity"]),
        "documents": sorted(documents.values(), key=lambda item: item["document_id"]),
        "subsections": sorted(subsections, key=lambda item: item["case_id"]),
        "exclusions": exclusions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.oracle, args.exclusions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())