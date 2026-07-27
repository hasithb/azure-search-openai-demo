"""Validate the release-bound canonical subsection manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class SubsectionManifestError(ValueError):
    """Raised when canonical subsection evidence is incomplete or stale."""


def validate(manifest: dict[str, Any], oracle_path: Path) -> dict[str, Any]:
    if manifest.get("schema_version") != 2 or manifest.get("manifest_type") != "v4_canonical_subsections":
        raise SubsectionManifestError("Unsupported canonical subsection manifest schema")
    actual_oracle_hash = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    if manifest.get("oracle_sha256") != actual_oracle_hash:
        raise SubsectionManifestError("Manifest is not bound to the supplied oracle")
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    oracle_cases = oracle.get("cases")
    if not isinstance(oracle_cases, list) or not oracle_cases:
        raise SubsectionManifestError("Supplied oracle has no cases")
    sources = manifest.get("sources")
    documents = manifest.get("documents")
    subsections = manifest.get("subsections")
    exclusions = manifest.get("exclusions")
    if not all(isinstance(value, list) for value in (sources, documents, subsections, exclusions)):
        raise SubsectionManifestError("Manifest collections must be arrays")
    if manifest.get("subsection_count") != len(subsections) or manifest.get("document_count") != len(documents):
        raise SubsectionManifestError("Manifest counts do not match evidence")
    if manifest.get("source_count") != len(sources):
        raise SubsectionManifestError("Manifest source_count does not match evidence")
    source_ids = [str(item.get("identity") or "") for item in sources if isinstance(item, dict)]
    if len(source_ids) != len(set(source_ids)) or any(not value for value in source_ids):
        raise SubsectionManifestError("Source identities must be present and unique")
    excluded = {str(item.get("identity") or "") for item in exclusions if isinstance(item, dict)}
    if len(excluded) != len(exclusions) or any(not identity for identity in excluded):
        raise SubsectionManifestError("Exclusions must have unique identities")
    if not excluded.issubset(set(source_ids)):
        raise SubsectionManifestError("Exclusion references an unknown source identity")
    case_ids = [str(item.get("case_id") or "") for item in subsections if isinstance(item, dict)]
    if len(case_ids) != len(subsections) or len(case_ids) != len(set(case_ids)) or any(not value for value in case_ids):
        raise SubsectionManifestError("Subsection case IDs must be present and unique")
    oracle_by_id = {str(case.get("case_id")): case for case in oracle_cases if isinstance(case, dict)}
    if set(case_ids) != set(oracle_by_id):
        raise SubsectionManifestError("Manifest subsection universe differs from the oracle")
    document_ids = [str(item.get("document_id") or "") for item in documents if isinstance(item, dict)]
    if len(document_ids) != len(documents) or len(document_ids) != len(set(document_ids)) or any(not value for value in document_ids):
        raise SubsectionManifestError("Document IDs must be present and unique")
    for subsection in subsections:
        if not isinstance(subsection, dict):
            raise SubsectionManifestError("Manifest contains a non-object subsection")
        case = oracle_by_id[subsection["case_id"]]
        for field in ("identity", "sourcefile", "sourcepage", "subsection_id", "expected_heading", "heading_locator", "body_sha256", "body_length"):
            if subsection.get(field) != case.get(field):
                raise SubsectionManifestError(f"Subsection {subsection['case_id']} disagrees with oracle field {field}")
        if not str(subsection.get("expected_heading") or "").strip() or int(subsection.get("body_length") or 0) <= 0:
            raise SubsectionManifestError("Subsection has an empty heading or body")
        next_locator = str(subsection.get("next_heading_locator") or "")
        if next_locator and not str(subsection.get("next_heading") or "").strip():
            raise SubsectionManifestError("Subsection has an invalid next-heading boundary")
        if subsection.get("document_id") not in document_ids:
            raise SubsectionManifestError("Subsection references an unknown document")
    included = sum(item.get("status") == "included" for item in subsections)
    if manifest.get("included_subsection_count") != included:
        raise SubsectionManifestError("Included subsection count is incorrect")
    return {
        "gate": "v4_canonical_subsection_manifest",
        "schema_version": 2,
        "status": "PASS",
        "oracle_sha256": actual_oracle_hash,
        "source_count": len(sources),
        "document_count": len(documents),
        "subsection_count": len(subsections),
        "included_subsection_count": included,
        "excluded_source_count": len(excluded),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise SubsectionManifestError("Manifest must be a JSON object")
        result = validate(manifest, args.oracle)
    except (OSError, json.JSONDecodeError, SubsectionManifestError, TypeError, ValueError) as error:
        result = {"schema_version": 2, "status": "FAIL", "error": str(error)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())