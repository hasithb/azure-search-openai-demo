"""Reconcile canonical subsection cases with an independently captured Search snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_source_documents import INDEX_SELECT_FIELDS, load_index_snapshot


class SearchReconciliationError(ValueError):
    """Raised when Search evidence cannot be joined exactly to the manifest."""


def _project(document: dict[str, Any]) -> dict[str, Any]:
    return {field: document.get(field) for field in INDEX_SELECT_FIELDS}


def _documents_hash(documents: list[dict[str, Any]]) -> str:
    ordered = sorted(documents, key=lambda document: str(document.get("id") or ""))
    payload = json.dumps(ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_artifact(path: Path) -> list[dict[str, Any]]:
    documents = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(document, dict) for document in documents):
        raise SearchReconciliationError("Artifact contains a non-object document")
    ids = [str(document.get("id") or "").strip() for document in documents]
    if any(not document_id for document_id in ids) or len(ids) != len(set(ids)):
        raise SearchReconciliationError("Artifact contains missing or duplicate document IDs")
    return documents


def _sourcefile_matches(manifest_sourcefile: str, artifact_sourcefile: str) -> bool:
    """Allow generated descriptive source labels to extend canonical names."""
    canonical = manifest_sourcefile.strip().casefold()
    generated = artifact_sourcefile.strip().casefold()
    if generated == canonical:
        return True
    suffix = generated[len(canonical) :] if generated.startswith(canonical) else ""
    return bool(suffix) and suffix[0] in " -:"


def reconcile(manifest: dict[str, Any], snapshot_path: Path, artifact_path: Path) -> dict[str, Any]:
    if manifest.get("schema_version") != 2:
        raise SearchReconciliationError("Canonical subsection manifest must use schema version 2")
    documents, provenance = load_index_snapshot(snapshot_path)
    artifact_documents = _load_artifact(artifact_path)
    search_by_id: dict[str, dict[str, Any]] = {}
    for document in documents:
        document_id = str(document.get("id") or "").strip()
        if not document_id or document_id in search_by_id:
            raise SearchReconciliationError(f"Search snapshot has a missing or duplicate document id: {document_id}")
        search_by_id[document_id] = document

    failures: list[str] = []
    artifact_by_id = {str(document["id"]): document for document in artifact_documents}
    if set(artifact_by_id) != set(search_by_id):
        failures.append("artifact/Search document ID sets do not match")
    included_cases = [case for case in manifest.get("subsections", []) if case.get("status") == "included"]
    case_matches: dict[str, list[str]] = {str(case["case_id"]): [] for case in included_cases}
    mappings: list[dict[str, Any]] = []
    for search_id, document in search_by_id.items():
        artifact = artifact_by_id.get(search_id)
        if artifact is None:
            continue
        if _project(document) != _project(artifact):
            failures.append(f"artifact/Search projected document mismatch: {search_id}")
        if str(document.get("sourcefile") or "") != str(artifact.get("sourcefile") or ""):
            failures.append(f"artifact/Search sourcefile mismatch: {search_id}")
        if not str(document.get("content") or "").strip():
            failures.append(f"empty Search content: {search_id}")
        matched_cases = []
        content = str(artifact.get("content") or "")
        for case in included_cases:
            subsection_id = str(case.get("subsection_id") or "")
            if _sourcefile_matches(str(case.get("sourcefile") or ""), str(artifact.get("sourcefile") or "")) and (
                subsection_id.casefold() in content.casefold()
                or subsection_id.casefold() == str(artifact.get("subsection_id") or "").casefold()
                or subsection_id.casefold() in {str(value).casefold() for value in artifact.get("subsections", []) if value}
            ):
                matched_cases.append(str(case["case_id"]))
                case_matches[str(case["case_id"])].append(search_id)
        mappings.append({"search_document_id": search_id, "case_ids": sorted(matched_cases)})
    for case_id, search_ids in case_matches.items():
        if not search_ids:
            failures.append(f"included subsection has no artifact/Search document: {case_id}")
    unexpected = sorted(set(search_by_id) - set(artifact_by_id))
    if unexpected:
        failures.append("unexpected Search documents: " + ", ".join(unexpected))
    result = {
        "schema_version": 2,
        "gate": "v4_search_reconciliation",
        "status": "PASS" if not failures else "FAIL",
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "documents_sha256": provenance.get("documents_sha256"),
        "artifact_documents_sha256": _documents_hash([_project(document) for document in artifact_documents]),
        "search_documents_sha256": _documents_hash([_project(document) for document in documents]),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "artifact_document_count": len(artifact_documents),
        "search_document_count": len(documents),
        "subsection_count": len(included_cases),
        "mappings": mappings,
        "failures": failures,
    }
    if failures:
        raise SearchReconciliationError(json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise SearchReconciliationError("Manifest must be a JSON object")
        result = reconcile(manifest, args.snapshot, args.artifact)
    except (OSError, json.JSONDecodeError, SearchReconciliationError, ValueError) as error:
        result = {"schema_version": 2, "gate": "v4_search_reconciliation", "status": "FAIL", "error": str(error)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())