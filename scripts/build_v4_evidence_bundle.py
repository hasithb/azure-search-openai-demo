"""Build a deterministic, fail-closed v4 release evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    """Raised when release evidence is incomplete or fails the fidelity gate."""


IMMUTABLE_IMAGE = re.compile(r"^[^/\s]+(?:/[^/@\s]+)+@sha256:[0-9a-fA-F]{64}$")


SEARCH_FIELDS = (
    "id",
    "content",
    "category",
    "sourcepage",
    "sourcefile",
    "storageUrl",
    "updated",
    "parent_id",
    "subsection_id",
    "subsections",
)

APPLICATION_PROVENANCE_FIELDS = (
    "release_id",
    "git_sha",
    "deployment_id",
    "artifact_sha256",
    "search_snapshot_sha256",
    "search_service",
    "search_index",
    "knowledge_base",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceError(f"Evidence input must be a JSON object: {path}")
    return payload


def fidelity_gate(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != 2:
        raise EvidenceError("Fidelity report must use schema version 2")
    if report.get("complete") is not True:
        raise EvidenceError("Fidelity report is incomplete")
    for field in ("run_id", "started_at_utc", "completed_at_utc"):
        if not isinstance(report.get(field), str) or not report[field].strip():
            raise EvidenceError(f"Fidelity report is missing {field}")
    try:
        started_at = datetime.fromisoformat(report["started_at_utc"].replace("Z", "+00:00"))
        completed_at = datetime.fromisoformat(report["completed_at_utc"].replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError("Fidelity report timestamps are invalid") from error
    if started_at.tzinfo is None or completed_at.tzinfo is None or completed_at < started_at:
        raise EvidenceError("Fidelity report timestamp ordering is invalid")
    snapshot_provenance = report.get("snapshot_provenance")
    if not isinstance(snapshot_provenance, dict) or snapshot_provenance.get("verified") is not True:
        raise EvidenceError("Fidelity report requires a verified Search snapshot provenance envelope")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise EvidenceError("Fidelity report is missing summary")
    statuses = summary.get("statuses")
    if not isinstance(statuses, dict):
        raise EvidenceError("Fidelity report is missing status counts")
    sources = report.get("sources")
    if not isinstance(sources, list):
        raise EvidenceError("Fidelity report is missing source-level evidence required for 100% coverage")
    source_count = int(summary.get("source_count", 0) or 0)
    if source_count <= 0 or report.get("expected_source_count") != source_count or report.get("processed_source_count") != source_count:
        raise EvidenceError("Fidelity report source completion counts are incomplete")
    if source_count != len(sources):
        raise EvidenceError("Fidelity source count does not match source-level evidence")
    source_keys = [
        (str(source.get("source_type") or ""), str(source.get("category") or ""), str(source.get("sourcefile") or ""))
        for source in sources
        if isinstance(source, dict)
    ]
    if len(source_keys) != len(sources) or len(set(source_keys)) != len(source_keys):
        raise EvidenceError("Fidelity source-level evidence is not a unique one-to-one reconciliation")
    if int(statuses.get("PASS", 0) or 0) != sum(source.get("status") == "PASS" for source in sources):
        raise EvidenceError("Fidelity status counts do not match source-level evidence")
    pass_count = sum(source.get("status") == "PASS" for source in sources)
    coverage = pass_count / source_count if source_count else 0.0
    unmatched = 0
    ambiguous = 0
    substantive_block_count = 0
    remediation_counts: dict[str, int] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise EvidenceError("Fidelity source evidence must be an object")
        if source.get("status") not in {"PASS"}:
            raise EvidenceError("Fidelity report contains a non-passing source")
        remediation = source.get("remediation_status")
        if not isinstance(remediation, str) or not remediation:
            raise EvidenceError("Fidelity source is missing remediation disposition")
        remediation_counts[remediation] = remediation_counts.get(remediation, 0) + 1
        blocks = source.get("metrics", {}).get("substantive_blocks", {})
        if not isinstance(blocks, dict):
            raise EvidenceError("Fidelity source is missing substantive block evidence")
        unmatched += int(blocks.get("unmatched_block_count", 0) or 0)
        ambiguous += int(blocks.get("ambiguous_block_count", 0) or 0)
        source_blocks = int(blocks.get("source_block_count", 0) or 0)
        matched_blocks = int(blocks.get("matched_block_count", 0) or 0)
        unmatched_blocks = int(blocks.get("unmatched_block_count", 0) or 0)
        ambiguous_blocks = int(blocks.get("ambiguous_block_count", 0) or 0)
        overlap_blocks = int(blocks.get("cross_document_overlap_count", 0) or 0)
        if source_blocks <= 0 or matched_blocks + unmatched_blocks + ambiguous_blocks + overlap_blocks != source_blocks:
            raise EvidenceError("Fidelity substantive block counts do not reconcile")
        if len(blocks.get("occurrence_ledger", [])) != source_blocks:
            raise EvidenceError("Fidelity occurrence ledger does not match source block count")
        substantive_block_count += source_blocks
    if report.get("remediation") != remediation_counts:
        raise EvidenceError("Fidelity remediation counts do not match source-level evidence")
    if substantive_block_count <= 0:
        raise EvidenceError("Fidelity report contains no substantive block evidence")
    gate = {
        "substantive_coverage": coverage,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "unavailable": int(statuses.get("UNAVAILABLE", 0) or 0) + int(statuses.get("MISSING_FROM_INDEX", 0) or 0),
        "unclassified": int(statuses.get("INDEX_ONLY", 0) or 0) + int(statuses.get("UNMAPPED", 0) or 0),
        "substantive_block_count": substantive_block_count,
        "dispositions": remediation_counts,
    }
    if coverage != 1.0 or any(statuses.get(status, 0) for status in ("WARN", "FAIL", "UNAVAILABLE", "MISSING_FROM_INDEX", "INDEX_ONLY", "UNMAPPED")) or any(
        gate[field] for field in ("unmatched", "ambiguous", "unavailable", "unclassified")
    ):
        raise EvidenceError(f"Fidelity gate is not clean or 100% substantive: {gate}")
    return gate


def transition_gate(report: dict[str, Any]) -> dict[str, int]:
    if not isinstance(report, dict):
        raise EvidenceError("HTML transition report must be a JSON object")
    gate = {
        "snapshot_count": int(report.get("snapshot_count", 0) or 0),
        "failed_count": int(report.get("failed_count", 0) or 0),
        "blocked_count": int(report.get("blocked_count", 0) or 0),
    }
    if not gate["snapshot_count"] or gate["failed_count"] or gate["blocked_count"]:
        raise EvidenceError(f"HTML transition gate is not clean: {gate}")
    return gate


def artifact_search_gate(artifact_path: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    documents_path = artifact_path.parent / "documents_with_embeddings.jsonl"
    if not documents_path.exists():
        raise EvidenceError(f"Artifact documents are missing: {documents_path}")
    artifact_documents = [
        json.loads(line)
        for line in documents_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    snapshot_documents = snapshot.get("documents")
    if not isinstance(snapshot_documents, list):
        raise EvidenceError("Search snapshot has no documents for artifact equality")

    def project(document: dict[str, Any]) -> dict[str, Any]:
        return {field: document.get(field) for field in SEARCH_FIELDS}

    artifact_by_id = {str(document.get("id") or ""): project(document) for document in artifact_documents}
    snapshot_by_id = {str(document.get("id") or ""): project(document) for document in snapshot_documents}
    missing = sorted(set(artifact_by_id) - set(snapshot_by_id))
    extra = sorted(set(snapshot_by_id) - set(artifact_by_id))
    mismatched = sorted(
        document_id
        for document_id in set(artifact_by_id) & set(snapshot_by_id)
        if artifact_by_id[document_id] != snapshot_by_id[document_id]
    )
    result = {
        "artifact_document_count": len(artifact_by_id),
        "search_document_count": len(snapshot_by_id),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "mismatched_count": len(mismatched),
        "missing_ids": missing[:100],
        "extra_ids": extra[:100],
        "mismatched_ids": mismatched[:100],
    }
    if missing or extra or mismatched or len(artifact_by_id) != len(artifact_documents):
        raise EvidenceError(f"Artifact/Search equality gate is not clean: {result}")
    return result


def candidate_validation_gate(report: dict[str, Any]) -> dict[str, Any]:
    candidate = report.get("candidate")
    if not isinstance(candidate, dict) or candidate.get("status") != "PASS":
        raise EvidenceError("Candidate Search validation gate is not clean")
    return candidate


def candidate_validation_matches_snapshot(report: dict[str, Any], snapshot: dict[str, Any]) -> None:
    provenance = report.get("provenance")
    expected = {
        key: snapshot.get(key)
        for key in ("schema_version", "service", "index", "captured_at_utc", "selected_fields", "document_count", "documents_sha256")
    }
    if provenance != expected:
        raise EvidenceError("Candidate validation provenance does not match the Search snapshot")


def citation_coverage_gate(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != 1 or report.get("status") != "PASS":
        raise EvidenceError("Exhaustive citation coverage report is not passing")
    counts = report.get("counts")
    case_count = int(report.get("case_count", 0) or 0)
    if not isinstance(counts, dict) or case_count <= 0:
        raise EvidenceError("Exhaustive citation coverage report has no cases or counts")
    required = ("manifest", "rendered", "clicked", "supporting_content", "primary_source")
    if any(int(counts.get(field, -1)) != case_count for field in required):
        raise EvidenceError("Exhaustive citation coverage counts do not reconcile")
    if int(counts.get("search_documents", -1)) != case_count:
        raise EvidenceError("Exhaustive citation coverage Search joins are incomplete")
    if report.get("failures") not in ([], None):
        raise EvidenceError("Exhaustive citation coverage contains failures")
    return report


def runtime_identity_gate(report: dict[str, Any], candidate_index: str, candidate_knowledgebase: str) -> dict[str, Any]:
    required = ("active_revision", "expected_revision", "deployed_image", "expected_image", "traffic_weight", "running_state", "health_state")
    if not isinstance(report, dict) or any(not str(report.get(field) or "").strip() for field in required if field not in {"traffic_weight"}):
        raise EvidenceError("Runtime identity evidence is incomplete")
    if report.get("active_revision") != report.get("expected_revision"):
        raise EvidenceError("Runtime identity active revision does not match expected revision")
    if not IMMUTABLE_IMAGE.fullmatch(str(report.get("expected_image") or "")) or report.get("deployed_image") != report.get("expected_image"):
        raise EvidenceError("Runtime identity does not prove the immutable candidate image")
    if report.get("traffic_weight") != 100 or report.get("running_state") != "Running" or report.get("health_state") != "Healthy":
        raise EvidenceError("Runtime identity is not healthy at 100% candidate traffic")
    environment = report.get("environment")
    if not isinstance(environment, dict) or environment.get("AZURE_SEARCH_INDEX") != candidate_index or environment.get("AZURE_SEARCH_KNOWLEDGEBASE_NAME") != candidate_knowledgebase:
        raise EvidenceError("Runtime identity Search environment does not match the candidate pair")
    return report


def application_gate_gate(
    report: dict[str, Any],
    candidate_index: str,
    candidate_knowledgebase: str,
    artifact_sha256: str,
    search_snapshot_sha256: str,
) -> dict[str, Any]:
    if report.get("schema_version") != 1 or report.get("status") != "PASS":
        raise EvidenceError("Application-gate report is not a passing version 1 report")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise EvidenceError("Application-gate report is missing provenance")
    missing = [field for field in APPLICATION_PROVENANCE_FIELDS if not str(provenance.get(field) or "").strip()]
    if missing:
        raise EvidenceError("Application-gate provenance is missing: " + ", ".join(missing))
    expected = {
        "search_index": candidate_index,
        "knowledge_base": candidate_knowledgebase,
        "artifact_sha256": artifact_sha256,
        "search_snapshot_sha256": search_snapshot_sha256,
    }
    mismatched = [field for field, value in expected.items() if provenance.get(field) != value]
    if mismatched:
        raise EvidenceError("Application-gate provenance mismatch: " + ", ".join(mismatched))
    gates = report.get("gates")
    if not isinstance(gates, dict) or set(gates) != {"retrieval", "category", "source_hierarchy", "citation", "acl", "highlight"}:
        raise EvidenceError("Application-gate report must contain all required gates")
    if any(not isinstance(gate, dict) or gate.get("status") != "PASS" for gate in gates.values()):
        raise EvidenceError("Application-gate report contains a non-passing gate")
    for gate_name, gate in gates.items():
        checks = gate.get("checks")
        if not isinstance(checks, list) or not checks or any(not isinstance(check, dict) for check in checks):
            raise EvidenceError(f"Application-gate {gate_name} is missing substantive check evidence")
    highlight = gates["highlight"]
    browser_evidence = highlight.get("browser_evidence")
    if not isinstance(browser_evidence, dict) or browser_evidence.get("highlight_visible") is not True:
        raise EvidenceError("Application-gate highlight is missing live browser evidence")
    return report


def build_bundle(
    artifact_manifest_path: Path,
    search_snapshot_path: Path,
    fidelity_report_path: Path,
    transition_report_path: Path,
    candidate_validation_path: Path,
    candidate_runtime_identity_path: Path,
    candidate_index: str,
    candidate_knowledgebase: str,
    rollback_index: str,
    rollback_knowledgebase: str,
    application_gate_path: Path,
    highlight_oracle_path: Path | None = None,
    citation_coverage_path: Path | None = None,
    release_index_uniqueness_path: Path | None = None,
    release_safety_path: Path | None = None,
    approved: bool = False,
    approval_environment: str = "",
) -> dict[str, Any]:
    evidence_paths = (
        artifact_manifest_path,
        search_snapshot_path,
        fidelity_report_path,
        transition_report_path,
        candidate_validation_path,
        candidate_runtime_identity_path,
        application_gate_path,
    )
    for path in evidence_paths:
        if not path.exists():
            raise EvidenceError(f"Missing release evidence input: {path}")
    if highlight_oracle_path is not None and not highlight_oracle_path.exists():
        raise EvidenceError(f"Missing highlight oracle evidence: {highlight_oracle_path}")

    manifest = _load_object(artifact_manifest_path)
    artifact_documents_path = artifact_manifest_path.parent / "documents_with_embeddings.jsonl"
    if not artifact_documents_path.exists():
        raise EvidenceError(f"Artifact documents are missing: {artifact_documents_path}")
    snapshot = _load_object(search_snapshot_path)
    report = _load_object(fidelity_report_path)
    transition_report = _load_object(transition_report_path)
    candidate_validation_report = _load_object(candidate_validation_path)
    candidate_runtime_identity_report = _load_object(candidate_runtime_identity_path)
    if manifest.get("embedding_dimensions") != 3072 or manifest.get("embedding_model") != "text-embedding-3-large":
        raise EvidenceError("Artifact embedding metadata is not the approved configuration")
    if snapshot.get("documents_sha256") is None:
        raise EvidenceError("Search snapshot is not a verified provenance envelope")
    fidelity = fidelity_gate(report)
    transition = transition_gate(transition_report)
    artifact_search = artifact_search_gate(artifact_manifest_path, snapshot)
    if citation_coverage_path is None or not citation_coverage_path.exists():
        raise EvidenceError("Missing exhaustive citation coverage evidence")
    citation_coverage = citation_coverage_gate(_load_object(citation_coverage_path))
    if release_index_uniqueness_path is None or not release_index_uniqueness_path.exists():
        raise EvidenceError("Missing read-only release-index uniqueness evidence")
    release_index_uniqueness = _load_object(release_index_uniqueness_path)
    if release_index_uniqueness.get("schema_version") != 1 or release_index_uniqueness.get("status") != "PASS" or release_index_uniqueness.get("read_only") is not True:
        raise EvidenceError("Release-index uniqueness evidence is not a read-only PASS report")
    if release_safety_path is None or not release_safety_path.exists():
        raise EvidenceError("Missing read-only release safety evidence")
    release_safety = _load_object(release_safety_path)
    if release_safety.get("schema_version") != 1 or release_safety.get("status") != "PASS" or release_safety.get("read_only") is not True:
        raise EvidenceError("Release safety evidence is not a read-only PASS report")
    candidate_validation = candidate_validation_gate(candidate_validation_report)
    candidate_validation_matches_snapshot(candidate_validation_report, snapshot)
    runtime_identity = runtime_identity_gate(candidate_runtime_identity_report, candidate_index, candidate_knowledgebase)
    application_gate_report = _load_object(application_gate_path)
    application_gates = application_gate_gate(
        application_gate_report,
        candidate_index,
        candidate_knowledgebase,
        sha256_file(artifact_documents_path),
        sha256_file(search_snapshot_path),
    )
    bundle = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "approved": approved,
        "approval_environment": approval_environment,
        "candidate_index": candidate_index,
        "candidate_knowledgebase": candidate_knowledgebase,
        "rollback_index": rollback_index,
        "rollback_knowledgebase": rollback_knowledgebase,
        "artifact_path": str(artifact_documents_path),
        "artifact_manifest_path": str(artifact_manifest_path),
        "artifact_sha256": sha256_file(artifact_documents_path),
        "search_snapshot_path": str(search_snapshot_path),
        "search_snapshot_sha256": sha256_file(search_snapshot_path),
        "search_snapshot_documents_sha256": snapshot["documents_sha256"],
        "fidelity_report_path": str(fidelity_report_path),
        "fidelity": fidelity,
        "transition": transition,
        "artifact_search": artifact_search,
        "candidate_validation": candidate_validation,
        "candidate_runtime_identity": runtime_identity,
        "application_gates": application_gates,
        "citation_coverage": citation_coverage,
        "release_index_uniqueness": release_index_uniqueness,
        "release_safety": release_safety,
        "application_provenance": application_gates["provenance"],
        "highlight_oracle_path": str(highlight_oracle_path) if highlight_oracle_path else "",
        "artifact": {
            "document_count": manifest.get("document_count"),
            "source_count": manifest.get("source_count"),
            "snapshot_count": manifest.get("snapshot_count"),
            "source_snapshot_hashes": manifest.get("source_snapshot_hashes", {}),
        },
    }
    canonical = json.dumps(
        {key: value for key, value in bundle.items() if key not in {"created_at_utc", "approved", "approval_environment"}},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    bundle["evidence_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--search-snapshot", type=Path, required=True)
    parser.add_argument("--fidelity-report", type=Path, required=True)
    parser.add_argument("--transition-report", type=Path, required=True)
    parser.add_argument("--candidate-validation", type=Path, required=True)
    parser.add_argument("--candidate-runtime-identity", type=Path, required=True)
    parser.add_argument("--application-gates", type=Path, required=True)
    parser.add_argument("--highlight-oracle", type=Path)
    parser.add_argument("--citation-coverage", type=Path, required=True)
    parser.add_argument("--release-index-uniqueness", type=Path, required=True)
    parser.add_argument("--release-safety", type=Path, required=True)
    parser.add_argument("--candidate-index", required=True)
    parser.add_argument("--candidate-knowledgebase", required=True)
    parser.add_argument("--rollback-index", default="legal-court-rag-index-v3")
    parser.add_argument("--rollback-knowledgebase", default="legal-court-rag-index-v3-agent-upgrade")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--approval-environment", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = build_bundle(
        args.artifact_manifest,
        args.search_snapshot,
        args.fidelity_report,
        args.transition_report,
        args.candidate_validation,
        args.candidate_runtime_identity,
        args.candidate_index,
        args.candidate_knowledgebase,
        args.rollback_index,
        args.rollback_knowledgebase,
        application_gate_path=args.application_gates,
        highlight_oracle_path=args.highlight_oracle,
        citation_coverage_path=args.citation_coverage,
        release_index_uniqueness_path=args.release_index_uniqueness,
        release_safety_path=args.release_safety,
        approved=args.approved,
        approval_environment=args.approval_environment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "artifact_sha256": bundle["artifact_sha256"], "search_snapshot_sha256": bundle["search_snapshot_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
