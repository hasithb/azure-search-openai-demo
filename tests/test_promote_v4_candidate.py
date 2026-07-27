import json

import pytest

from scripts.build_v4_evidence_bundle import EvidenceError, artifact_search_gate, build_bundle
from scripts.promote_v4_candidate import PromotionError, validate_evidence_bundle


VALID_BUNDLE = {
    "approved": True,
    "approval_environment": "Production",
    "candidate_index": "legal-court-rag-v4-prod-20260713",
    "candidate_knowledgebase": "legal-court-rag-v4-prod-20260713-agent-upgrade",
    "rollback_index": "legal-court-rag-index-v3",
    "rollback_knowledgebase": "legal-court-rag-index-v3-agent-upgrade",
    "artifact_sha256": "artifact-hash",
    "search_snapshot_sha256": "snapshot-hash",
    "fidelity": {
        "substantive_coverage": 1.0,
        "unmatched": 0,
        "ambiguous": 0,
        "unavailable": 0,
        "unclassified": 0,
    },
    "artifact_search": {"missing_count": 0, "extra_count": 0, "mismatched_count": 0},
    "candidate_validation": {"status": "PASS"},
    "candidate_runtime_identity": {
        "active_revision": "g-candidate",
        "expected_revision": "g-candidate",
        "deployed_image": "registry.azurecr.io/v4-candidate@sha256:" + "a" * 64,
        "expected_image": "registry.azurecr.io/v4-candidate@sha256:" + "a" * 64,
        "traffic_weight": 100,
        "running_state": "Running",
        "health_state": "Healthy",
        "environment": {
            "AZURE_SEARCH_INDEX": "legal-court-rag-v4-prod-20260713",
            "AZURE_SEARCH_KNOWLEDGEBASE_NAME": "legal-court-rag-v4-prod-20260713-agent-upgrade",
        },
    },
    "application_gates": {
        "schema_version": 2,
        "status": "PASS",
        "provenance": {
            "release_id": "20260713-r3",
            "git_sha": "git-hash",
            "deployment_id": "deployment-1",
            "search_index": "legal-court-rag-v4-prod-20260713",
            "knowledge_base": "legal-court-rag-v4-prod-20260713-agent-upgrade",
            "artifact_sha256": "artifact-hash",
            "search_snapshot_sha256": "snapshot-hash",
        },
        "gates": {
            "retrieval": {"gate": "retrieval", "status": "PASS", "case_count": 4},
            "category": {"gate": "category", "status": "PASS", "case_count": 4},
            "source_hierarchy": {"gate": "source_hierarchy", "status": "PASS", "case_count": 3},
            "citation": {"gate": "citation", "status": "PASS", "case_count": 4},
            "acl": {"gate": "acl", "status": "PASS", "case_count": 1},
            "highlight": {
                "gate": "highlight",
                "status": "PASS",
                "case_count": 1650,
                "source_count": 178,
            },
        },
    },
}


def test_valid_evidence_bundle_returns_exact_cutover_targets():
    targets = validate_evidence_bundle(VALID_BUNDLE)

    assert targets["candidate_index"] == VALID_BUNDLE["candidate_index"]
    assert targets["candidate_knowledgebase"] == VALID_BUNDLE["candidate_knowledgebase"]
    assert targets["artifact_sha256"] == "artifact-hash"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"approved": False}, "not approved"),
        ({"approval_environment": "Staging"}, "Production approval"),
        ({"candidate_index": "legal-court-rag-index-v3"}, "legacy v3"),
        ({"candidate_knowledgebase": "legal-court-rag-index-v3-agent-upgrade"}, "legacy v3"),
        ({"fidelity": {"substantive_coverage": 1.0, "ambiguous": 1}}, "not clean"),
        ({"fidelity": {"substantive_coverage": 0.99}}, "100%"),
        ({"application_gates": {"schema_version": 1, "status": "FAIL"}}, "Application-gate"),
        ({"application_gates": {"schema_version": 2, "status": "PASS", "provenance": {}, "gates": {}}}, "all six required gates"),
    ],
)
def test_invalid_evidence_bundle_cannot_promote(change, message):
    bundle = {**VALID_BUNDLE, **change}

    with pytest.raises(PromotionError, match=message):
        validate_evidence_bundle(bundle)


def test_candidate_knowledgebase_must_identify_index():
    bundle = {**VALID_BUNDLE, "candidate_knowledgebase": "legal-court-rag-v4-other-agent-upgrade"}

    with pytest.raises(PromotionError, match="identify the candidate index"):
        validate_evidence_bundle(bundle)


def test_evidence_builder_requires_clean_fidelity(tmp_path):
    artifact = tmp_path / "manifest.json"
    snapshot = tmp_path / "search.json"
    fidelity = tmp_path / "fidelity.json"
    transition = tmp_path / "transition.json"
    artifact.write_text(json.dumps({
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 3072,
        "document_count": 1,
        "source_count": 1,
        "snapshot_count": 1,
    }))
    (tmp_path / "documents_with_embeddings.jsonl").write_text(json.dumps({"id": "doc-1"}) + "\n")
    snapshot.write_text(json.dumps({"documents_sha256": "docs"}))
    fidelity.write_text(json.dumps({"summary": {"source_count": 2, "statuses": {"PASS": 1, "WARN": 1}}}))
    transition.write_text(json.dumps({"snapshot_count": 1, "failed_count": 0, "blocked_count": 0}))
    candidate_validation = tmp_path / "candidate-validation.json"
    candidate_validation.write_text(json.dumps({"candidate": {"status": "PASS"}, "provenance": {
        "schema_version": 1,
        "service": "search.test",
        "index": "v4",
        "captured_at_utc": "2026-07-13T00:00:00Z",
        "selected_fields": ["id", "content", "category", "sourcepage", "sourcefile", "storageUrl", "updated", "parent_id", "subsection_id", "subsections"],
        "document_count": 1,
        "documents_sha256": "snapshot-hash",
    }}))

    with pytest.raises(EvidenceError, match="schema version 2"):
        application_gates = tmp_path / "application-gates.json"
        application_gates.write_text(json.dumps({"schema_version": 1, "status": "PASS", "provenance": {
            "search_index": "legal-court-rag-v4",
            "knowledge_base": "legal-court-rag-v4-agent",
            "artifact_sha256": "placeholder",
            "search_snapshot_sha256": "placeholder",
        }, "gates": {name: {"status": "PASS"} for name in ("retrieval", "category", "source_hierarchy", "citation", "acl")}}))
        runtime_identity = tmp_path / "runtime-identity.json"
        runtime_identity.write_text(json.dumps({
            "active_revision": "g-candidate",
            "expected_revision": "g-candidate",
            "deployed_image": "registry.azurecr.io/v4-candidate@sha256:" + "a" * 64,
            "expected_image": "registry.azurecr.io/v4-candidate@sha256:" + "a" * 64,
            "traffic_weight": 100,
            "running_state": "Running",
            "health_state": "Healthy",
            "environment": {
                "AZURE_SEARCH_INDEX": "legal-court-rag-v4",
                "AZURE_SEARCH_KNOWLEDGEBASE_NAME": "legal-court-rag-v4-agent",
            },
        }))
        build_bundle(artifact, snapshot, fidelity, transition, candidate_validation, runtime_identity, "legal-court-rag-v4", "legal-court-rag-v4-agent", "v3", "v3-agent", application_gates)


def test_artifact_search_equality_gate_accepts_exact_selected_fields(tmp_path):
    manifest = tmp_path / "manifest.json"
    documents = tmp_path / "documents_with_embeddings.jsonl"
    document = {
        "id": "doc-1",
        "content": "The court must act",
        "category": "Civil Procedure Rules and Practice Directions",
        "sourcepage": "Part 1",
        "sourcefile": "Part 1",
        "storageUrl": "https://example.test/part1",
        "updated": "2026-07-13",
        "parent_id": "parent-1",
        "subsection_id": "1.1",
        "subsections": ["1.1"],
    }
    manifest.write_text("{}")
    documents.write_text(json.dumps(document) + "\n")

    result = artifact_search_gate(
        manifest,
        {"documents": [document]},
    )

    assert result["artifact_document_count"] == 1
    assert result["missing_count"] == 0
    assert result["extra_count"] == 0
    assert result["mismatched_count"] == 0


def test_artifact_search_equality_gate_rejects_content_mismatch(tmp_path):
    manifest = tmp_path / "manifest.json"
    documents = tmp_path / "documents_with_embeddings.jsonl"
    document = {"id": "doc-1", "content": "Canonical text"}
    manifest.write_text("{}")
    documents.write_text(json.dumps(document) + "\n")

    with pytest.raises(EvidenceError, match="Artifact/Search equality gate"):
        artifact_search_gate(manifest, {"documents": [{"id": "doc-1", "content": "Changed text"}]})


def test_artifact_search_equality_gate_rejects_duplicate_artifact_ids(tmp_path):
    manifest = tmp_path / "manifest.json"
    documents = tmp_path / "documents_with_embeddings.jsonl"
    manifest.write_text("{}")
    documents.write_text(json.dumps({"id": "doc-1", "content": "one"}) + "\n" + json.dumps({"id": "doc-1", "content": "two"}) + "\n")

    with pytest.raises(EvidenceError, match="Artifact/Search equality gate"):
        artifact_search_gate(manifest, {"documents": [{"id": "doc-1", "content": "one"}]})
