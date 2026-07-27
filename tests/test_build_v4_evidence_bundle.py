import pytest

from scripts.build_v4_evidence_bundle import EvidenceError, application_gate_gate, search_reconciliation_gate


PROVENANCE = {
    "release_id": "release-1",
    "git_sha": "git-1",
    "deployment_id": "deployment-1",
    "artifact_sha256": "artifact-1",
    "search_snapshot_sha256": "snapshot-1",
    "search_service": "search-1",
    "search_index": "index-1",
    "knowledge_base": "kb-1",
}


def application_report(highlight):
    gates = {
        name: {"status": "PASS", "checks": [{"name": name, "status": "PASS"}]}
        for name in ("retrieval", "category", "source_hierarchy", "citation", "acl")
    }
    gates["highlight"] = highlight
    return {"schema_version": 2, "status": "PASS", "provenance": PROVENANCE, "gates": gates}


def test_application_gate_requires_live_browser_highlight_evidence():
    with pytest.raises(EvidenceError, match="live browser evidence"):
        application_gate_gate(
            application_report({"status": "PASS", "checks": [{"name": "highlight", "status": "PASS"}]}),
            "index-1",
            "kb-1",
            "artifact-1",
            "snapshot-1",
        )


def test_application_gate_rejects_nominal_pass_without_checks():
    report = application_report({"status": "PASS", "checks": [{"name": "highlight"}], "browser_evidence": {"highlight_visible": True}})
    report["gates"]["retrieval"] = {"status": "PASS"}

    with pytest.raises(EvidenceError, match="retrieval.*substantive check evidence"):
        application_gate_gate(report, "index-1", "kb-1", "artifact-1", "snapshot-1")


def test_application_gate_accepts_live_browser_highlight_evidence():
    report = application_report(
        {
            "status": "PASS",
            "checks": [{"name": "highlight", "status": "PASS"}],
            "browser_evidence": {"highlight_visible": True},
        }
    )
    assert application_gate_gate(report, "index-1", "kb-1", "artifact-1", "snapshot-1") is report


def reconciliation_report(**overrides):
    report = {
        "schema_version": 2,
        "gate": "v4_search_reconciliation",
        "status": "PASS",
        "artifact_document_count": 2,
        "search_document_count": 2,
        "mappings": [{"search_document_id": "doc-1", "case_ids": ["case-1"]}],
        "snapshot_sha256": "snapshot-bytes",
        "documents_sha256": "snapshot-documents",
        "artifact_documents_sha256": "projected-documents",
        "search_documents_sha256": "projected-documents",
        "manifest_sha256": "manifest",
    }
    report.update(overrides)
    return report


def test_search_reconciliation_requires_equal_hash_bound_evidence():
    assert search_reconciliation_gate(reconciliation_report())["status"] == "PASS"

    with pytest.raises(EvidenceError, match="document hashes"):
        search_reconciliation_gate(reconciliation_report(search_documents_sha256="different"))

    with pytest.raises(EvidenceError, match="missing hash"):
        search_reconciliation_gate(reconciliation_report(manifest_sha256=""))