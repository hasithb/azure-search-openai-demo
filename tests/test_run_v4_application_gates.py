import json

import pytest

from scripts.run_v4_application_gates import ApplicationGatesError, load_gate_reports

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


def write_report(tmp_path, name, status="PASS", provenance=None):
    path = tmp_path / f"{name}.json"
    payload = {"status": status, "checks": [{"name": name, "status": "PASS"}], "provenance": provenance or PROVENANCE}
    if name == "highlight":
        payload.update({
            "gate": "highlight",
            "oracle_version": "2026-07-15",
            "case_count": 10,
            "source_count": 2,
            "snapshot_manifest_sha256": "manifest-hash",
            "browser_evidence": {"highlight_visible": True, "replay_hash": "replay-1", "request_serialization_hash": "request-1"},
        })
        payload["checks"] = [{"id": "canonical_citation_highlight", "status": "PASS"}]
    path.write_text(json.dumps(payload))
    return f"{name}={path}"


def test_load_gate_reports_requires_all_release_gates(tmp_path):
    reports = load_gate_reports(
        [write_report(tmp_path, name) for name in ("retrieval", "category", "source_hierarchy", "citation", "acl", "highlight")],
        expected_provenance=PROVENANCE,
    )

    assert tuple(reports) == ("retrieval", "category", "source_hierarchy", "citation", "acl", "highlight")


def test_application_gate_report_schema_is_v2():
    from scripts.run_v4_application_gates import run

    assert run.__annotations__["return"] == "dict[str, Any]"


@pytest.mark.parametrize(
    "items, message",
    [
        ([write_report.__name__], "name=path"),
        (["retrieval=/missing.json"], "Cannot load retrieval"),
        (["retrieval=/missing.json", "category=/missing.json", "source_hierarchy=/missing.json", "citation=/missing.json"], "Cannot load retrieval"),
    ],
)
def test_load_gate_reports_fails_closed(tmp_path, items, message):
    if items == [write_report.__name__]:
        items = [write_report.__name__]
    with pytest.raises(ApplicationGatesError, match=message):
        load_gate_reports(items)


def test_load_gate_reports_rejects_skipped_gate(tmp_path):
    items = [write_report(tmp_path, name) for name in ("retrieval", "category", "source_hierarchy", "citation", "acl")]
    skipped = tmp_path / "highlight.json"
    skipped.write_text(json.dumps({"status": "SKIPPED"}))
    items.append(f"highlight={skipped}")

    with pytest.raises(ApplicationGatesError, match="highlight.*status PASS"):
        load_gate_reports(items)


def test_load_gate_reports_rejects_incomplete_highlight_oracle(tmp_path):
    items = [write_report(tmp_path, name) for name in ("retrieval", "category", "source_hierarchy", "citation", "acl")]
    incomplete = tmp_path / "highlight.json"
    incomplete.write_text(json.dumps({"status": "PASS", "gate": "highlight"}))
    items.append(f"highlight={incomplete}")

    with pytest.raises(ApplicationGatesError, match="missing oracle evidence"):
        load_gate_reports(items)


def test_load_gate_reports_accepts_exhaustive_schema_v2_highlight_coverage(tmp_path):
    items = [write_report(tmp_path, name) for name in ("retrieval", "category", "source_hierarchy", "citation", "acl")]
    payload = {
        "schema_version": 2,
        "gate": "highlight",
        "status": "PASS",
        "oracle_version": "2",
        "case_count": 2,
        "source_count": 1,
        "snapshot_manifest_sha256": "manifest-hash",
        "replay_hash": "replay-1",
        "request_serialization_hash": "request-1",
        "coverage": {
            "schema_version": 2,
            "status": "PASS",
            "case_count": 2,
            "replay_hash": "replay-1",
            "request_serialization_hash": "request-1",
            "coverage_summary": {"expected_cases": 2, "passed_cases": 2, "failed_cases": 0, "unique_documents": 1},
        },
        "checks": [{"id": "exhaustive", "status": "PASS"}],
    }
    path = tmp_path / "highlight.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    items.append(f"highlight={path}")

    reports = load_gate_reports(items)

    assert reports["highlight"]["coverage"]["status"] == "PASS"


def test_load_gate_reports_rejects_stale_provenance(tmp_path):
    items = [
        write_report(
            tmp_path,
            name,
            provenance={**PROVENANCE, "search_index": "stale-index"},
        )
        for name in ("retrieval", "category", "source_hierarchy", "citation", "acl", "highlight")
    ]

    with pytest.raises(ApplicationGatesError, match="retrieval provenance mismatch: search_index"):
        load_gate_reports(items, expected_provenance=PROVENANCE)