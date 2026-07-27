import json

import pytest

from scripts.audit_source_documents import write_index_snapshot
from scripts.build_v4_subsection_manifest import build_manifest
from scripts.reconcile_v4_search_snapshot import SearchReconciliationError, reconcile


def test_reconcile_requires_explicit_search_binding(tmp_path):
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"cases": [{
        "case_id": "case-1", "identity": "source-1", "sourcefile": "source.html", "sourcepage": "1.1",
        "subsection_id": "1.1", "expected_heading": "1.1", "heading_locator": "h1", "body_sha256": "hash",
        "body_length": 1, "snapshot_file": "snapshot.json", "snapshot_content_sha256": "snapshot-hash",
    }]}), encoding="utf-8")
    manifest = build_manifest(oracle)
    snapshot = tmp_path / "search.json"
    write_index_snapshot(snapshot, [{"id": "search-1", "content": "1.1", "sourcefile": "source.html"}], "service", "index")

    artifact = tmp_path / "artifact.jsonl"
    artifact.write_text(json.dumps({"id": "search-1", "content": "unrelated content", "sourcefile": "source.html"}) + "\n", encoding="utf-8")

    with pytest.raises(SearchReconciliationError, match="included subsection"):
        reconcile(manifest, snapshot, artifact)


def test_reconcile_accepts_many_cases_for_one_search_document(tmp_path):
    oracle = tmp_path / "oracle.json"
    cases = []
    for case_id, subsection in (("case-1", "1.1"), ("case-2", "1.2")):
        cases.append({
            "case_id": case_id, "identity": "source-1", "sourcefile": "source.html", "sourcepage": subsection,
            "subsection_id": subsection, "expected_heading": subsection, "heading_locator": f"h{subsection}",
            "body_sha256": f"hash-{subsection}", "body_length": 1, "snapshot_file": "snapshot.json",
            "snapshot_content_sha256": "snapshot-hash",
        })
    oracle.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    manifest = build_manifest(oracle)
    snapshot = tmp_path / "search.json"
    write_index_snapshot(snapshot, [{"id": "search-1", "content": "1.1 1.2", "sourcefile": "source.html"}], "service", "index")

    artifact = tmp_path / "artifact.jsonl"
    artifact.write_text(json.dumps({"id": "search-1", "content": "1.1 1.2", "sourcefile": "source.html"}) + "\n", encoding="utf-8")

    result = reconcile(manifest, snapshot, artifact)

    assert result["status"] == "PASS"
    assert result["mappings"][0]["case_ids"] == ["case-1", "case-2"]
    assert result["artifact_documents_sha256"] == result["search_documents_sha256"]


def test_reconcile_accepts_descriptive_generated_sourcefile(tmp_path):
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"cases": [{
        "case_id": "case-1", "identity": "source-1", "sourcefile": "Practice Direction 59", "sourcepage": "7.11",
        "subsection_id": "7.11", "expected_heading": "7.11", "heading_locator": "h1", "body_sha256": "hash",
        "body_length": 1, "snapshot_file": "snapshot.json", "snapshot_content_sha256": "snapshot-hash",
    }]}), encoding="utf-8")
    manifest = build_manifest(oracle)
    snapshot = tmp_path / "search.json"
    document = {
        "id": "search-1", "content": "7.11 The court may...", "sourcefile": "Practice Direction 59 - Circuit Commercial Courts"
    }
    write_index_snapshot(snapshot, [document], "service", "index")
    artifact = tmp_path / "artifact.jsonl"
    artifact.write_text(json.dumps(document) + "\n", encoding="utf-8")

    result = reconcile(manifest, snapshot, artifact)

    assert result["status"] == "PASS"
    assert result["mappings"][0]["case_ids"] == ["case-1"]


def test_reconcile_accepts_punctuation_separated_sourcefile(tmp_path):
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"cases": [{
        "case_id": "case-1", "identity": "source-1", "sourcefile": "Practice Direction 49C", "sourcepage": "9.4",
        "subsection_id": "9.4", "expected_heading": "9.4", "heading_locator": "h1", "body_sha256": "hash",
        "body_length": 1, "snapshot_file": "snapshot.json", "snapshot_content_sha256": "snapshot-hash",
    }]}), encoding="utf-8")
    manifest = build_manifest(oracle)
    snapshot = tmp_path / "search.json"
    document = {
        "id": "search-1", "content": "9.4 The claimant must...",
        "sourcefile": "Practice direction 49C: Consumer Credit Act 2006",
    }
    write_index_snapshot(snapshot, [document], "service", "index")
    artifact = tmp_path / "artifact.jsonl"
    artifact.write_text(json.dumps(document) + "\n", encoding="utf-8")

    result = reconcile(manifest, snapshot, artifact)

    assert result["status"] == "PASS"
    assert result["mappings"][0]["case_ids"] == ["case-1"]


def test_reconcile_rejects_any_projected_search_field_mismatch(tmp_path):
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"cases": [{
        "case_id": "case-1", "identity": "source-1", "sourcefile": "source.html", "sourcepage": "1.1",
        "subsection_id": "1.1", "expected_heading": "1.1", "heading_locator": "h1", "body_sha256": "hash",
        "body_length": 1, "snapshot_file": "snapshot.json", "snapshot_content_sha256": "snapshot-hash",
    }]}), encoding="utf-8")
    manifest = build_manifest(oracle)
    snapshot = tmp_path / "search.json"
    document = {"id": "search-1", "content": "1.1", "sourcefile": "source.html", "category": "legal"}
    write_index_snapshot(snapshot, [document], "service", "index")
    artifact = tmp_path / "artifact.jsonl"
    artifact.write_text(json.dumps({**document, "category": "different"}) + "\n", encoding="utf-8")

    with pytest.raises(SearchReconciliationError, match="projected document mismatch"):
        reconcile(manifest, snapshot, artifact)