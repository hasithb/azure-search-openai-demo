import pytest

from scripts.gate_highlight_browser import (
    BrowserGateError,
    build_replay_hash,
    build_request_serialization_hash,
    merge_exhaustive_browser_reports,
    validate_highlight_boundaries,
)


def oracle():
    return {
        "cases": [
            {
                "case_id": "case-1",
                "sourcefile": "part-24",
                "sourcepage": "Part 24",
                "subsection_id": "24.2",
            },
            {
                "case_id": "case-2",
                "sourcefile": "part-24",
                "sourcepage": "Part 24",
                "subsection_id": "24.3",
            },
        ]
    }


def shard(case_id, document_id):
    return {
        "schema_version": 2,
        "replay_hash": build_replay_hash(oracle()),
        "request_serialization_hash": build_request_serialization_hash(oracle()),
        "manifest": [{"case_id": case_id, "document_id": document_id}],
        "browser_document_observations": [{"id": document_id, "source_revision": "rev-1"}],
    }


def test_replay_hash_is_deterministic_and_tracks_case_inputs():
    first = build_replay_hash(oracle())
    assert first == build_replay_hash(oracle())
    changed = oracle()
    changed["cases"][0]["subsection_id"] = "24.4"
    assert build_replay_hash(changed) != first


def test_merge_requires_complete_unique_case_coverage():
    result = merge_exhaustive_browser_reports(
        oracle(), [shard("case-2", "doc-2"), shard("case-1", "doc-1")]
    )

    assert result["status"] == "PASS"
    assert [record["case_id"] for record in result["manifest"]] == ["case-1", "case-2"]
    assert result["coverage_summary"] == {
        "expected_cases": 2,
        "passed_cases": 2,
        "failed_cases": 0,
        "unique_documents": 2,
    }


@pytest.mark.parametrize(
    "reports, message",
    [
        ([shard("case-1", "doc-1"), shard("case-1", "doc-1")], "duplicate"),
        ([shard("case-1", "doc-1")], "missing"),
    ],
)
def test_merge_rejects_duplicate_or_missing_cases(reports, message):
    with pytest.raises(BrowserGateError, match=message):
        merge_exhaustive_browser_reports(oracle(), reports)


def test_merge_rejects_mixed_request_serialization_hashes():
    mismatched = shard("case-2", "doc-2")
    mismatched["request_serialization_hash"] = "different-request-hash"
    with pytest.raises(BrowserGateError, match="request serialization hash"):
        merge_exhaustive_browser_reports(oracle(), [shard("case-1", "doc-1"), mismatched])


def test_highlight_boundaries_reject_adjacent_content():
    validate_highlight_boundaries("target subsection", "previous subsection", "next subsection")

    with pytest.raises(BrowserGateError, match="preceding"):
        validate_highlight_boundaries("previous subsection target", "previous subsection", "next")
    with pytest.raises(BrowserGateError, match="following"):
        validate_highlight_boundaries("target next subsection", "previous", "next subsection")
