import pytest

from scripts.validate_v4_citation_coverage import CitationCoverageError, validate_report


def record(document_id="doc-1", **changes):
    value = {
        "case_id": "case-1",
        "source_revision": "rev-1",
        "source_id": "source-1",
        "document_id": document_id,
        "subsection_id": "24.2",
        "canonical_text_sha256": "hash-1",
        "rendered": True,
        "clicked": True,
        "supporting_content_count": 1,
        "primary_source_count": 1,
        "highlighted_text_sha256": "hash-1",
        "primary_source_identity": {
            "source-revision": "rev-1",
            "source-id": "source-1",
            "document-id": document_id,
            "subsection-id": "24.2",
            "canonical-text-sha256": "hash-1",
        },
    }
    value.update(changes)
    value["primary_source_identity"]["subsection-id"] = value["subsection_id"]
    return value


def valid_report(record_value=None, documents=None):
    record_value = record_value or record()
    if documents is None:
        documents = [{"id": record_value["document_id"], "source_revision": "rev-1", "canonical_text_sha256": "hash-1"}]
    return {
        "schema_version": 2,
        "replay_hash": "replay-1",
        "coverage_summary": {"expected_cases": 1, "passed_cases": 1, "failed_cases": 0, "unique_documents": 1},
        "manifest": [record_value],
        "search_documents": documents,
    }


def test_citation_coverage_enforces_exact_complete_identity_join():
    result = validate_report(valid_report())

    assert result["status"] == "PASS"
    assert result["counts"] == {"manifest": 1, "unique_documents": 1, "search_documents": 1, "rendered": 1, "clicked": 1, "supporting_content": 1, "primary_source": 1}


@pytest.mark.parametrize("changes", [{"clicked": False}, {"supporting_content_count": 2}, {"primary_source_count": 0}, {"highlighted_text_sha256": "wrong"}])
def test_citation_coverage_rejects_incomplete_ui_evidence(changes):
    with pytest.raises(CitationCoverageError):
        validate_report(valid_report(record(**changes)))


def test_citation_coverage_rejects_missing_search_document():
    with pytest.raises(CitationCoverageError, match="missing Search document"):
        validate_report(valid_report(documents=[]))


def test_citation_coverage_rejects_duplicate_identity():
    payload = valid_report()
    payload["manifest"].append(record())
    payload["search_documents"].append({"id": "doc-2", "source_revision": "rev-1", "canonical_text_sha256": "hash-1"})

    with pytest.raises(CitationCoverageError, match="Duplicate immutable"):
        validate_report(payload)


def test_schema_v2_allows_many_subsections_per_search_document():
    first = record(subsection_id="24.2")
    second = record(subsection_id="24.3")
    second["case_id"] = "case-2"
    payload = {
        "schema_version": 2,
        "replay_hash": "replay-1",
        "coverage_summary": {"expected_cases": 2, "passed_cases": 2, "failed_cases": 0, "unique_documents": 1},
        "manifest": [first, second],
        "search_documents": [{
            "id": "doc-1",
            "source_revision": "rev-1",
            "canonical_text_sha256": "hash-1",
            "subsection_id": "24.2",
            "subsections": ["24.2", "24.3"],
        }],
    }

    result = validate_report(payload)

    assert result["status"] == "PASS"
    assert result["counts"]["manifest"] == 2
    assert result["counts"]["unique_documents"] == 1


def test_schema_v2_rejects_unexpected_search_document():
    payload = valid_report()
    payload["schema_version"] = 2
    payload["search_documents"].append({"id": "doc-extra", "source_revision": "rev-1", "canonical_text_sha256": "hash-1"})

    with pytest.raises(CitationCoverageError, match="unexpected Search documents"):
        validate_report(payload)