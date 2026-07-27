import pytest

from scripts.validate_v4_citation_coverage import CitationCoverageError, validate_report


def record(document_id="doc-1", **changes):
    value = {
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
    }
    value.update(changes)
    return value


def valid_report(record_value=None, documents=None):
    record_value = record_value or record()
    if documents is None:
        documents = [{"id": record_value["document_id"], "source_revision": "rev-1", "canonical_text_sha256": "hash-1"}]
    return {
        "schema_version": 1,
        "manifest": [record_value],
        "search_documents": documents,
    }


def test_citation_coverage_enforces_exact_complete_identity_join():
    result = validate_report(valid_report())

    assert result["status"] == "PASS"
    assert result["counts"] == {"manifest": 1, "search_documents": 1, "rendered": 1, "clicked": 1, "supporting_content": 1, "primary_source": 1}


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