import pytest

from scripts.application_gate import ApplicationGateError, validate_candidate_url, validate_provenance


VALID_PROVENANCE = {
    "schema_version": 1,
    "release_id": "release-123",
    "git_sha": "abc123",
    "deployment_id": "candidate-456",
    "artifact_sha256": "artifact-hash",
    "search_snapshot_sha256": "snapshot-hash",
    "search_service": "search-service",
    "search_index": "legal-court-rag-v4-release-123",
    "knowledge_base": "legal-court-rag-v4-release-123-agent-upgrade",
}

EXPECTED = {field: VALID_PROVENANCE[field] for field in VALID_PROVENANCE if field != "schema_version"}


def test_validate_candidate_url_accepts_explicit_staging_https_url():
    assert validate_candidate_url("https://candidate.example.test/") == "https://candidate.example.test"


@pytest.mark.parametrize(
    "candidate_url, message",
    [
        ("", "HTTPS URL"),
        ("http://candidate.example.test", "HTTPS URL"),
        ("https://localhost:50505", "must not be local"),
        ("https://legal-rag-v3.example.test", "must not identify a v3"),
    ],
)
def test_validate_candidate_url_rejects_unsafe_fallbacks(candidate_url, message):
    with pytest.raises(ApplicationGateError, match=message):
        validate_candidate_url(candidate_url)


def test_validate_provenance_accepts_complete_matching_payload():
    assert validate_provenance(VALID_PROVENANCE, EXPECTED) == EXPECTED


@pytest.mark.parametrize(
    "change, message",
    [
        ({"schema_version": 2}, "schema version"),
        ({"release_id": ""}, "missing: release_id"),
        ({"search_index": "legal-court-rag-index-v3"}, "mismatch: search_index"),
    ],
)
def test_validate_provenance_rejects_untrusted_payloads(change, message):
    payload = {**VALID_PROVENANCE, **change}
    with pytest.raises(ApplicationGateError, match=message):
        validate_provenance(payload, EXPECTED)