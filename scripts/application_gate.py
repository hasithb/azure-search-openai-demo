"""Validation primitives for the v4 candidate application release gate."""

from __future__ import annotations

from urllib.parse import urlparse


class ApplicationGateError(ValueError):
    """Raised when candidate application identity cannot be trusted."""


PROVENANCE_FIELDS = (
    "release_id",
    "git_sha",
    "deployment_id",
    "artifact_sha256",
    "search_snapshot_sha256",
    "search_service",
    "search_index",
    "knowledge_base",
)


def validate_candidate_url(candidate_url: str) -> str:
    """Require an explicit HTTPS candidate URL and reject local/v3 fallbacks."""
    value = candidate_url.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ApplicationGateError("Candidate application URL must be an HTTPS URL")
    hostname = (parsed.hostname or "").casefold()
    if hostname in {"localhost", "127.0.0.1", "::1"} or hostname.endswith(".localhost"):
        raise ApplicationGateError("Candidate application URL must not be local")
    if "v3" in value.casefold():
        raise ApplicationGateError("Candidate application URL must not identify a v3 deployment")
    return value


def validate_provenance(payload: object, expected: dict[str, str]) -> dict[str, str]:
    """Validate the complete provenance envelope against release expectations."""
    if not isinstance(payload, dict):
        raise ApplicationGateError("Candidate provenance must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ApplicationGateError("Candidate provenance schema version is unsupported")

    missing = [field for field in PROVENANCE_FIELDS if not str(payload.get(field) or "").strip()]
    if missing:
        raise ApplicationGateError(f"Candidate provenance is missing: {', '.join(missing)}")

    mismatched = [
        field
        for field in PROVENANCE_FIELDS
        if field in expected and str(payload.get(field)).strip() != str(expected[field]).strip()
    ]
    if mismatched:
        raise ApplicationGateError(f"Candidate provenance mismatch: {', '.join(mismatched)}")

    return {field: str(payload[field]).strip() for field in PROVENANCE_FIELDS}