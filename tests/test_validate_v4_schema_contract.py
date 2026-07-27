import json

import pytest

from scripts.validate_v4_schema_contract import SchemaContractError, validate_document


SCHEMAS = __import__("pathlib").Path(__file__).parents[1] / "scripts" / "schemas" / "v4"


def merged_report():
    return {
        "schema_version": 2,
        "status": "PASS",
        "replay_hash": "replay-1",
        "request_serialization_hash": "request-1",
        "case_count": 1,
        "manifest": [{"case_id": "case-1"}],
        "browser_document_observations": [],
        "coverage_summary": {"expected_cases": 1, "passed_cases": 1, "failed_cases": 0, "unique_documents": 1},
        "failures": [],
    }


def test_merged_coverage_schema_accepts_required_contract():
    validate_document(merged_report(), SCHEMAS / "merged-coverage.schema.json")


def test_merged_coverage_schema_rejects_missing_request_hash():
    report = merged_report()
    del report["request_serialization_hash"]
    with pytest.raises(SchemaContractError, match="request_serialization_hash"):
        validate_document(report, SCHEMAS / "merged-coverage.schema.json")


def test_search_reconciliation_schema_requires_hashes():
    report = {
        "schema_version": 2,
        "gate": "v4_search_reconciliation",
        "status": "PASS",
        "snapshot_sha256": "snapshot",
        "documents_sha256": "documents",
        "artifact_documents_sha256": "projected",
        "search_documents_sha256": "projected",
        "manifest_sha256": "manifest",
        "artifact_document_count": 1,
        "search_document_count": 1,
        "mappings": [{"search_document_id": "doc-1", "case_ids": ["case-1"]}],
        "failures": [],
    }
    validate_document(report, SCHEMAS / "search-reconciliation.schema.json")
    report.pop("manifest_sha256")
    with pytest.raises(SchemaContractError, match="manifest_sha256"):
        validate_document(report, SCHEMAS / "search-reconciliation.schema.json")