import hashlib
import json

import pytest

from scripts.validate_highlight_oracle import OracleValidationError, validate


def make_report(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({
        "status": "ok",
        "schema_census": {"blocks": [
            {"kind": "heading", "locator": "h-1", "text": "1.1 The objective"},
            {"kind": "p", "locator": "p-1", "text": "The court must deal with cases justly."},
            {"kind": "heading", "locator": "h-2", "text": "1.2 Next"},
        ]},
    }), encoding="utf-8")
    body_text = "1.1 The objective The court must deal with cases justly."
    case = {
        "case_id": "case-1",
        "oracle_version": "2026-07-15",
        "identity": "source-1",
        "sourcefile": "source.html",
        "sourcepage": "Part 1",
        "subsection_id": "1.1",
        "expected_heading": "1.1 The objective",
        "heading_locator": "h-1",
        "next_heading": "1.2 Next",
        "next_heading_locator": "h-2",
        "snapshot_file": "snapshot.json",
        "body_text": body_text,
        "body_sha256": hashlib.sha256(body_text.casefold().encode("utf-8")).hexdigest(),
        "body_length": len(body_text),
    }
    return manifest, {
        "schema_version": 1,
        "oracle_version": "2026-07-15",
        "case_count": 1,
        "source_count": 1,
        "source_identities": ["source-1"],
        "snapshot_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "cases": [case],
    }


def test_validate_highlight_oracle_accepts_bound_unique_cases(tmp_path):
    manifest, report = make_report(tmp_path)
    provenance = {"release_id": "r1", "search_index": "candidate"}

    result = validate(report, tmp_path, provenance=provenance)

    assert result["gate"] == "highlight"
    assert result["case_count"] == 1
    assert result["snapshot_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert result["provenance"] == provenance


@pytest.mark.parametrize("field", ["snapshot_manifest_sha256", "oracle_version", "heading_locator", "body_sha256", "body_text", "body_length"])
def test_validate_highlight_oracle_rejects_tampered_evidence(tmp_path, field):
    _, report = make_report(tmp_path)
    if field == "snapshot_manifest_sha256":
        report[field] = "tampered"
    elif field == "oracle_version":
        report["oracle_version"] = "tampered"
    elif field == "body_sha256":
        report["cases"][0][field] = "tampered"
    elif field == "body_text":
        report["cases"][0][field] = "tampered"
    elif field == "body_length":
        report["cases"][0][field] += 1
    else:
        report["cases"][0][field] = ""

    with pytest.raises(OracleValidationError):
        validate(report, tmp_path)


def test_validate_highlight_oracle_rejects_duplicate_case_ids(tmp_path):
    _, report = make_report(tmp_path)
    report["cases"].append(dict(report["cases"][0]))
    report["case_count"] = 2

    with pytest.raises(OracleValidationError, match="Duplicate"):
        validate(report, tmp_path)