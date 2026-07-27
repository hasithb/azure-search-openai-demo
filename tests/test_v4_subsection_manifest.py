import copy
import json

import pytest

from scripts.build_v4_subsection_manifest import build_manifest
from scripts.validate_v4_subsection_manifest import SubsectionManifestError, validate


def make_oracle(tmp_path):
    oracle_path = tmp_path / "oracle.json"
    cases = [{
        "case_id": "case-1",
        "identity": "source-1",
        "sourcefile": "source.html",
        "sourcepage": "1.1 The objective",
        "subsection_id": "1.1",
        "expected_heading": "1.1 The objective",
        "heading_locator": "h-1",
        "next_heading": "1.2 Next",
        "next_heading_locator": "h-2",
        "body_sha256": "body-hash",
        "body_length": 12,
        "snapshot_file": "snapshot.json",
        "snapshot_content_sha256": "snapshot-hash",
    }]
    oracle_path.write_text(json.dumps({
        "oracle_version": "2",
        "snapshot_manifest_sha256": "manifest-hash",
        "cases": cases,
    }), encoding="utf-8")
    return oracle_path


def test_manifest_binds_many_subsections_to_one_document(tmp_path):
    oracle_path = make_oracle(tmp_path)
    manifest = build_manifest(oracle_path)

    result = validate(manifest, oracle_path)

    assert result["status"] == "PASS"
    assert manifest["schema_version"] == 2
    assert manifest["document_count"] == 1
    assert manifest["subsection_count"] == 1


@pytest.mark.parametrize("mutation", [
    lambda data: data["subsections"].__setitem__(0, {**data["subsections"][0], "expected_heading": ""}),
    lambda data: data["subsections"].__setitem__(0, {**data["subsections"][0], "next_heading": ""}),
    lambda data: data["subsections"].append(copy.deepcopy(data["subsections"][0])),
])
def test_manifest_rejects_incomplete_or_duplicate_cases(tmp_path, mutation):
    oracle_path = make_oracle(tmp_path)
    manifest = build_manifest(oracle_path)
    mutation(manifest)

    with pytest.raises(SubsectionManifestError):
        validate(manifest, oracle_path)


def test_manifest_rejects_stale_oracle_hash(tmp_path):
    oracle_path = make_oracle(tmp_path)
    manifest = build_manifest(oracle_path)
    manifest["oracle_sha256"] = "stale"

    with pytest.raises(SubsectionManifestError, match="bound"):
        validate(manifest, oracle_path)


def test_manifest_records_reviewed_exclusions(tmp_path):
    oracle_path = make_oracle(tmp_path)
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps([{"identity": "source-1", "reason": "not published", "reviewed_by": "release"}]), encoding="utf-8")

    manifest = build_manifest(oracle_path, exclusions)

    assert manifest["excluded_source_count"] == 1
    assert manifest["subsections"][0]["status"] == "excluded"