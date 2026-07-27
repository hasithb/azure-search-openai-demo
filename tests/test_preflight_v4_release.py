import json
from pathlib import Path

import pytest

from scripts.preflight_v4_release import PreflightError, run_preflight


FIXTURES = Path(__file__).parents[1] / "tests" / "fixtures" / "v4"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name / "preflight.json").read_text(encoding="utf-8"))


def test_ready_fixture_passes_without_promotion_eligibility():
    result = run_preflight(load_fixture("ready"))
    assert result["status"] == "PASS"
    assert result["simulation"] is True
    assert result["read_only"] is True
    assert result["promotion_eligible"] is False


def test_reconstructed_r7_fixture_fails_closed_on_empty_image():
    with pytest.raises(PreflightError, match="candidate_image is empty"):
        run_preflight(load_fixture("r7-reconstructed"))


def test_preflight_rejects_snapshot_hash_drift():
    observations = load_fixture("ready")
    observations["search_snapshot"]["documents_sha256"] = "d" * 64
    with pytest.raises(PreflightError, match="documents_sha256"):
        run_preflight(observations)


import hashlib
import json

import pytest

from scripts.preflight_v4_release import PreflightError, run_preflight


IMAGE = "registry.azurecr.io/v4-candidate@sha256:" + "a" * 64
INDEX = "legal-court-rag-v4-staging-20260725-r8"
KNOWLEDGE_BASE = INDEX + "-agent-upgrade"
SNAPSHOT_DOCUMENTS = [{"id": "doc-1", "content": "canonical"}]


def valid_payload():
    documents_hash = hashlib.sha256(json.dumps(SNAPSHOT_DOCUMENTS, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "release_id": "20260725-r8",
        "git_sha": "b" * 40,
        "candidate_app_name": "candidate-v4",
        "candidate_url": "https://candidate-v4.example.azurecontainerapps.io",
        "expected_search_index": INDEX,
        "expected_knowledge_base": KNOWLEDGE_BASE,
        "expected_revision": "g-bbbbbbbb",
        "candidate_image": IMAGE,
        "artifact_sha256": "c" * 64,
        "search_snapshot_sha256": documents_hash,
        "candidate_app": {
            "properties": {
                "provisioningState": "Succeeded",
                "latestRevisionName": "candidate-v4--g-bbbbbbbb",
                "latestReadyRevisionName": "candidate-v4--g-bbbbbbbb",
                "configuration": {"ingress": {"fqdn": "candidate-v4.example.azurecontainerapps.io"}},
            },
        },
        "candidate_revisions": [{
            "name": "candidate-v4--g-bbbbbbbb",
            "properties": {
                "trafficWeight": 100,
                "runningState": "Running",
                "healthState": "Healthy",
                "template": {"containers": [{
                    "image": IMAGE,
                    "env": [
                        {"name": "AZURE_SEARCH_INDEX", "value": INDEX},
                        {"name": "AZURE_SEARCH_KNOWLEDGEBASE_NAME", "value": KNOWLEDGE_BASE},
                    ],
                }]},
            },
        }],
        "search_snapshot": {
            "schema_version": 1,
            "index": INDEX,
            "documents": SNAPSHOT_DOCUMENTS,
            "documents_sha256": documents_hash,
        },
        "provenance": {
            "release_id": "20260725-r8",
            "git_sha": "b" * 40,
            "artifact_sha256": "c" * 64,
            "search_snapshot_sha256": documents_hash,
            "search_index": INDEX,
            "knowledge_base": KNOWLEDGE_BASE,
            "candidate_revision": "g-bbbbbbbb",
            "candidate_image": IMAGE,
        },
    }


def test_valid_ready_candidate_passes_without_promotion_eligibility():
    result = run_preflight(valid_payload())
    assert result["status"] == "PASS"
    assert result["simulation"] is True
    assert result["promotion_eligible"] is False


def test_r7_failure_reports_empty_image_before_runtime_mutation():
    payload = valid_payload()
    payload["candidate_image"] = ""
    payload["candidate_revisions"][0]["properties"]["runningState"] = "Activating"
    payload["candidate_revisions"][0]["properties"]["healthState"] = None
    payload["candidate_app"]["properties"]["latestReadyRevisionName"] = "candidate-v4--v4-previous-r7"

    with pytest.raises(PreflightError, match="candidate_image is empty"):
        run_preflight(payload)


@pytest.mark.parametrize(
    "path, value, message",
    [
        (("candidate_revisions", 0, "properties", "runningState"), "Activating", "is Activating"),
        (("candidate_revisions", 0, "properties", "healthState"), None, "health_state"),
        (("candidate_app", "properties", "latestReadyRevisionName"), "candidate-v4--old", "latest ready"),
        (("candidate_revisions", 0, "properties", "trafficWeight"), 50, "100% traffic"),
    ],
)
def test_readiness_failures_are_rejected(path, value, message):
    payload = valid_payload()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(PreflightError, match=message):
        run_preflight(payload)


def test_image_mismatch_is_rejected():
    payload = valid_payload()
    payload["candidate_revisions"][0]["properties"]["template"]["containers"][0]["image"] = "registry.azurecr.io/v4-candidate@sha256:" + "d" * 64
    with pytest.raises(PreflightError, match="image"):
        run_preflight(payload)


def test_wrong_search_binding_is_rejected():
    payload = valid_payload()
    payload["candidate_revisions"][0]["properties"]["template"]["containers"][0]["env"][0]["value"] = "wrong-index"
    with pytest.raises(PreflightError, match="AZURE_SEARCH_INDEX"):
        run_preflight(payload)