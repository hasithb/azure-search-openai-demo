"""Run a non-mutating preflight against captured v4 release observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
from pathlib import Path
from typing import Any


class PreflightError(ValueError):
    """Raised when captured release observations are not safe to use."""


HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
IMAGE_PATTERN = re.compile(r"^[^/\s]+(?:/[^/@\s]+)+@sha256:[0-9a-fA-F]{64}$")
RELEASE_PATTERN = re.compile(r"^[0-9]{8}-r[0-9]+$")


def normalize_revision_name(value: Any) -> str:
    return str(value or "").strip().rsplit("--", 1)[-1]


def require_hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not HASH_PATTERN.fullmatch(normalized):
        raise PreflightError(f"{field} must be a non-empty SHA-256 hexadecimal value")
    return normalized.lower()


def require_image(value: Any, field: str) -> str:
    image = str(value or "").strip()
    if not image:
        raise PreflightError(
            "PRECHECK FAILED: candidate_image is empty in audit job environment. "
            "Producer: deploy-candidate-app.steps.image.outputs.candidate_image. "
            "Expected: a non-empty immutable ACR image reference. "
            "Action: stop before Container App update and workflow dispatch."
        )
    if not IMAGE_PATTERN.fullmatch(image):
        raise PreflightError(f"{field} must be an immutable ACR image reference including @sha256:digest")
    return image


def validate_url(candidate_url: str, fqdn: str) -> None:
    parsed = urllib.parse.urlparse(candidate_url.strip())
    if parsed.scheme != "https" or not parsed.hostname:
        raise PreflightError("Candidate URL must be an HTTPS origin with a hostname")
    if parsed.hostname != fqdn.strip():
        raise PreflightError(f"Candidate URL host {parsed.hostname!r} does not match candidate FQDN {fqdn!r}")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise PreflightError("Candidate URL must be an origin URL without a path, query, or fragment")
    if parsed.hostname.casefold() in {"localhost", "127.0.0.1", "::1"}:
        raise PreflightError("Candidate URL must not point to localhost")


def validate_snapshot(snapshot: dict[str, Any], expected_index: str) -> str:
    if snapshot.get("schema_version") != 1:
        raise PreflightError("Search snapshot must use schema_version 1")
    if str(snapshot.get("index") or "") != expected_index:
        raise PreflightError("Search snapshot index does not match the expected staging index")
    documents = snapshot.get("documents")
    if not isinstance(documents, list):
        raise PreflightError("Search snapshot documents must be a JSON array")
    supplied_hash = require_hash(snapshot.get("documents_sha256"), "Search snapshot documents_sha256")
    canonical = json.dumps(documents, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    actual_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if supplied_hash != actual_hash:
        raise PreflightError("Search snapshot documents_sha256 does not match its documents")
    return supplied_hash


def validate_runtime(
    app: dict[str, Any],
    revisions: list[dict[str, Any]],
    *,
    expected_revision: str,
    expected_image: str,
    expected_environment: dict[str, str],
) -> None:
    properties = app.get("properties", {})
    if properties.get("provisioningState") != "Succeeded":
        raise PreflightError(f"Candidate app provisioning state is {properties.get('provisioningState')!r}")
    normalized_expected = normalize_revision_name(expected_revision)
    latest = normalize_revision_name(properties.get("latestRevisionName"))
    latest_ready = normalize_revision_name(properties.get("latestReadyRevisionName"))
    if latest != normalized_expected:
        raise PreflightError(f"Active revision {latest!r} does not match expected revision {normalized_expected!r}")
    if latest_ready != normalized_expected:
        raise PreflightError(
            f"PRECHECK FAILED: candidate revision {normalized_expected} is not latest ready. "
            f"latest_ready_revision is {latest_ready or '<missing>'}, likely still the previous v4 revision. "
            "Action: wait for bounded readiness or fail before runtime validation."
        )
    matching = [item for item in revisions if normalize_revision_name(item.get("name") or item.get("properties", {}).get("name")) == normalized_expected]
    if len(matching) != 1:
        raise PreflightError(f"Expected exactly one candidate revision; found {len(matching)}")
    revision = matching[0]
    revision_properties = revision.get("properties", {})
    running_state = revision_properties.get("runningState")
    health_state = revision_properties.get("healthState")
    if running_state != "Running":
        raise PreflightError(
            f"PRECHECK FAILED: candidate revision {normalized_expected} is {running_state or '<missing>'}. "
            f"latest_ready_revision is {latest_ready or '<missing>'}. "
            "Action: wait for bounded readiness or fail before runtime validation."
        )
    if health_state != "Healthy":
        raise PreflightError(f"Candidate revision {normalized_expected} health_state is {health_state!r}, expected 'Healthy'")
    if int(revision_properties.get("trafficWeight", revision_properties.get("traffic_weight", 0)) or 0) != 100:
        raise PreflightError(f"Candidate revision {normalized_expected} does not receive 100% traffic")
    containers = revision_properties.get("template", {}).get("containers", [])
    if not isinstance(containers, list) or len(containers) != 1:
        raise PreflightError("Candidate revision must expose exactly one container")
    container = containers[0]
    if str(container.get("image") or "").strip() != expected_image:
        raise PreflightError("Observed candidate image does not match the expected immutable image")
    observed_environment = {
        str(item.get("name")): str(item.get("value", ""))
        for item in container.get("env", [])
        if isinstance(item, dict) and item.get("name")
    }
    for name, value in expected_environment.items():
        if observed_environment.get(name) != value:
            raise PreflightError(f"Candidate environment {name} does not match the expected value")


def run_preflight(payload: dict[str, Any]) -> dict[str, Any]:
    release_id = str(payload.get("release_id") or "").strip()
    if not RELEASE_PATTERN.fullmatch(release_id):
        raise PreflightError("Release ID must match YYYYMMDD-rN")
    candidate_app = str(payload.get("candidate_app_name") or "").strip()
    if not candidate_app or "v3" in candidate_app.casefold():
        raise PreflightError("Candidate app is missing or targets the blocked v3 environment")
    expected_index = str(payload.get("expected_search_index") or "").strip()
    expected_knowledge_base = str(payload.get("expected_knowledge_base") or "").strip()
    if "v4" not in expected_index.casefold() or "staging" not in expected_index.casefold():
        raise PreflightError("Expected Search index must be a v4 staging index")
    if expected_index not in expected_knowledge_base or "v4" not in expected_knowledge_base.casefold():
        raise PreflightError("Expected knowledge base must identify the v4 staging index")
    candidate_app_json = payload.get("candidate_app")
    if not isinstance(candidate_app_json, dict):
        raise PreflightError("candidate_app must be a JSON object")
    ingress = candidate_app_json.get("properties", {}).get("configuration", {}).get("ingress", {})
    fqdn = str(ingress.get("fqdn") or "").strip()
    if not fqdn:
        raise PreflightError("Candidate app has no ingress FQDN")
    validate_url(str(payload.get("candidate_url") or ""), fqdn)
    artifact_sha = require_hash(payload.get("artifact_sha256"), "artifact_sha256")
    snapshot_sha = validate_snapshot(payload.get("search_snapshot") or {}, expected_index)
    supplied_snapshot_sha = require_hash(payload.get("search_snapshot_sha256"), "search_snapshot_sha256")
    if supplied_snapshot_sha != snapshot_sha:
        raise PreflightError("Provided Search snapshot SHA does not match the verified snapshot")
    expected_image = require_image(payload.get("candidate_image"), "candidate_image")
    validate_runtime(
        candidate_app_json,
        payload.get("candidate_revisions") or [],
        expected_revision=str(payload.get("expected_revision") or ""),
        expected_image=expected_image,
        expected_environment={
            "AZURE_SEARCH_INDEX": expected_index,
            "AZURE_SEARCH_KNOWLEDGEBASE_NAME": expected_knowledge_base,
        },
    )
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise PreflightError("Candidate provenance must be a JSON object")
    expected_provenance = {
        "release_id": release_id,
        "git_sha": str(payload.get("git_sha") or "").strip(),
        "artifact_sha256": artifact_sha,
        "search_snapshot_sha256": snapshot_sha,
        "search_index": expected_index,
        "knowledge_base": expected_knowledge_base,
        "candidate_revision": normalize_revision_name(payload.get("expected_revision")),
        "candidate_image": expected_image,
    }
    for field, value in expected_provenance.items():
        if not value or provenance.get(field) != value:
            raise PreflightError(f"Candidate provenance does not match {field}")
    return {
        "simulation": True,
        "promotion_eligible": False,
        "status": "PASS",
        "release_id": release_id,
        "candidate_app": candidate_app,
        "expected_revision": normalize_revision_name(payload.get("expected_revision")),
        "candidate_image": expected_image,
        "artifact_sha256": artifact_sha,
        "search_snapshot_sha256": snapshot_sha,
    }


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PreflightError("Preflight input must be a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_preflight(load_payload(args.input))
    except (OSError, json.JSONDecodeError, PreflightError) as error:
        print(f"PRECHECK FAILED: {error}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())