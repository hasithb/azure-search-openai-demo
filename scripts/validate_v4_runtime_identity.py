"""Fail-closed validation of the disposable v4 Container App runtime identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class RuntimeIdentityError(ValueError):
    """Raised when the candidate app does not serve the expected release."""


def _revision_name(revision: dict[str, Any]) -> str:
    name = str(revision.get("name") or revision.get("properties", {}).get("name") or "").strip()
    return name.rsplit("--", 1)[-1]


def _revision_image(revision: dict[str, Any]) -> str:
    containers = revision.get("properties", {}).get("template", {}).get("containers", [])
    if not isinstance(containers, list) or not containers:
        return ""
    return str(containers[0].get("image") or "").strip()


def _traffic_weight(revision: dict[str, Any]) -> int:
    value = revision.get("properties", {}).get("trafficWeight", revision.get("trafficWeight", 0))
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def validate_runtime_identity(
    app: dict[str, Any],
    revisions: list[dict[str, Any]],
    *,
    expected_revision: str,
    expected_image: str,
    expected_environment: dict[str, str],
) -> dict[str, Any]:
    properties = app.get("properties", {})
    observed: dict[str, Any] = {
        "app": app.get("name", ""),
        "latest_revision": properties.get("latestRevisionName", ""),
        "latest_ready_revision": properties.get("latestReadyRevisionName", ""),
        "expected_revision": expected_revision,
        "expected_image": expected_image,
        "search_index": expected_environment.get("AZURE_SEARCH_INDEX", ""),
        "knowledge_base": expected_environment.get("AZURE_SEARCH_KNOWLEDGEBASE_NAME", ""),
    }
    if properties.get("provisioningState") != "Succeeded":
        raise RuntimeIdentityError(f"Candidate app provisioning state is not Succeeded: {properties.get('provisioningState')!r}")
    candidates = [revision for revision in revisions if _revision_name(revision) == expected_revision]
    if len(candidates) != 1:
        raise RuntimeIdentityError(
            f"Expected exactly one active candidate revision; expected={expected_revision!r}, "
            f"observed={[(_revision_name(item), _revision_image(item), _traffic_weight(item)) for item in revisions]!r}"
        )
    revision = candidates[0]
    observed.update(
        {
            "active_revision": _revision_name(revision),
            "deployed_image": _revision_image(revision),
            "traffic_weight": _traffic_weight(revision),
            "running_state": revision.get("properties", {}).get("runningState", ""),
            "health_state": revision.get("properties", {}).get("healthState", ""),
        }
    )
    mismatches: list[str] = []
    if _revision_image(revision) != expected_image:
        mismatches.append("image")
    if _traffic_weight(revision) != 100:
        mismatches.append("traffic_weight")
    latest_ready_revision = str(properties.get("latestReadyRevisionName") or "").strip().rsplit("--", 1)[-1]
    if latest_ready_revision != expected_revision:
        mismatches.append("latest_ready_revision")
    if revision.get("properties", {}).get("runningState") != "Running":
        mismatches.append("running_state")
    if revision.get("properties", {}).get("healthState") != "Healthy":
        mismatches.append("health_state")
    containers = revision.get("properties", {}).get("template", {}).get("containers", [])
    revision_env = containers[0].get("env", []) if containers else []
    observed_env = {str(item.get("name")): str(item.get("value", "")) for item in revision_env if isinstance(item, dict) and item.get("name")}
    observed["environment"] = {key: observed_env.get(key, "") for key in expected_environment}
    mismatches.extend(key for key, value in expected_environment.items() if observed_env.get(key) != value)
    if mismatches:
        raise RuntimeIdentityError(
            f"Candidate runtime identity mismatch: fields={sorted(set(mismatches))}, observed={json.dumps(observed, sort_keys=True)}"
        )
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--revisions", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--environment", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_environment = dict(item.split("=", 1) for item in args.environment)
    app = json.loads(args.app.read_text(encoding="utf-8"))
    revisions = json.loads(args.revisions.read_text(encoding="utf-8"))
    result = validate_runtime_identity(
        app,
        revisions,
        expected_revision=args.revision,
        expected_image=args.image,
        expected_environment=expected_environment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())