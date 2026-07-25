"""Wait for a v4 Container App revision to become ready without mutating Azure."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


class ReadinessError(ValueError):
    """Raised when a candidate cannot become ready within the bounded poll."""


def _revision_name(revision: dict[str, Any]) -> str:
    value = revision.get("name") or revision.get("properties", {}).get("name") or ""
    return str(value).strip().rsplit("--", 1)[-1]


def _readiness_state(app: dict[str, Any], revisions: list[dict[str, Any]], expected_revision: str) -> tuple[bool, str]:
    expected = expected_revision.strip().rsplit("--", 1)[-1]
    properties = app.get("properties", {})
    provisioning_state = properties.get("provisioningState")
    if provisioning_state in {"Failed", "Canceled"}:
        raise ReadinessError(f"Candidate app provisioning state is terminal: {provisioning_state}")
    latest_ready = str(properties.get("latestReadyRevisionName") or "").strip().rsplit("--", 1)[-1]
    matching = [revision for revision in revisions if _revision_name(revision) == expected]
    if len(matching) > 1:
        raise ReadinessError(f"Expected exactly one candidate revision, found {len(matching)}")
    if not matching:
        return False, f"candidate revision {expected} is not listed yet; latest ready is {latest_ready or '<missing>'}"
    revision_properties = matching[0].get("properties", {})
    running_state = revision_properties.get("runningState")
    health_state = revision_properties.get("healthState")
    if running_state in {"Failed", "Stopped"} or health_state in {"Unhealthy", "Failed"}:
        raise ReadinessError(
            f"Candidate revision {expected} reached a terminal state: running={running_state!r}, health={health_state!r}"
        )
    if latest_ready != expected or running_state != "Running" or health_state != "Healthy":
        return False, (
            f"candidate revision {expected} is not ready: latest_ready={latest_ready or '<missing>'}, "
            f"running={running_state or '<missing>'}, health={health_state or '<missing>'}"
        )
    return True, f"candidate revision {expected} is ready"


def wait_for_readiness(
    read_app: Callable[[], dict[str, Any]],
    read_revisions: Callable[[], list[dict[str, Any]]],
    *,
    expected_revision: str,
    attempts: int = 60,
    sleep_seconds: float = 10,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if attempts < 1:
        raise ReadinessError("Readiness attempts must be positive")
    last_message = ""
    for attempt in range(1, attempts + 1):
        try:
            app = read_app()
            revisions = read_revisions()
            ready, message = _readiness_state(app, revisions, expected_revision)
        except ReadinessError:
            raise
        except Exception as error:
            ready = False
            message = f"read failed: {error}"
        last_message = message
        if ready:
            return {
                "status": "READY",
                "attempt": attempt,
                "expected_revision": expected_revision.rsplit("--", 1)[-1],
                "message": message,
            }
        if attempt < attempts:
            sleep_fn(sleep_seconds)
    raise ReadinessError(f"Candidate readiness timed out after {attempts} attempts: {last_message}")


def azure_reader(resource_group: str, app_name: str, revisions: bool = False) -> dict[str, Any] | list[dict[str, Any]]:
    command = ["az", "containerapp", "revision" if revisions else "show"]
    if revisions:
        command.append("list")
    command.extend(["--resource-group", resource_group, "--name", app_name, "--only-show-errors", "-o", "json"])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    if revisions and not isinstance(payload, list):
        raise ReadinessError("Azure revision list response must be an array")
    if not revisions and not isinstance(payload, dict):
        raise ReadinessError("Azure Container App response must be an object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--app", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--attempts", type=int, default=60)
    parser.add_argument("--sleep-seconds", type=float, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = wait_for_readiness(
        lambda: azure_reader(args.resource_group, args.app),
        lambda: azure_reader(args.resource_group, args.app, revisions=True),
        expected_revision=args.revision,
        attempts=args.attempts,
        sleep_seconds=args.sleep_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())