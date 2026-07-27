"""Validate immutable release-index naming from a read-only Search inventory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


class ReleaseIndexError(ValueError):
    """Raised when a release target is missing, duplicated, or reused."""


RELEASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{1,80}$")


def expected_names(release_id: str) -> tuple[str, str]:
    if not RELEASE_ID_RE.fullmatch(release_id):
        raise ReleaseIndexError("release_id contains unsupported characters or length")
    prefix = f"legal-court-rag-v4-staging-{release_id}"
    return prefix, f"{prefix}-agent-upgrade"


def _names(payload: dict[str, Any]) -> list[str]:
    values = payload.get("value")
    if not isinstance(values, list):
        raise ReleaseIndexError("Search index inventory must contain a value list")
    names: list[str] = []
    for item in values:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ReleaseIndexError("Search index inventory contains an invalid index entry")
        names.append(item["name"])
    return names


def validate_inventory(payload: dict[str, Any], release_id: str) -> dict[str, Any]:
    index_name, knowledge_base_name = expected_names(release_id)
    names = _names(payload)
    release_prefix = f"legal-court-rag-v4-staging-{release_id}"
    same_release = sorted(name for name in names if name.startswith(release_prefix))
    report = {
        "schema_version": 1,
        "status": "PASS",
        "release_id": release_id,
        "expected_index": index_name,
        "expected_knowledge_base": knowledge_base_name,
        "observed_release_names": same_release,
        "expected_index_count": names.count(index_name),
        "expected_knowledge_base_count": names.count(knowledge_base_name),
        "read_only": True,
    }
    if same_release:
        report["status"] = "FAIL"
        raise ReleaseIndexError(f"Release id has already been used by Search indexes: {same_release}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True, help="Read-only Azure AI Search /indexes response")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(args.inventory.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ReleaseIndexError("Search index inventory must be a JSON object")
        report = validate_inventory(payload, args.release_id)
    except (OSError, json.JSONDecodeError, ReleaseIndexError) as error:
        report = {"schema_version": 1, "status": "FAIL", "release_id": args.release_id, "error": str(error), "read_only": True}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())