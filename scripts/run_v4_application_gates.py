"""Run fail-closed application gates against a v4 candidate deployment."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from application_gate import ApplicationGateError, validate_candidate_url, validate_provenance

REQUIRED_GATES = ("retrieval", "category", "source_hierarchy", "citation", "acl", "highlight")


class ApplicationGatesError(ValueError):
    """Raised when the complete application gate set is not proven."""


async def fetch_provenance(candidate_url: str, token: str = "") -> dict[str, Any]:
    headers = {"X-V4-Provenance-Token": token} if token else {}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{candidate_url}/api/provenance", headers=headers)
    except httpx.HTTPError as error:
        raise ApplicationGatesError(f"Candidate provenance request failed: {error}") from error
    if response.status_code != 200:
        raise ApplicationGatesError(f"Candidate provenance returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as error:
        raise ApplicationGatesError("Candidate provenance response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ApplicationGatesError("Candidate provenance response must be a JSON object")
    return payload


def load_gate_reports(paths: list[str], expected_provenance: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for item in paths:
        try:
            name, path_text = item.split("=", 1)
        except ValueError as error:
            raise ApplicationGatesError(f"Gate report must use name=path syntax: {item}") from error
        if name not in REQUIRED_GATES:
            raise ApplicationGatesError(f"Unknown application gate: {name}")
        if name in reports:
            raise ApplicationGatesError(f"Duplicate application gate: {name}")
        path = Path(path_text)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ApplicationGatesError(f"Cannot load {name} gate report: {path}") from error
        if not isinstance(payload, dict) or payload.get("status") != "PASS":
            raise ApplicationGatesError(f"Application gate {name} is missing status PASS")
        if payload.get("gate") not in (None, name):
            raise ApplicationGatesError(f"Application gate {name} is missing matching gate identity")
        if expected_provenance is not None:
            provenance = payload.get("provenance")
            if not isinstance(provenance, dict):
                raise ApplicationGatesError(f"Application gate {name} is missing provenance")
            mismatched = [
                field
                for field, value in expected_provenance.items()
                if str(provenance.get(field) or "").strip() != str(value).strip()
            ]
            if mismatched:
                raise ApplicationGatesError(
                    f"Application gate {name} provenance mismatch: {', '.join(mismatched)}"
                )
        if name == "highlight":
            required = ("gate", "oracle_version", "case_count", "source_count", "snapshot_manifest_sha256")
            if payload.get("gate") != "highlight" or any(not str(payload.get(field) or "").strip() for field in required[1:]):
                raise ApplicationGatesError("Application gate highlight is missing oracle evidence")
            if int(payload.get("case_count", 0)) <= 0 or int(payload.get("source_count", 0)) <= 0:
                raise ApplicationGatesError("Application gate highlight has no oracle cases or sources")
            browser_evidence = payload.get("browser_evidence")
            if not isinstance(browser_evidence, dict) or browser_evidence.get("highlight_visible") is not True:
                raise ApplicationGatesError("Application gate highlight is missing live browser evidence")
            payload = {"gate": name, **payload}
        reports[name] = payload

    missing = [name for name in REQUIRED_GATES if name not in reports]
    if missing:
        raise ApplicationGatesError(f"Application gate reports are missing: {', '.join(missing)}")
    return reports


def expected_provenance(args: argparse.Namespace) -> dict[str, str]:
    return {
        "release_id": args.release_id,
        "git_sha": args.git_sha,
        "deployment_id": args.deployment_id,
        "artifact_sha256": args.artifact_sha256,
        "search_snapshot_sha256": args.search_snapshot_sha256,
        "search_service": args.search_service,
        "search_index": args.search_index,
        "knowledge_base": args.knowledge_base,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_url = validate_candidate_url(args.candidate_url)
    provenance = await fetch_provenance(candidate_url, args.provenance_token)
    validated_provenance = validate_provenance(provenance, expected_provenance(args))
    reports = load_gate_reports(args.gate_report, expected_provenance=validated_provenance)
    return {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_url": candidate_url,
        "provenance": validated_provenance,
        "gates": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--search-snapshot-sha256", required=True)
    parser.add_argument("--search-service", required=True)
    parser.add_argument("--search-index", required=True)
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--provenance-token", default="")
    parser.add_argument("--gate-report", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = asyncio.run(run(args))
    except (ApplicationGateError, ApplicationGatesError) as error:
        print(json.dumps({"schema_version": 1, "status": "FAIL", "error": str(error)}, sort_keys=True))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent, prefix=f".{args.output.name}.", delete=False
    ) as temporary:
        temporary.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(args.output)
    print(json.dumps({"output": str(args.output), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())