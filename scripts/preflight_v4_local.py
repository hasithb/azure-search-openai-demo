"""Run deterministic local v4 application gates without Azure mutation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from application_gate import validate_candidate_url
from gate_acl import run as run_acl
from gate_category import run as run_category
from gate_citation import run as run_citation
from gate_common import GateFailure, post_chat
from gate_highlight_browser import build_report as build_highlight_report
from gate_retrieval import run as run_retrieval
from gate_source_hierarchy import run as run_source_hierarchy
from run_v4_application_gates import (
    ApplicationGatesError,
    fetch_provenance,
    load_gate_reports,
)
from v4_local_test_server import PROVENANCE, start_fixture_server
from validate_v4_runtime_identity import validate_runtime_identity
from wait_v4_candidate_readiness import wait_for_readiness

GateRunner = Callable[[str, dict[str, str]], Awaitable[dict[str, Any]]]
LOCAL_IMAGE = "local.azurecr.io/v4-candidate@sha256:" + "d" * 64
LOCAL_REVISION = "g-local-r1"
LOCAL_ENVIRONMENT = {
    "AZURE_SEARCH_INDEX": PROVENANCE["search_index"],
    "AZURE_SEARCH_KNOWLEDGEBASE_NAME": PROVENANCE["knowledge_base"],
}


class _TimeoutClient:
    async def post(self, *args: Any, **kwargs: Any) -> None:
        raise httpx.ReadTimeout("fixture timeout")


class _InvalidJsonResponse:
    def raise_for_status(self) -> None:
        return

    def json(self) -> dict[str, Any]:
        raise ValueError("fixture response is not JSON")


class _InvalidJsonClient:
    async def post(self, *args: Any, **kwargs: Any) -> _InvalidJsonResponse:
        return _InvalidJsonResponse()


async def validate_chat_failure_contracts() -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for client, expected in (
        (_TimeoutClient(), "Candidate chat request timed out"),
        (_InvalidJsonClient(), "Candidate chat response is not valid JSON"),
    ):
        try:
            await post_chat(client, "https://fixture.invalid", "fixture failure contract")
        except GateFailure as error:
            if str(error) != expected:
                raise ApplicationGatesError(
                    f"Unexpected chat failure contract: expected={expected!r}, observed={str(error)!r}"
                ) from error
            checks.append({"id": expected, "status": "PASS"})
        else:
            raise ApplicationGatesError(f"Chat failure contract did not fail closed: {expected}")
    return checks


def fixture_highlight_report() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "PASS",
        "gate": "highlight",
        "oracle_version": "local-fixture-v1",
        "case_count": 1,
        "source_count": 1,
        "snapshot_manifest_sha256": "c" * 64,
        "browser_evidence": {"highlight_visible": True, "mode": "offline-fixture", "real_browser": False},
        "checks": [{"id": "fixture_highlight_contract", "status": "PASS"}],
        "provenance": PROVENANCE,
    }


def local_runtime_contract() -> dict[str, Any]:
    app = {
        "name": "local-v4-candidate",
        "properties": {
            "provisioningState": "Succeeded",
            "latestRevisionName": LOCAL_REVISION,
            "latestReadyRevisionName": LOCAL_REVISION,
        },
    }
    revisions = [{
        "name": LOCAL_REVISION,
        "properties": {
            "trafficWeight": 100,
            "runningState": "Running",
            "healthState": "Healthy",
            "template": {
                "containers": [{
                    "image": LOCAL_IMAGE,
                    "env": [{"name": key, "value": value} for key, value in LOCAL_ENVIRONMENT.items()],
                }],
            },
        },
    }]
    readiness = wait_for_readiness(
        lambda: app,
        lambda: revisions,
        expected_revision=LOCAL_REVISION,
        attempts=1,
        sleep_seconds=0,
    )
    identity = validate_runtime_identity(
        app,
        revisions,
        expected_revision=LOCAL_REVISION,
        expected_image=LOCAL_IMAGE,
        expected_environment=LOCAL_ENVIRONMENT,
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "mode": "offline-fixture",
        "readiness": readiness,
        "runtime_identity": identity,
        "checks": [
            {"id": "candidate_revision_ready", "status": "PASS"},
            {"id": "candidate_image_immutable", "status": "PASS"},
            {"id": "candidate_environment_bound", "status": "PASS"},
        ],
    }


async def run_api_gates(candidate: str) -> dict[str, dict[str, Any]]:
    runners: tuple[tuple[str, GateRunner], ...] = (
        ("retrieval", run_retrieval),
        ("category", run_category),
        ("source_hierarchy", run_source_hierarchy),
        ("citation", run_citation),
        ("acl", run_acl),
    )
    reports: dict[str, dict[str, Any]] = {}
    for name, runner in runners:
        reports[name] = await runner(candidate, PROVENANCE)
    reports["highlight"] = fixture_highlight_report()
    return reports


async def run_live_smoke(
    candidate: str,
    provenance_path: Path,
    oracle_path: Path,
    snapshot_dir: Path,
    output: Path,
    question: str,
    provenance_token: str = "",
) -> dict[str, Any]:
    candidate_url = validate_candidate_url(candidate)
    expected_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(expected_provenance, dict):
        raise ValueError("Live-smoke provenance file must contain a JSON object")
    observed_provenance = await fetch_provenance(candidate_url, provenance_token)
    if observed_provenance != expected_provenance:
        raise ApplicationGatesError("Live candidate provenance does not match the supplied provenance file")

    reports: dict[str, dict[str, Any]] = {}
    for name, runner in (
        ("retrieval", run_retrieval),
        ("category", run_category),
        ("source_hierarchy", run_source_hierarchy),
        ("citation", run_citation),
        ("acl", run_acl),
    ):
        reports[name] = await runner(candidate_url, expected_provenance)
    reports["highlight"] = build_highlight_report(
        candidate_url, oracle_path, snapshot_dir, expected_provenance, question
    )
    write_reports(output, reports)
    paths = [f"{name}={output / f'{name}.json'}" for name in reports]
    loaded = load_gate_reports(paths, expected_provenance=expected_provenance)
    result = {
        "mode": "live-smoke",
        "promotion_eligible": False,
        "status": "PASS",
        "candidate_url": candidate_url,
        "gates": sorted(loaded),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def write_reports(output: Path, reports: dict[str, dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        (output / f"{name}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "live-smoke"), default="offline")
    parser.add_argument(
        "--require-runtime-contract",
        action="store_true",
        help="Require the deterministic readiness and runtime-identity contract in offline mode",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/v4-local"))
    parser.add_argument("--candidate-url", help="Explicit HTTPS deployed candidate URL for live-smoke")
    parser.add_argument("--provenance", type=Path, help="Expected candidate provenance JSON for live-smoke")
    parser.add_argument("--oracle", type=Path, help="Highlight oracle JSON for live-smoke")
    parser.add_argument("--snapshot-dir", type=Path, help="Canonical snapshot directory for live-smoke")
    parser.add_argument("--provenance-token", default="")
    parser.add_argument("--question", default="What is CPR Part 24 rule 24.2 and the test for summary judgment?")
    args = parser.parse_args()
    if args.mode == "live-smoke":
        missing = [
            name
            for name, value in (
                ("--candidate-url", args.candidate_url),
                ("--provenance", args.provenance),
                ("--oracle", args.oracle),
                ("--snapshot-dir", args.snapshot_dir),
            )
            if value is None
        ]
        if missing:
            parser.error(f"live-smoke requires: {', '.join(missing)}")
        try:
            result = asyncio.run(
                run_live_smoke(
                    args.candidate_url,
                    args.provenance,
                    args.oracle,
                    args.snapshot_dir,
                    args.output,
                    args.question,
                    args.provenance_token,
                )
            )
        except (OSError, json.JSONDecodeError, ApplicationGatesError, ValueError) as error:
            print(json.dumps({"mode": "live-smoke", "promotion_eligible": False, "status": "FAIL", "error": str(error)}))
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0

    os.environ["V4_LOCAL_FIXTURE"] = "1"
    server, candidate = start_fixture_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        reports = asyncio.run(run_api_gates(candidate))
        chat_failure_contracts = asyncio.run(validate_chat_failure_contracts())
        runtime_contract = local_runtime_contract()
        if args.require_runtime_contract and runtime_contract.get("status") != "PASS":
            raise ApplicationGatesError("Local runtime contract did not pass")
        write_reports(args.output, reports)
        (args.output / "runtime_contract.json").write_text(
            json.dumps(runtime_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        paths = [f"{name}={args.output / f'{name}.json'}" for name in reports]
        loaded = load_gate_reports(paths, expected_provenance=PROVENANCE)
        result = {
            "mode": "offline",
            "promotion_eligible": False,
            "status": "PASS",
            "gates": sorted(loaded),
            "runtime_contract": runtime_contract["status"],
            "chat_failure_contracts": chat_failure_contracts,
        }
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())