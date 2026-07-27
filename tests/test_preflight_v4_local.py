import json
import threading

import pytest

from application_gate import ApplicationGateError, validate_candidate_url

from scripts.preflight_v4_local import (
    local_runtime_contract,
    run_api_gates,
    run_live_smoke,
    wait_for_local_app,
    validate_chat_failure_contracts,
)
from scripts.run_v4_application_gates import load_gate_reports
from scripts.v4_local_test_server import PROVENANCE, start_fixture_server


@pytest.fixture
def fixture_origin():
    server, origin = start_fixture_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield origin
    finally:
        server.shutdown()
        server.server_close()


def test_live_candidate_policy_rejects_localhost() -> None:
    with pytest.raises(ApplicationGateError, match="must be an HTTPS URL"):
        validate_candidate_url("http://127.0.0.1:50505")

    assert validate_candidate_url("http://127.0.0.1:50505", allow_local=True) == "http://127.0.0.1:50505"


def test_local_runtime_contract_covers_candidate_readiness_identity_and_configuration() -> None:
    report = local_runtime_contract()

    assert report["status"] == "PASS"
    assert report["readiness"]["status"] == "READY"
    assert report["runtime_identity"]["traffic_weight"] == 100
    assert report["runtime_identity"]["running_state"] == "Running"
    assert report["runtime_identity"]["health_state"] == "Healthy"


@pytest.mark.asyncio
async def test_fixture_server_produces_all_gate_contracts(fixture_origin, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("V4_LOCAL_FIXTURE", "1")
    reports = await run_api_gates(fixture_origin)

    assert set(reports) == {"retrieval", "category", "source_hierarchy", "citation", "acl", "highlight"}
    assert reports["highlight"]["browser_evidence"]["real_browser"] is False
    assert all(report["status"] == "PASS" for report in reports.values())

    paths = []
    for name, report in reports.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(f"{name}={path}")
    loaded = load_gate_reports(paths, expected_provenance=PROVENANCE)
    assert tuple(loaded) == ("retrieval", "category", "source_hierarchy", "citation", "acl", "highlight")


@pytest.mark.asyncio
async def test_live_smoke_requires_real_https_candidate(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("V4_LOCAL_FIXTURE", raising=False)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(PROVENANCE), encoding="utf-8")

    with pytest.raises(ApplicationGateError, match="must be an HTTPS URL"):
        await run_live_smoke(
            "http://127.0.0.1:50505",
            provenance,
            tmp_path / "oracle.json",
            tmp_path / "snapshots",
            tmp_path / "reports",
            "test question",
        )


@pytest.mark.asyncio
async def test_local_preflight_captures_chat_failure_contracts() -> None:
    checks = await validate_chat_failure_contracts()

    assert [check["id"] for check in checks] == [
        "Candidate chat request timed out",
        "Candidate chat response is not valid JSON",
    ]


@pytest.mark.asyncio
async def test_local_readiness_attaches_to_fixture_server(fixture_origin) -> None:
    readiness = await wait_for_local_app(fixture_origin, "/api/provenance", 1)

    assert readiness["status"] == "READY"
    assert readiness["http_status"] == 200


@pytest.mark.asyncio
async def test_local_readiness_fails_with_bounded_timeout() -> None:
    with pytest.raises(Exception, match="did not become ready within 0.0s"):
        await wait_for_local_app("http://127.0.0.1:1", "/api/provenance", 0)