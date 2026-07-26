import json
import threading

import pytest

from application_gate import ApplicationGateError, validate_candidate_url

from scripts.preflight_v4_local import run_api_gates, run_live_smoke
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