import pytest

from scripts.wait_v4_candidate_readiness import ReadinessError, wait_for_readiness


def app(latest_ready="candidate--g-new"):
    return {"properties": {"provisioningState": "Succeeded", "latestReadyRevisionName": latest_ready}}


def revision(running="Running", health="Healthy"):
    return {"name": "candidate--g-new", "properties": {"runningState": running, "healthState": health}}


def test_retries_eventual_consistency_then_returns_ready():
    app_responses = iter([app(latest_ready="candidate--old"), app()])
    revision_responses = iter([[revision(running="Activating", health=None)], [revision()]])
    sleeps = []

    result = wait_for_readiness(
        lambda: next(app_responses),
        lambda: next(revision_responses),
        expected_revision="g-new",
        attempts=2,
        sleep_seconds=3,
        sleep_fn=sleeps.append,
    )

    assert result["status"] == "READY"
    assert result["attempt"] == 2
    assert sleeps == [3]


def test_retries_read_failures_but_still_times_out():
    reads = {"count": 0}

    def read_app():
        reads["count"] += 1
        raise OSError("connection reset")

    with pytest.raises(ReadinessError, match="timed out"):
        wait_for_readiness(read_app, lambda: [], expected_revision="g-new", attempts=2, sleep_fn=lambda _: None)
    assert reads["count"] == 2


@pytest.mark.parametrize(
    "running, health, message",
    [("Failed", "Unhealthy", "terminal"), ("Stopped", "Healthy", "terminal")],
)
def test_terminal_revision_state_fails_immediately(running, health, message):
    reads = {"count": 0}

    def read_revisions():
        reads["count"] += 1
        return [revision(running=running, health=health)]

    with pytest.raises(ReadinessError, match=message):
        wait_for_readiness(lambda: app(), read_revisions, expected_revision="g-new", attempts=5)
    assert reads["count"] == 1