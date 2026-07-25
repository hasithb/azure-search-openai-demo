import pytest

from scripts.validate_v4_runtime_identity import RuntimeIdentityError, validate_runtime_identity


ENVIRONMENT = {
    "AZURE_SEARCH_INDEX": "legal-court-rag-v4-staging-release-1",
    "AZURE_SEARCH_KNOWLEDGEBASE_NAME": "legal-court-rag-v4-staging-release-1-agent-upgrade",
}


def runtime_fixture(image="registry.azurecr.io/v4-candidate:git-1", traffic=100, ready="v4-release-1", env=None):
    app = {
        "name": "candidate-app",
        "properties": {
            "provisioningState": "Succeeded",
            "latestRevisionName": "v4-release-1",
            "latestReadyRevisionName": ready,
        },
    }
    revision = {
        "name": "v4-release-1",
        "properties": {
            "trafficWeight": traffic,
            "runningState": "Running",
            "healthState": "Healthy",
            "template": {
                "containers": [{
                    "image": image,
                    "env": [{"name": key, "value": value} for key, value in (env or ENVIRONMENT).items()],
                }],
            },
        },
    }
    return app, [revision]


def test_runtime_identity_requires_exact_image_revision_traffic_and_environment():
    app, revisions = runtime_fixture()
    result = validate_runtime_identity(
        app,
        revisions,
        expected_revision="v4-release-1",
        expected_image="registry.azurecr.io/v4-candidate:git-1",
        expected_environment=ENVIRONMENT,
    )
    assert result["active_revision"] == "v4-release-1"
    assert result["traffic_weight"] == 100


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"image": "registry.azurecr.io/v4-candidate:old"}, "image"),
        ({"traffic": 0}, "traffic_weight"),
        ({"ready": "v4-old"}, "latest_ready_revision"),
        ({"env": {"AZURE_SEARCH_INDEX": "legal-court-rag-index-v3", **{key: value for key, value in ENVIRONMENT.items() if key != "AZURE_SEARCH_INDEX"}}}, "AZURE_SEARCH_INDEX"),
    ],
)
def test_runtime_identity_rejects_stale_or_mismatched_runtime(kwargs, message):
    app, revisions = runtime_fixture(**kwargs)
    with pytest.raises(RuntimeIdentityError, match=message):
        validate_runtime_identity(
            app,
            revisions,
            expected_revision="v4-release-1",
            expected_image="registry.azurecr.io/v4-candidate:git-1",
            expected_environment=ENVIRONMENT,
        )


def test_runtime_identity_rejects_duplicate_revision_names():
    app, revisions = runtime_fixture()
    with pytest.raises(RuntimeIdentityError, match="exactly one"):
        validate_runtime_identity(
            app,
            revisions + revisions,
            expected_revision="v4-release-1",
            expected_image="registry.azurecr.io/v4-candidate:git-1",
            expected_environment=ENVIRONMENT,
        )


def test_runtime_identity_normalizes_azure_qualified_revision_names():
    app, revisions = runtime_fixture()
    app["properties"]["latestReadyRevisionName"] = "candidate-app--v4-release-1"
    revisions[0]["name"] = "candidate-app--v4-release-1"
    result = validate_runtime_identity(
        app,
        revisions,
        expected_revision="v4-release-1",
        expected_image="registry.azurecr.io/v4-candidate:git-1",
        expected_environment=ENVIRONMENT,
    )
    assert result["active_revision"] == "v4-release-1"