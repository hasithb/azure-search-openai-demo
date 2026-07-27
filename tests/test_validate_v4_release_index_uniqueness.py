import pytest

from scripts.validate_v4_release_index_uniqueness import ReleaseIndexError, validate_inventory


def inventory(*names):
    return {"value": [{"name": name} for name in names]}


def test_release_index_inventory_requires_exact_pair_once():
    report = validate_inventory(inventory("legal-court-rag-v4-staging-20260726-r2"), "20260726-r1")

    assert report["status"] == "PASS"
    assert report["read_only"] is True


@pytest.mark.parametrize(
    "names",
    [
        ("legal-court-rag-v4-staging-20260726-r1",),
        ("legal-court-rag-v4-staging-20260726-r1-agent-upgrade",),
        ("legal-court-rag-v4-staging-20260726-r1-debug",),
    ],
)
def test_release_index_inventory_rejects_missing_duplicate_or_unexpected(names):
    with pytest.raises(ReleaseIndexError):
        validate_inventory(inventory(*names), "20260726-r1")