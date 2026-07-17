import json
from pathlib import Path

import pytest

from scripts.audit_source_documents import CanonicalSource
from scripts.generate_v4_artifacts import (
    GUIDE_FILES,
    ROOT,
    deduplicate_sources_by_url,
    enrich_retrieval_metadata,
    snapshot_hash,
    validate_source_snapshot,
    expand_oversized_embedding_windows,
)


def test_retrieval_metadata_preserves_content_and_builds_hierarchy():
    document = {
        "content": "Rule 31.16 permits an application for specific disclosure.",
        "sourcefile": "Part 31",
        "sourcepage": "Part 31 Disclosure",
        "category": "Civil Procedure Rules and Practice Directions",
        "subsection_id": "31.16",
    }

    enrich_retrieval_metadata(document)

    assert document["content"].startswith("Rule 31.16")
    assert document["section_title"] == "31.16"
    assert document["hierarchy_path"] == "Part 31 > Part 31 Disclosure > 31.16"
    assert "31.16" in document["legal_references"]
    assert "HIERARCHY: Part 31 > Part 31 Disclosure > 31.16" in document["embedding_text"]


def test_oversized_embedding_windows_preserve_canonical_content():
    document = {
        "id": "part-31",
        "content": "## 31.1 Heading\n" + ("Disclosure detail. " * 6000),
        "sourcefile": "Part 31",
        "sourcepage": "Part 31 Disclosure",
        "category": "Civil Procedure Rules and Practice Directions",
        "subsection_id": "31.1",
    }
    enrich_retrieval_metadata(document)

    children = expand_oversized_embedding_windows([document])

    assert len(children) > 1
    assert all(child["content"] == document["content"] for child in children)
    assert all(child["parent_id"] == "part-31" for child in children)
    assert [child["child_window"] for child in children] == list(range(1, len(children) + 1))


COURT_GUIDES_DIR = ROOT / "scripts" / "court_guides_processing_pipeline" / "outputs_azure_di"


def test_all_configured_court_guides_have_processed_artifacts():
    for guide in GUIDE_FILES.values():
        path = COURT_GUIDES_DIR / guide["file"]
        assert path.exists(), path
        documents = json.loads(path.read_text(encoding="utf-8"))
        assert documents
        assert all(document.get("sourcefile") == guide["sourcefile"] for document in documents)
        assert all(document.get("category") == guide["category"] for document in documents)


def test_ipec_processed_artifact_is_release_ready():
    guide = GUIDE_FILES["Intellectual Property Enterprise Court"]
    path = COURT_GUIDES_DIR / guide["file"]
    documents = json.loads(path.read_text(encoding="utf-8"))

    assert len(documents) == 73
    assert all(document.get("content") for document in documents)
    assert all(document.get("storageUrl") for document in documents)


def test_source_snapshot_hash_is_deterministic():
    first = {"identity": "cpr::part 1", "html": "<p>one</p>", "status": "ok"}
    second = {"status": "ok", "html": "<p>one</p>", "identity": "cpr::part 1"}

    assert snapshot_hash(first) == snapshot_hash(second)


def test_duplicate_url_sources_prefer_descriptive_identity():
    short = CanonicalSource(
        source_type="html",
        sourcefile="Part 83",
        category="CPR",
        url="https://example.test/part-83",
    )
    descriptive = CanonicalSource(
        source_type="html",
        sourcefile="Part 83 Writs and Warrants",
        category="CPR",
        url="https://example.test/part-83",
    )

    selected = deduplicate_sources_by_url([short, descriptive])

    assert list(selected) == [descriptive.identity]


def test_pdf_source_snapshot_requires_verified_provenance(tmp_path):
    source = CanonicalSource(
        source_type="pdf",
        sourcefile="Pre-Action Protocol for Debt Claims",
        category="Civil Procedure Rules and Practice Directions",
        url="https://example.test/debt-pap.pdf",
    )
    snapshot = {
        "status": "ok",
        "source_type": "pdf",
        "content_type": "application/pdf",
        "source_sha256": "abc123",
        "extracted_text": "Debt Claims content",
        "html": "<html><body>Debt Claims content</body></html>",
    }

    validate_source_snapshot(snapshot, source, tmp_path / "debt.json")


@pytest.mark.parametrize("missing", ["source_sha256", "extracted_text", "html"])
def test_pdf_source_snapshot_rejects_missing_provenance(tmp_path, missing):
    source = CanonicalSource(
        source_type="pdf",
        sourcefile="Pre-Action Protocol for Debt Claims",
        category="Civil Procedure Rules and Practice Directions",
    )
    snapshot = {
        "status": "ok",
        "source_type": "pdf",
        "content_type": "application/pdf",
        "source_sha256": "abc123",
        "extracted_text": "Debt Claims content",
        "html": "<html><body>Debt Claims content</body></html>",
    }
    snapshot.pop(missing)

    with pytest.raises(ValueError, match="PDF snapshot|Source snapshot"):
        validate_source_snapshot(snapshot, source, tmp_path / "debt.json")
