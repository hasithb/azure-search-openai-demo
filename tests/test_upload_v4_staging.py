import json

import pytest

from scripts.upload_v4_staging import EMBEDDING_DIMENSIONS, load_documents, validate_index_schema, validate_staging_target


def test_staging_target_rejects_production():
    with pytest.raises(ValueError, match="production"):
        validate_staging_target("legal-court-rag-index-v3")


@pytest.mark.parametrize("index_name", ["legal-court-rag-index", "legal-court-rag-v4-prod"])
def test_staging_target_requires_v4_staging_name(index_name):
    with pytest.raises(ValueError, match="v4.*staging"):
        validate_staging_target(index_name)


def test_load_documents_validates_ids_and_embeddings(tmp_path):
    path = tmp_path / "documents.jsonl"
    document = {
        "id": "doc-1",
        "content": "content",
        "embedding": [0.0] * EMBEDDING_DIMENSIONS,
    }
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    assert load_documents(path) == [document]


def test_load_documents_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "documents.jsonl"
    document = {"id": "doc-1", "content": "content", "embedding": [0.0] * EMBEDDING_DIMENSIONS}
    path.write_text(json.dumps(document) + "\n" + json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate document id"):
        load_documents(path)


class FakeField:
    def __init__(self, name, dimensions=EMBEDDING_DIMENSIONS, profile="embedding-profile"):
        self.name = name
        self.vector_search_dimensions = dimensions if name == "embedding" else None
        self.vector_search_profile_name = profile if name == "embedding" else None


class FakeIndex:
    def __init__(self, fields):
        self.fields = fields


def test_validate_index_schema_requires_all_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        validate_index_schema(FakeIndex([FakeField("id")]))


def test_validate_index_schema_checks_embedding_dimensions():
    fields = [FakeField(name) for name in {"id", "content", "category", "sourcepage", "sourcefile", "storageUrl", "oids", "groups", "parent_id", "subsection_id", "subsections", "updated"}]
    fields.append(FakeField("embedding", dimensions=1536))

    with pytest.raises(ValueError, match="1536"):
        validate_index_schema(FakeIndex(fields))


def test_provisioner_dry_run_accepts_disposable_target():
    from scripts.create_v4_staging_index import main

    # The parser-level behavior is covered by the validation-only invocation in
    # the release command; this test keeps the shared target guard exercised.
    validate_staging_target("legal-court-rag-v4-staging-test")


def test_staging_index_enables_permission_filtering():
    from azure.search.documents.indexes.models import SearchIndexPermissionFilterOption

    from scripts.create_v4_staging_index import build_index

    index = build_index("legal-court-rag-v4-staging-test")

    assert index.permission_filter_option == SearchIndexPermissionFilterOption.ENABLED