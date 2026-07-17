import json

import pytest

from scripts.embed_v4_artifacts import EMBEDDING_DIMENSIONS, content_hash, load_reusable_embeddings, prepare_text
from scripts.generate_v4_embeddings import MAX_TOKENS, embedding_input


def test_embedding_reuse_requires_matching_document_content_hash(tmp_path):
    document = {
        "id": "one",
        "content": "original",
        "sourcefile": "guide",
        "sourcepage": "1",
        "category": "court",
        "storageUrl": "",
        "updated": "",
    }
    output_path = tmp_path / "documents_with_embeddings.jsonl"
    state_path = tmp_path / "embedding_state.json"
    output_path.write_text(
        json.dumps({**document, "embedding": [0.0] * EMBEDDING_DIMENSIONS}) + "\n",
        encoding="utf-8",
    )
    state_path.write_text(json.dumps({"one": content_hash(document)}), encoding="utf-8")

    assert "one" in load_reusable_embeddings(output_path, state_path, [document])
    assert load_reusable_embeddings(output_path, state_path, [{**document, "content": "changed"}]) == {}


def test_embedding_input_is_bounded_to_model_token_limit():
    text, token_count, truncated = embedding_input("word " * (MAX_TOKENS + 100))

    assert token_count == MAX_TOKENS
    assert truncated is True
    assert len(text) > 0


def test_embedding_input_strict_mode_rejects_truncation():
    with pytest.raises(ValueError, match="exceeds"):
        embedding_input("word " * (MAX_TOKENS + 100), strict=True)


def test_embedding_hash_includes_embedding_text():
    document = {"id": "one", "content": "same", "embedding_text": "hierarchy one"}

    changed = {**document, "embedding_text": "hierarchy two"}

    assert content_hash(document) != content_hash(changed)


def test_prepare_text_rejects_oversized_embedding_input():
    with pytest.raises(ValueError, match="exceeds"):
        prepare_text("word " * (MAX_TOKENS + 100))