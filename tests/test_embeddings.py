import asyncio

import pytest

from prepdocslib.embeddings import OpenAIEmbeddings

from .test_prepdocs import MockClient, MockEmbeddingsClient


@pytest.fixture
def embeddings_client():
    return OpenAIEmbeddings(
        open_ai_client=MockClient(MockEmbeddingsClient(None)),
        open_ai_model_name="text-embedding-ada-002",
        open_ai_dimensions=1536,
        disable_batch=False,
    )


def test_split_text_into_batches_single_batch(embeddings_client):
    texts = ["hello world", "foo bar"]
    batches = embeddings_client.split_text_into_batches(texts)
    assert len(batches) == 1
    assert batches[0].texts == texts


def test_split_text_into_batches_multiple_batches(embeddings_client):
    # Use single-token words to get predictable token counts
    # Each text needs > 4050 tokens so together they exceed the 8100 limit
    texts = ["a " * 5000, "b " * 5000]
    batches = embeddings_client.split_text_into_batches(texts)
    assert len(batches) == 2
    assert batches[0].texts == [texts[0]]
    assert batches[1].texts == [texts[1]]


def test_split_text_into_batches_respects_max_batch_size(embeddings_client):
    # max_batch_size is 16, so 17 small texts should produce 2 batches
    texts = ["hi"] * 17
    batches = embeddings_client.split_text_into_batches(texts)
    assert len(batches) == 2
    assert len(batches[0].texts) == 16
    assert len(batches[1].texts) == 1


def test_split_text_into_batches_splits_oversized_text(embeddings_client):
    """Text with more tokens than batch_token_limit should be split into smaller chunks."""
    # Create a text that exceeds the 8100 token limit
    oversized_text = "word " * 9000  # ~9000 tokens
    texts = [oversized_text]
    batches = embeddings_client.split_text_into_batches(texts)
    # Should be split into 2 chunks (8100 + remainder)
    all_texts = [t for b in batches for t in b.texts]
    assert len(all_texts) == 2
    # No batch should exceed the token limit
    for batch in batches:
        assert batch.token_length <= 8100
    # All original content should be preserved across chunks
    import tiktoken

    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    original_tokens = encoding.encode(oversized_text)
    reconstructed_tokens = []
    for text in all_texts:
        reconstructed_tokens.extend(encoding.encode(text))
    assert len(reconstructed_tokens) == len(original_tokens)


def test_split_text_into_batches_splits_oversized_text_with_others(embeddings_client):
    """Oversized text mixed with normal texts should all be batched correctly."""
    oversized_text = "word " * 9000
    normal_text = "hello world"
    texts = [oversized_text, normal_text]
    batches = embeddings_client.split_text_into_batches(texts)
    # Oversized text splits into 2 chunks + 1 normal text = 3 texts total
    all_texts = [t for b in batches for t in b.texts]
    assert len(all_texts) == 3
    # No batch should exceed the token limit
    for batch in batches:
        assert batch.token_length <= 8100


def test_split_text_into_batches_empty(embeddings_client):
    batches = embeddings_client.split_text_into_batches([])
    assert len(batches) == 0


def test_split_text_unsupported_model():
    embeddings = OpenAIEmbeddings(
        open_ai_client=MockClient(MockEmbeddingsClient(None)),
        open_ai_model_name="unsupported-model",
        open_ai_dimensions=1536,
    )
    with pytest.raises(NotImplementedError):
        embeddings.split_text_into_batches(["test"])


@pytest.mark.asyncio
async def test_create_embeddings_concurrent_limits_concurrency_and_preserves_order(embeddings_client, monkeypatch):
    active = 0
    peak = 0

    async def create_batch(batch, dimensions_args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        batch_start = int(batch.texts[0].split("-")[1])
        await asyncio.sleep(0.001 if batch_start == 0 else 0)
        active -= 1
        return [[float(index)] for index in range(batch_start, batch_start + len(batch.texts))]

    monkeypatch.setattr(embeddings_client, "_create_embedding_batch_with_retry", create_batch)
    prepared = [(f"item-{index}", 600) for index in range(32)]

    result = await embeddings_client.create_embeddings_concurrent(prepared, concurrency=2)

    assert peak == 2
    assert result == [[float(index)] for index in range(32)]


def test_split_prepared_into_batches_respects_token_and_input_limits(embeddings_client):
    prepared = [(f"text-{index}", 600) for index in range(17)]

    batches = embeddings_client.split_prepared_into_batches(prepared)

    assert [len(batch.texts) for batch in batches] == [13, 4]
    assert all(batch.token_length <= 8100 for batch in batches)


def test_split_prepared_into_batches_preserves_embedding_input_metadata_shape(embeddings_client):
    prepared = [("text", 600, False)]

    batches = embeddings_client.split_prepared_into_batches(prepared)

    assert batches[0].texts == ["text"]
    assert batches[0].token_length == 600


def test_split_prepared_into_batches_rejects_oversized_input(embeddings_client):
    with pytest.raises(ValueError, match="8100-token"):
        embeddings_client.split_prepared_into_batches([("oversized", 8101)])
