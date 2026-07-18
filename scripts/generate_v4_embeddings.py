"""Generate Azure OpenAI embeddings for a validated v4 JSONL artifact."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from azure.identity.aio import (  # noqa: E402
    DefaultAzureCredential,
    WorkloadIdentityCredential,
    get_bearer_token_provider,
)
from openai import AsyncAzureOpenAI  # noqa: E402

from prepdocslib.embeddings import OpenAIEmbeddings  # noqa: E402

MODEL = "text-embedding-3-large"
DIMENSIONS = 3072
MAX_TOKENS = 8191
ENCODING = tiktoken.encoding_for_model(MODEL)


def load_documents(path: Path) -> list[dict]:
    documents = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not documents:
        raise ValueError(f"Artifact is empty: {path}")
    return documents


def load_checkpoint(path: Path) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        document = json.loads(line)
        document_id = str(document.get("id") or "")
        if not document_id or document_id in completed:
            raise ValueError(f"Checkpoint contains duplicate or empty document id: {document_id!r}")
        completed[document_id] = document
    return completed


def embedding_input(content: str | list[str], *, strict: bool = False) -> tuple[str, int, bool]:
    text = "\n".join(content) if isinstance(content, list) else str(content or "")
    tokens = ENCODING.encode(text)
    truncated = len(tokens) > MAX_TOKENS
    if strict and truncated:
        raise ValueError(f"Embedding input exceeds {MAX_TOKENS} tokens")
    bounded_tokens = tokens[:MAX_TOKENS]
    return ENCODING.decode(bounded_tokens), len(bounded_tokens), truncated


def create_credential():
    if os.environ.get("AZURE_FEDERATED_TOKEN_FILE"):
        return WorkloadIdentityCredential()
    return DefaultAzureCredential()


async def generate(
    input_path: Path,
    output_path: Path,
    endpoint: str,
    deployment: str,
    batch_size: int = 100,
    max_batches: int | None = None,
    concurrency: int = 8,
) -> int:
    documents = load_documents(input_path)
    checkpoint_path = output_path.with_suffix(output_path.suffix + ".checkpoint")
    completed = load_checkpoint(checkpoint_path)
    input_ids: set[str] = set()
    for document in documents:
        document_id = str(document.get("id") or "")
        if not document_id or document_id in input_ids:
            raise ValueError(f"Input contains duplicate or empty document id: {document_id!r}")
        input_ids.add(document_id)
    unknown_checkpoint_ids = set(completed) - input_ids
    if unknown_checkpoint_ids:
        raise ValueError(f"Checkpoint contains ids missing from input: {sorted(unknown_checkpoint_ids)[:5]}")
    documents = [document for document in documents if str(document["id"]) not in completed]
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    credential = create_credential()
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
    client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        azure_ad_token_provider=token_provider,
    )
    embeddings = OpenAIEmbeddings(
        client,
        MODEL,
        DIMENSIONS,
        azure_deployment_name=deployment,
        azure_endpoint=endpoint,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            total_documents = len(completed) + len(documents)
            written = len(completed)
            for document in completed.values():
                temporary.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
            checkpoint = checkpoint_path.open("a", encoding="utf-8")
            processed_batches = 0
            try:
                for start in range(0, len(documents), batch_size):
                    if max_batches is not None and processed_batches >= max_batches:
                        break
                    batch_documents = documents[start : start + batch_size]
                    prepared = [
                        embedding_input(document.get("embedding_text", document.get("content", "")), strict=True)
                        for document in batch_documents
                    ]
                    vectors = await embeddings.create_embeddings_concurrent(prepared, concurrency)
                    if len(vectors) != len(batch_documents) or any(len(vector) != DIMENSIONS for vector in vectors):
                        raise ValueError(f"Embedding response count or dimensions do not match batch starting at {start}")
                    for document, vector, prepared_item in zip(batch_documents, vectors, prepared):
                        document["embedding"] = vector
                        input_text, token_count, truncated = prepared_item
                        document["embedding_input_sha256"] = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
                        document["embedding_input_token_count"] = token_count
                        document["embedding_input_truncated"] = truncated
                        serialized = json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n"
                        temporary.write(serialized)
                        checkpoint.write(serialized)
                        written += 1
                    temporary.flush()
                    checkpoint.flush()
                    os.fsync(checkpoint.fileno())
                    processed_batches += 1
            finally:
                checkpoint.close()
            expected_batches = (len(documents) + batch_size - 1) // batch_size
            if max_batches is not None and processed_batches < expected_batches:
                temporary_path.unlink(missing_ok=True)
                return len(documents)
            if written != total_documents:
                temporary_path.unlink(missing_ok=True)
                raise ValueError(f"Expected {total_documents} embedded documents, wrote {written}")
            temporary_path.replace(output_path)
            checkpoint_path.unlink(missing_ok=True)
            return len(documents)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    finally:
        await client.close()
        await credential.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_OPENAI_ENDPOINT", ""))
    parser.add_argument("--deployment", default=os.environ.get("AZURE_OPENAI_EMB_DEPLOYMENT", ""))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if not args.endpoint or not args.deployment:
        raise ValueError("--endpoint/--deployment or AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_EMB_DEPLOYMENT is required")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be at least 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    count = asyncio.run(
        generate(args.input, args.output, args.endpoint, args.deployment, args.batch_size, args.max_batches, args.concurrency)
    )
    print(json.dumps({"documents": count, "dimensions": DIMENSIONS, "max_tokens": MAX_TOKENS, "model": MODEL, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
