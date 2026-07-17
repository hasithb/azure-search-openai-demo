"""Generate Azure OpenAI embeddings for a validated v4 JSONL artifact."""

from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider  # noqa: E402
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


def embedding_input(content: str | list[str], *, strict: bool = False) -> tuple[str, int, bool]:
    text = "\n".join(content) if isinstance(content, list) else str(content or "")
    tokens = ENCODING.encode(text)
    truncated = len(tokens) > MAX_TOKENS
    if strict and truncated:
        raise ValueError(f"Embedding input exceeds {MAX_TOKENS} tokens")
    bounded_tokens = tokens[:MAX_TOKENS]
    return ENCODING.decode(bounded_tokens), len(bounded_tokens), truncated


async def generate(
    input_path: Path, output_path: Path, endpoint: str, deployment: str, batch_size: int = 100
) -> int:
    documents = load_documents(input_path)
    checkpoint_path = output_path.with_suffix(output_path.suffix + ".checkpoint")
    completed: dict[str, dict] = {}
    if checkpoint_path.exists():
        completed = {str(document.get("id")): document for document in load_documents(checkpoint_path)}
        documents = [document for document in documents if str(document.get("id")) not in completed]
    credential = DefaultAzureCredential()
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
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            total_documents = len(completed) + len(documents)
            written = len(completed)
            for document in completed.values():
                temporary.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
            for start in range(0, len(documents), batch_size):
                batch_documents = documents[start : start + batch_size]
                prepared = [
                    embedding_input(document.get("embedding_text", document.get("content", "")), strict=True)
                    for document in batch_documents
                ]
                vectors = await embeddings.create_embeddings([item[0] for item in prepared])
                if len(vectors) != len(batch_documents) or any(len(vector) != DIMENSIONS for vector in vectors):
                    raise ValueError(f"Embedding response count or dimensions do not match batch starting at {start}")
                for document, vector, prepared_item in zip(batch_documents, vectors, prepared):
                    document["embedding"] = vector
                    input_text, token_count, truncated = prepared_item
                    document["embedding_input_sha256"] = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
                    document["embedding_input_token_count"] = token_count
                    document["embedding_input_truncated"] = truncated
                    temporary.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
                    written += 1
                temporary.flush()
                checkpoint_path.write_text(temporary_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    if written != total_documents:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"Expected {total_documents} embedded documents, wrote {written}")
    temporary_path.replace(output_path)
    checkpoint_path.unlink(missing_ok=True)
    await client.close()
    await credential.close()
    return len(documents)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_OPENAI_ENDPOINT", ""))
    parser.add_argument("--deployment", default=os.environ.get("AZURE_OPENAI_EMB_DEPLOYMENT", ""))
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if not args.endpoint or not args.deployment:
        raise ValueError("--endpoint/--deployment or AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_EMB_DEPLOYMENT is required")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    count = asyncio.run(generate(args.input, args.output, args.endpoint, args.deployment, args.batch_size))
    print(json.dumps({"documents": count, "dimensions": DIMENSIONS, "max_tokens": MAX_TOKENS, "model": MODEL, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
