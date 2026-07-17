"""Generate and validate local embeddings for the v4 JSONL artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import tiktoken
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

EMBEDDING_DIMENSIONS = 3072
MAX_TOKENS = 8191
ENCODING = tiktoken.encoding_for_model("text-embedding-3-large")


def prepare_text(content: str | list[str], *, strict: bool = True) -> str:
    text = " ".join(content) if isinstance(content, list) else str(content or "")
    tokens = ENCODING.encode(text.replace("\n", " "))
    if strict and len(tokens) > MAX_TOKENS:
        raise ValueError(f"Embedding input exceeds {MAX_TOKENS} tokens")
    return ENCODING.decode(tokens[:MAX_TOKENS])


def content_hash(document: dict) -> str:
    content = document.get("embedding_text", document.get("content", ""))
    if isinstance(content, list):
        content = "\n".join(content)
    value = "|".join(
        str(document.get(field, "") or "")
        for field in ("id", "sourcefile", "sourcepage", "category", "storageUrl", "updated")
    ) + f"|{content}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_reusable_embeddings(output_path: Path, state_path: Path, documents: list[dict]) -> dict[str, list[float]]:
    if not output_path.exists() or not state_path.exists():
        return {}
    embedded = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        return {}
    current_hashes = {str(document["id"]): content_hash(document) for document in documents}
    reusable: dict[str, list[float]] = {}
    for document in embedded:
        document_id = str(document.get("id") or "")
        vector = document.get("embedding")
        if (
            document_id
            and state.get(document_id) == current_hashes.get(document_id)
            and isinstance(vector, list)
            and len(vector) == EMBEDDING_DIMENSIONS
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector)
        ):
            reusable[document_id] = vector
    return reusable


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate local v4 embeddings")
    parser.add_argument("--artifact-dir", type=Path, default=Path("reports/index_v4_artifacts"))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-retries", type=int, default=6)
    args = parser.parse_args()
    input_path = args.artifact_dir / "documents.jsonl"
    output_path = args.artifact_dir / "documents_with_embeddings.jsonl"
    state_path = args.artifact_dir / "embedding_state.json"
    documents = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    deployment = os.environ.get("AZURE_OPENAI_EMB_DEPLOYMENT", "text-embedding-3-large")
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = AzureOpenAI(azure_endpoint=endpoint, azure_ad_token_provider=token_provider, api_version="2024-10-21")

    reusable = load_reusable_embeddings(output_path, state_path, documents)
    embedded: list[dict] = []
    pending: list[dict] = []
    for document in documents:
        vector = reusable.get(str(document.get("id") or ""))
        if vector is None:
            pending.append(document)
        else:
            embedded.append({**document, "embedding": vector})
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        for attempt in range(args.max_retries + 1):
            try:
                response = client.embeddings.create(
                    input=[prepare_text(doc.get("embedding_text", doc.get("content", ""))) for doc in batch],
                    model=deployment,
                )
                break
            except Exception as exc:
                if getattr(exc, "status_code", None) != 429 or attempt >= args.max_retries:
                    raise
                retry_after = getattr(exc, "response", None)
                retry_after = retry_after.headers.get("retry-after") if retry_after is not None else None
                delay = max(1, int(retry_after)) if retry_after and str(retry_after).isdigit() else min(60, 2 ** attempt)
                print(f"throttled; retrying in {delay}s ({attempt + 1}/{args.max_retries})", flush=True)
                time.sleep(delay)
        by_index = {item.index: item.embedding for item in response.data}
        if len(by_index) != len(batch):
            raise ValueError(f"Embedding count mismatch: {len(by_index)} != {len(batch)}")
        for index, document in enumerate(batch):
            vector = by_index[index]
            if len(vector) != EMBEDDING_DIMENSIONS or not all(math.isfinite(value) for value in vector):
                raise ValueError(f"Invalid embedding for {document['id']}: dimensions={len(vector)}")
            document["embedding"] = vector
            embedded.append(document)
        output_path.write_text(
            "".join(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n" for document in embedded),
            encoding="utf-8",
        )
        state_path.write_text(
            json.dumps({str(document["id"]): content_hash(document) for document in embedded}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"embedded {len(embedded)}/{len(documents)}")

    output_path.write_text("".join(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n" for document in embedded), encoding="utf-8")
    state_path.write_text(
        json.dumps({str(document["id"]): content_hash(document) for document in embedded}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = args.artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "embedding_status": "validated",
        "embedding_document_count": len(embedded),
        "embedding_vector_dimensions": EMBEDDING_DIMENSIONS,
        "embedding_input_token_limit": MAX_TOKENS,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("document_count", "embedding_document_count", "embedding_vector_dimensions", "embedding_status")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())