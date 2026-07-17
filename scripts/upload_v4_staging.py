"""Validate and optionally upload a v4 artifact to a disposable Search index.

The command is intentionally staging-only. It refuses the production index name,
validates the embedded JSONL before creating an Azure client, and never deletes
documents. Provisioning the index itself remains a separate, explicit operation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

EMBEDDING_DIMENSIONS = 3072
PRODUCTION_INDEX = "legal-court-rag-index-v3"
MAX_BATCH_SIZE = 1000
REQUIRED_FIELDS = {
    "id",
    "content",
    "embedding",
    "category",
    "sourcepage",
    "sourcefile",
    "storageUrl",
    "oids",
    "groups",
    "parent_id",
    "subsection_id",
    "subsections",
    "updated",
}
INDEX_FIELDS = REQUIRED_FIELDS | {
    "embedding_text",
    "section_title",
    "hierarchy_path",
    "legal_references",
    "child_window",
    "child_window_count",
}


def validate_staging_target(index_name: str) -> None:
    normalized = index_name.strip().casefold()
    if normalized == PRODUCTION_INDEX.casefold():
        raise ValueError(f"Refusing to target production index: {index_name}")
    if "staging" not in normalized or "v4" not in normalized:
        raise ValueError("Target index must contain both 'v4' and 'staging'")


def load_documents(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            document = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
        if not isinstance(document, dict):
            raise ValueError(f"Document on line {line_number} is not an object")
        document_id = str(document.get("id") or "")
        if not document_id:
            raise ValueError(f"Document on line {line_number} has no id")
        if document_id in seen_ids:
            raise ValueError(f"Duplicate document id: {document_id}")
        seen_ids.add(document_id)
        content = document.get("content")
        if not isinstance(content, (str, list)) or not content:
            raise ValueError(f"Document {document_id} has no content")
        vector = document.get("embedding")
        if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Document {document_id} has invalid embedding dimensions: "
                f"{len(vector) if isinstance(vector, list) else 'missing'}"
            )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
            raise ValueError(f"Document {document_id} has a non-finite embedding value")
        documents.append(document)
    if not documents:
        raise ValueError(f"Artifact is empty: {path}")
    return documents


def validate_index_schema(index: Any) -> None:
    fields = {field.name: field for field in index.fields}
    missing = sorted(REQUIRED_FIELDS - fields.keys())
    if missing:
        raise ValueError(f"Staging index is missing required fields: {', '.join(missing)}")
    embedding = fields["embedding"]
    dimensions = getattr(embedding, "vector_search_dimensions", None)
    if dimensions != EMBEDDING_DIMENSIONS:
        raise ValueError(f"Staging embedding dimensions are {dimensions}, expected {EMBEDDING_DIMENSIONS}")
    profile = getattr(embedding, "vector_search_profile_name", None)
    if not profile:
        raise ValueError("Staging embedding field has no vector search profile")


def project_document(document: dict[str, Any]) -> dict[str, Any]:
    return {field: document[field] for field in INDEX_FIELDS if field in document}


def upload_documents(index_name: str, service: str, documents: list[dict[str, Any]], batch_size: int) -> int:
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    endpoint = service if service.startswith("https://") else f"https://{service}.search.windows.net"
    credential = DefaultAzureCredential()
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    validate_index_schema(index_client.get_index(index_name))
    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
    uploaded = 0
    for start in range(0, len(documents), batch_size):
        batch = [project_document(document) for document in documents[start:start + batch_size]]
        results = client.upload_documents(documents=batch)
        failures = [result for result in results if not result.succeeded]
        if failures:
            details = ", ".join(f"{result.key}: {result.error_message}" for result in failures)
            raise RuntimeError(f"Upload failed: {details}")
        uploaded += len(results)
    return uploaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("reports/index_v4_artifacts/documents_with_embeddings.jsonl"))
    parser.add_argument("--index", required=True, help="Disposable v4 staging index name")
    parser.add_argument("--service", default=os.environ.get("AZURE_SEARCH_SERVICE", ""))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--execute", action="store_true", help="Upload after validation; default is validation-only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_staging_target(args.index)
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch size must be between 1 and {MAX_BATCH_SIZE}")
    documents = load_documents(args.artifact)
    if args.execute:
        if not args.service:
            raise ValueError("--service or AZURE_SEARCH_SERVICE is required with --execute")
        count = upload_documents(args.index, args.service, documents, args.batch_size)
        print(json.dumps({"index": args.index, "uploaded": count, "validated": len(documents)}))
    else:
        print(json.dumps({"index": args.index, "validated": len(documents), "execute": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())