#!/usr/bin/env python3
"""
Restore a v3 backup JSON file directly to a fresh Azure AI Search index.

This script:
1. Creates the search index with the correct schema (no permission filtering)
2. Reads documents from the backup JSON
3. Computes embeddings via Azure OpenAI (text-embedding-3-large, 3072 dims)
4. Uploads documents in small batches (100 per batch) to avoid payload limits

Usage:
    python scripts/restore_backup_to_index.py

Environment variables (from .env):
    AZURE_SEARCH_SERVICE        - Search service name
    AZURE_OPENAI_ENDPOINT       - OpenAI endpoint URL
    AZURE_OPENAI_EMB_DEPLOYMENT - Embedding deployment name
    AZURE_OPENAI_EMB_DIMENSIONS - Embedding dimensions (3072)
"""

import asyncio
import json
import logging
import os
import sys
import time

from azure.identity import AzureDeveloperCliCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from openai import AzureOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Configuration
BACKUP_FILE = "data/legal-scraper/backup/legal-court-rag-index-v3_backup.json"
INDEX_NAME = "legal-court-rag-index-v3"
UPLOAD_BATCH_SIZE = 100  # Small batches to avoid 16MB payload limit with 3072-dim vectors
EMBEDDING_BATCH_SIZE = 16  # OpenAI embedding API batch size

# Index schema fields (matching the app's expected schema)
INDEX_FIELDS = ["id", "content", "category", "sourcepage", "sourcefile", "storageUrl", "oids", "groups"]


def load_dotenv():
    """Load .env file from repo root."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key.strip(), value)


def create_index(index_client: SearchIndexClient, openai_endpoint: str, emb_deployment: str):
    """Create the search index with vector search (no permission filtering)."""
    index = SearchIndex(
        name=INDEX_NAME,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="sourcepage", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="sourcefile", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="storageUrl", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="oids",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                searchable=True,
                filterable=True,
                facetable=True,
            ),
            SearchField(
                name="groups",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                searchable=True,
                filterable=True,
                facetable=True,
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=3072,
                vector_search_profile_name="embedding-profile",
            ),
        ],
        semantic_search=SemanticSearch(
            default_configuration_name="default",
            configurations=[
                SemanticConfiguration(
                    name="default",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="sourcepage"),
                        content_fields=[SemanticField(field_name="content")],
                    ),
                )
            ],
        ),
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw_config")],
            profiles=[VectorSearchProfile(name="embedding-profile", algorithm_configuration_name="hnsw_config",
                                          vectorizer_name=f"{INDEX_NAME}-vectorizer")],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name=f"{INDEX_NAME}-vectorizer",
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=openai_endpoint,
                        deployment_name=emb_deployment,
                        model_name=emb_deployment,
                    ),
                )
            ],
        ),
        # No permission_filter_option — defaults to disabled
    )

    try:
        index_client.delete_index(INDEX_NAME)
        logger.info("Deleted existing index '%s'", INDEX_NAME)
    except Exception:
        pass

    index_client.create_index(index)
    logger.info("Created index '%s'", INDEX_NAME)


def truncate_text(text: str, max_chars: int = 28000) -> str:
    """Truncate text to stay within embedding model token limits (~8192 tokens).
    Using ~3.5 chars per token as a rough estimate, 28000 chars ≈ 8000 tokens."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def compute_embeddings(openai_client: AzureOpenAI, texts: list[str], deployment: str, dimensions: int) -> list[list[float]]:
    """Compute embeddings for a list of texts in batches, with retry for oversized inputs."""
    all_embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        # Truncate any texts that might exceed the token limit
        batch = [truncate_text(t) for t in batch]
        try:
            response = openai_client.embeddings.create(input=batch, model=deployment, dimensions=dimensions)
        except Exception as e:
            if "maximum context length" in str(e):
                # Fall back to per-document embedding with aggressive truncation
                logger.warning("Batch %d-%d too large, falling back to per-document embedding", i, i + len(batch))
                for j, text in enumerate(batch):
                    truncated = truncate_text(text, max_chars=20000)
                    try:
                        resp = openai_client.embeddings.create(input=[truncated], model=deployment, dimensions=dimensions)
                        all_embeddings.append(resp.data[0].embedding)
                    except Exception as e2:
                        logger.warning("Doc %d still too large, truncating to 15000 chars: %s", i + j, str(e2)[:80])
                        truncated = text[:15000]
                        resp = openai_client.embeddings.create(input=[truncated], model=deployment, dimensions=dimensions)
                        all_embeddings.append(resp.data[0].embedding)
                continue
            raise
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        if i > 0 and i % (EMBEDDING_BATCH_SIZE * 10) == 0:
            logger.info("  Computed embeddings for %d/%d texts", i + len(batch), len(texts))
    return all_embeddings


def main():
    load_dotenv()

    search_service = os.environ.get("AZURE_SEARCH_SERVICE", "")
    openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    emb_deployment = os.environ.get("AZURE_OPENAI_EMB_DEPLOYMENT", "text-embedding-3-large")
    emb_dimensions = int(os.environ.get("AZURE_OPENAI_EMB_DIMENSIONS", "3072"))

    if not search_service:
        logger.error("AZURE_SEARCH_SERVICE not set")
        sys.exit(1)
    if not openai_endpoint:
        logger.error("AZURE_OPENAI_ENDPOINT not set")
        sys.exit(1)

    search_endpoint = f"https://{search_service}.search.windows.net"

    logger.info("Search service: %s", search_endpoint)
    logger.info("OpenAI endpoint: %s", openai_endpoint)
    logger.info("Embedding deployment: %s (dims=%d)", emb_deployment, emb_dimensions)

    # Load backup
    logger.info("Loading backup from %s", BACKUP_FILE)
    with open(BACKUP_FILE) as f:
        backup = json.load(f)

    docs = backup["documents"]
    logger.info("Loaded %d documents from backup", len(docs))
    logger.info("Categories: %s", json.dumps(backup["metadata"].get("categories", {}), indent=2))

    # Initialize clients
    cred = AzureDeveloperCliCredential()
    index_client = SearchIndexClient(search_endpoint, cred)
    openai_client = AzureOpenAI(
        azure_endpoint=openai_endpoint,
        azure_ad_token_provider=lambda: cred.get_token("https://cognitiveservices.azure.com/.default").token,
        api_version="2024-06-01",
    )

    # Step 1: Create index
    create_index(index_client, openai_endpoint, emb_deployment)

    # Step 2: Prepare documents (strip extra fields, compute embeddings)
    search_client = SearchClient(search_endpoint, INDEX_NAME, cred)

    # Extract texts for embedding
    texts = [doc.get("content", "") for doc in docs]
    logger.info("Computing embeddings for %d documents...", len(texts))
    start = time.time()
    embeddings = compute_embeddings(openai_client, texts, emb_deployment, emb_dimensions)
    elapsed = time.time() - start
    logger.info("Computed all embeddings in %.1f seconds", elapsed)

    # Step 3: Upload in batches
    total_uploaded = 0
    for batch_start in range(0, len(docs), UPLOAD_BATCH_SIZE):
        batch_docs = docs[batch_start : batch_start + UPLOAD_BATCH_SIZE]
        batch_embeddings = embeddings[batch_start : batch_start + UPLOAD_BATCH_SIZE]

        upload_docs = []
        for doc, emb in zip(batch_docs, batch_embeddings):
            upload_doc = {
                "id": doc["id"],
                "content": doc.get("content", ""),
                "category": doc.get("category", ""),
                "sourcepage": doc.get("sourcepage", ""),
                "sourcefile": doc.get("sourcefile", ""),
                "storageUrl": doc.get("storageUrl", ""),
                "oids": doc.get("oids", []),
                "groups": doc.get("groups", []),
                "embedding": emb,
            }
            upload_docs.append(upload_doc)

        results = search_client.upload_documents(upload_docs)
        succeeded = sum(1 for r in results if r.succeeded)
        failed = sum(1 for r in results if not r.succeeded)
        total_uploaded += succeeded

        batch_num = batch_start // UPLOAD_BATCH_SIZE + 1
        total_batches = (len(docs) + UPLOAD_BATCH_SIZE - 1) // UPLOAD_BATCH_SIZE
        logger.info(
            "Batch %d/%d: uploaded %d/%d (failed: %d) — total: %d/%d",
            batch_num,
            total_batches,
            succeeded,
            len(upload_docs),
            failed,
            total_uploaded,
            len(docs),
        )

        if failed > 0:
            for r in results:
                if not r.succeeded:
                    logger.error("  Failed doc %s: %s (status %d)", r.key, r.error_message, r.status_code)

    logger.info("Restore complete: %d/%d documents uploaded", total_uploaded, len(docs))

    # Step 4: Verify
    import time as t
    t.sleep(3)
    r = search_client.search("*", top=1, include_total_count=True)
    count = r.get_count()
    logger.info("Verification: %d documents searchable in index", count)

    stats = index_client.get_index_statistics(INDEX_NAME)
    logger.info("Index stats: %d documents, %.1f MB", stats["document_count"], stats["storage_size"] / 1024 / 1024)


if __name__ == "__main__":
    main()
