#!/usr/bin/env python
"""
Upload Court Guides from backup JSON to the target Azure Search index.

This is the last phase of the index v3 migration:
  1. Load court_guides_backup.json (created by export_court_guides_from_v2.py)
  2. Enrich each document with subsection_id and subsections
  3. Generate embeddings for documents that are missing them
  4. Upload to the target index (default: legal-court-rag-index-v3)

Usage:
    python upload_court_guides_backup.py [--index INDEX] [--dry-run]

Options:
    --index     Target index (default: AZURE_SEARCH_INDEX env or legal-court-rag-index-v3)
    --dry-run   Show what would be uploaded without doing it
    --file      Path to backup JSON (default: backup/court_guides_backup.json)
"""
import os
import sys
import json
import argparse
import logging
import re
import time
from pathlib import Path

from openai import AzureOpenAI, RateLimitError, APIConnectionError, APIError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Setup paths — insert script_dir first so config.py resolves to scripts/legal-scraper/config.py
# Then insert backend_dir so customizations.subsection_extractor is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(script_dir, "../../app/backend")
sys.path.insert(0, script_dir)

from config import Config  # scripts/legal-scraper/config.py

# Add backend dir AFTER importing Config to avoid shadowing
sys.path.insert(0, backend_dir)
from customizations.subsection_extractor import SubsectionExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def sanitize_id(doc_id: str) -> str:
    """Sanitize document ID for Azure Search (same logic as upload_with_embeddings.py)."""
    s = re.sub(r'[^a-zA-Z0-9_\-=]', '_', doc_id)
    s = re.sub(r'_{2,}', '___', s)
    return s.strip('_')


def build_credential():
    key = Config.AZURE_SEARCH_KEY
    if key:
        return AzureKeyCredential(key)
    logger.info("Using DefaultAzureCredential for Azure Search")
    return DefaultAzureCredential()


def get_endpoint() -> str:
    endpoint = Config.AZURE_SEARCH_SERVICE
    if not endpoint:
        raise ValueError("AZURE_SEARCH_SERVICE is not configured")
    if not endpoint.startswith("https://"):
        endpoint = f"https://{endpoint}.search.windows.net"
    return endpoint


# ── Embedding generation ──────────────────────────────────────────────────────

import tiktoken

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(5),
)
def create_embeddings_with_retry(client, texts: list, model: str):
    return client.embeddings.create(input=texts, model=model)


def generate_missing_embeddings(documents: list) -> list:
    """Generate embeddings for documents that are missing them."""
    docs_to_embed = [doc for doc in documents if not doc.get("embedding")]
    if not docs_to_embed:
        logger.info("All documents already have embeddings ✅")
        return documents

    logger.info(f"Generating embeddings for {len(docs_to_embed)} documents...")

    endpoint = Config.AZURE_OPENAI_SERVICE
    if not endpoint.startswith("https://"):
        endpoint = f"https://{endpoint}.openai.azure.com"
    deployment = Config.AZURE_OPENAI_EMB_DEPLOYMENT

    if Config.AZURE_OPENAI_KEY:
        client = AzureOpenAI(
            api_key=Config.AZURE_OPENAI_KEY,
            api_version="2023-05-15",
            azure_endpoint=endpoint,
            max_retries=3,
            timeout=120.0,
        )
    else:
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        client = AzureOpenAI(
            azure_ad_token_provider=token_provider,
            api_version="2023-05-15",
            azure_endpoint=endpoint,
            max_retries=3,
            timeout=120.0,
        )

    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    max_tokens = 8000

    def clamp(text: str) -> str:
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoding.decode(tokens[:max_tokens])

    batch_size = 100
    success = 0
    for i in range(0, len(docs_to_embed), batch_size):
        batch = docs_to_embed[i : i + batch_size]
        texts = []
        for doc in batch:
            content = doc.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            texts.append(clamp(content.replace("\n", " ")))

        try:
            response = create_embeddings_with_retry(client, texts, deployment)
            for j, data in enumerate(response.data):
                batch[j]["embedding"] = data.embedding
            success += len(batch)
            logger.info(
                f"✅ Embeddings: {success}/{len(docs_to_embed)} "
                f"({success / len(docs_to_embed) * 100:.1f}%)"
            )
            if i + batch_size < len(docs_to_embed):
                time.sleep(0.5)
        except Exception as e:
            logger.error(f"❌ Embedding batch {i // batch_size + 1} failed: {e}")
            logger.warning(f"Skipping {len(batch)} documents")

    return documents


# ── Document enrichment ────────────────────────────────────────────────────────

def enrich_with_subsections(doc: dict) -> dict:
    """Add/update subsection_id and subsections using SubsectionExtractor."""
    content = doc.get("content", "")
    if isinstance(content, list):
        content = "\n".join(content)

    result = SubsectionExtractor.extract_subsections_dict(content)
    subsection_id = result.get("primary", "") or ""
    subsections = result.get("all", []) or []

    doc["subsection_id"] = subsection_id
    doc["subsections"] = subsections
    return doc


def map_to_index_schema(doc: dict) -> dict:
    """Map document to Azure Search index schema, enriching missing fields."""
    doc_id = doc.get("id", "")
    sanitized_id = sanitize_id(doc_id)
    if doc_id != sanitized_id:
        logger.debug(f"Sanitized ID: '{doc_id}' -> '{sanitized_id}'")

    content = doc.get("content", "")
    if isinstance(content, list):
        content = "\n".join(content)

    return {
        "id": sanitized_id,
        "content": content,
        "embedding": doc.get("embedding", []),
        "category": doc.get("category", ""),
        "sourcepage": doc.get("sourcepage", ""),
        "sourcefile": doc.get("sourcefile", ""),
        "storageUrl": doc.get("storageUrl", ""),
        "oids": doc.get("oids", []) or [],
        "groups": doc.get("groups", []) or [],
        "parent_id": doc.get("parent_id", "") or "",
        "subsection_id": doc.get("subsection_id", "") or "",
        "subsections": doc.get("subsections", []) or [],
        "updated": doc.get("updated", "") or "",
    }


# ── Upload ─────────────────────────────────────────────────────────────────────

def upload_documents(
    index_name: str,
    documents: list[dict],
    batch_size: int = 100,
    dry_run: bool = False,
) -> int:
    endpoint = get_endpoint()
    credential = build_credential()

    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    try:
        index_client.get_index(index_name)
        logger.info(f"✅ Target index '{index_name}' found")
    except ResourceNotFoundError:
        logger.error(
            f"❌ Index '{index_name}' does not exist. "
            f"Run: AZURE_SEARCH_INDEX={index_name} python create_index.py"
        )
        return 0

    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    if dry_run:
        logger.info(f"DRY RUN: Would upload {len(documents)} Court Guide documents to '{index_name}'")
        for i, doc in enumerate(documents[:10], 1):
            logger.info(f"  [{i}] {doc.get('id', '?')} | category={doc.get('category')} "
                        f"| subsection_id={doc.get('subsection_id', '(none)')}")
        if len(documents) > 10:
            logger.info(f"  ... and {len(documents) - 10} more")
        return 0

    total_uploaded = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        try:
            results = client.upload_documents(batch)
            succeeded = sum(1 for r in results if r.succeeded)
            failed = len(batch) - succeeded
            total_uploaded += succeeded
            logger.info(
                f"Batch {i // batch_size + 1}: ✅ {succeeded} uploaded, "
                f"{'❌ ' + str(failed) + ' failed' if failed else '0 failed'} "
                f"(total: {total_uploaded}/{len(documents)})"
            )
            if failed:
                for r in results:
                    if not r.succeeded:
                        logger.warning(f"  Failed: key={r.key}, error={r.error_message}")
        except Exception as e:
            logger.error(f"Batch {i // batch_size + 1} upload error: {e}")

        time.sleep(0.1)  # Brief pause between batches

    logger.info(f"Upload complete: {total_uploaded}/{len(documents)} documents uploaded to '{index_name}'")
    return total_uploaded


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Upload Court Guides backup to Azure Search index (phase 4 of v3 migration)"
    )
    parser.add_argument(
        "--index",
        default=os.getenv("AZURE_SEARCH_INDEX", "legal-court-rag-index-v3"),
        help="Target index name (default: legal-court-rag-index-v3)",
    )
    parser.add_argument(
        "--file",
        default=os.path.join(Config.SCRAPER_DATA_DIR, "backup", "court_guides_backup.json"),
        help="Path to backup JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding generation (upload without embeddings — not recommended)",
    )
    args = parser.parse_args()

    valid, errors = Config.validate_azure_config()
    if not valid:
        logger.error("Azure configuration invalid:")
        for err in errors:
            logger.error(f"  - {err}")
        sys.exit(1)

    # ── Load backup ──────────────────────────────────────────────────────────
    if not os.path.exists(args.file):
        logger.error(f"Backup file not found: {args.file}")
        logger.error("Run export_court_guides_from_v2.py first")
        sys.exit(1)

    logger.info(f"Loading backup: {args.file}")
    with open(args.file, "r", encoding="utf-8") as f:
        documents: list[dict] = json.load(f)

    logger.info(f"Loaded {len(documents)} documents")

    # ── Enrich with subsections ───────────────────────────────────────────────
    logger.info("Enriching documents with subsection fields...")
    enriched = 0
    for doc in documents:
        doc = enrich_with_subsections(doc)
        if doc.get("subsection_id"):
            enriched += 1

    logger.info(
        f"Subsection enrichment: {enriched}/{len(documents)} documents have a primary subsection "
        f"({(enriched / len(documents) * 100):.1f}%)"
    )

    # ── Generate missing embeddings ───────────────────────────────────────────
    if not args.skip_embedding and not args.dry_run:
        documents = generate_missing_embeddings(documents)

    # ── Map to index schema ───────────────────────────────────────────────────
    mapped = [map_to_index_schema(doc) for doc in documents]

    # ── Validation ────────────────────────────────────────────────────────────
    invalid = [(d.get("id", "?"), "Missing content") for d in mapped if not d.get("content")]
    if invalid:
        logger.warning(f"{len(invalid)} documents have no content — they will still be uploaded")

    # ── Upload ────────────────────────────────────────────────────────────────
    logger.info(f"Uploading {len(mapped)} Court Guide documents to index '{args.index}'...")
    count = upload_documents(
        index_name=args.index,
        documents=mapped,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        if count == len(mapped):
            logger.info("✅ All Court Guide documents uploaded successfully")
        else:
            logger.warning(f"⚠️  Only {count}/{len(mapped)} documents uploaded")
            sys.exit(1)


if __name__ == "__main__":
    main()
