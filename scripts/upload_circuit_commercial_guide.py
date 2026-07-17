#!/usr/bin/env python3
"""Upload Circuit Commercial Court Guide to the v3 Azure Search index.

Reads the processed JSON (already has embeddings), enriches with subsection fields,
and uploads to legal-court-rag-index-v3.

Usage:
    python scripts/upload_circuit_commercial_guide.py [--dry-run]
"""
import json
import logging
import os
import re
import sys
import time

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.exceptions import ResourceNotFoundError

# Add backend to path for subsection extractor
backend_dir = os.path.join(os.path.dirname(__file__), '..', 'app', 'backend')
sys.path.insert(0, os.path.abspath(backend_dir))
from customizations.subsection_extractor import SubsectionExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Index field detection
_INDEX_FIELDS: set[str] | None = None


def load_index_fields():
    global _INDEX_FIELDS
    idx_client = SearchIndexClient(endpoint=ENDPOINT, credential=DefaultAzureCredential())
    index = idx_client.get_index(INDEX_NAME)
    _INDEX_FIELDS = {f.name for f in index.fields}
    logger.info("Index fields: %s", sorted(_INDEX_FIELDS))


INPUT_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'legal-scraper', 'processed', 'Upload',
    'Circuit-Commercial-Court-Guide-2023-web_processed.json'
)
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX", "legal-court-rag-index-v3")
_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE", "gptkb-gz2m4s637t5me")
ENDPOINT = f"https://{_SEARCH_SERVICE}.search.windows.net"


def sanitize_id(doc_id):
    s = re.sub(r'[^a-zA-Z0-9_\-=]', '_', doc_id)
    s = re.sub(r'_{2,}', '___', s)
    return s.strip('_')


def enrich_with_subsections(doc):
    content = doc.get("content", "")
    if isinstance(content, list):
        content = "\n".join(content)
    result = SubsectionExtractor.extract_subsections_dict(content)
    doc["subsection_id"] = result.get("primary", "") or ""
    doc["subsections"] = result.get("all", []) or []
    return doc


def map_to_index_schema(doc):
    content = doc.get("content", "")
    if isinstance(content, list):
        content = "\n".join(content)
    result = {
        "id": sanitize_id(doc.get("id", "")),
        "content": content,
        "embedding": doc.get("embedding", []),
        "category": doc.get("category", ""),
        "sourcepage": doc.get("sourcepage", ""),
        "sourcefile": doc.get("sourcefile", ""),
        "storageUrl": doc.get("storageUrl", ""),
        "oids": doc.get("oids", []) or [],
        "groups": doc.get("groups", []) or [],
    }
    # Only include extended fields if the target index supports them
    if _INDEX_FIELDS is not None:
        for fname, val in [
            ("parent_id", doc.get("parent_id", "") or ""),
            ("subsection_id", doc.get("subsection_id", "") or ""),
            ("subsections", doc.get("subsections", []) or []),
            ("updated", doc.get("updated", "") or ""),
        ]:
            if fname in _INDEX_FIELDS:
                result[fname] = val
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Upload Circuit Commercial Court Guide to v3 index")
    parser.add_argument("--dry-run", action="store_true", help="Preview without uploading")
    args = parser.parse_args()

    load_index_fields()

    # Load
    logger.info(f"Loading: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)
    logger.info(f"Loaded {len(documents)} documents")

    # Enrich
    logger.info("Enriching with subsection fields...")
    enriched = 0
    for doc in documents:
        enrich_with_subsections(doc)
        if doc.get("subsection_id"):
            enriched += 1
    logger.info(f"Subsection enrichment: {enriched}/{len(documents)} documents have a primary subsection")

    # Map
    mapped = [map_to_index_schema(doc) for doc in documents]

    # Validate
    no_embedding = sum(1 for d in mapped if not d.get("embedding"))
    no_content = sum(1 for d in mapped if not d.get("content"))
    logger.info(f"Validation: {no_embedding} without embeddings, {no_content} without content")

    if args.dry_run:
        logger.info(f"DRY RUN: Would upload {len(mapped)} documents to '{INDEX_NAME}'")
        for i, doc in enumerate(mapped[:5], 1):
            logger.info(f"  [{i}] id={doc['id'][:60]} | subsection_id={doc.get('subsection_id', '')}")
        if len(mapped) > 5:
            logger.info(f"  ... and {len(mapped) - 5} more")
        return

    # Upload
    credential = DefaultAzureCredential()
    index_client = SearchIndexClient(endpoint=ENDPOINT, credential=credential)
    try:
        index_client.get_index(INDEX_NAME)
        logger.info(f"Index '{INDEX_NAME}' found")
    except ResourceNotFoundError:
        logger.error(f"Index '{INDEX_NAME}' does not exist!")
        sys.exit(1)

    client = SearchClient(endpoint=ENDPOINT, index_name=INDEX_NAME, credential=credential)
    batch_size = 50
    total_uploaded = 0

    for i in range(0, len(mapped), batch_size):
        batch = mapped[i : i + batch_size]
        try:
            results = client.upload_documents(batch)
            succeeded = sum(1 for r in results if r.succeeded)
            failed = len(batch) - succeeded
            total_uploaded += succeeded
            logger.info(f"Batch {i // batch_size + 1}: {succeeded} uploaded, {failed} failed (total: {total_uploaded}/{len(mapped)})")
            if failed:
                for r in results:
                    if not r.succeeded:
                        logger.warning(f"  Failed: key={r.key}, error={r.error_message}")
        except Exception as e:
            logger.error(f"Batch upload error: {e}")
        time.sleep(0.1)

    logger.info(f"Upload complete: {total_uploaded}/{len(mapped)} documents to '{INDEX_NAME}'")


if __name__ == "__main__":
    main()
