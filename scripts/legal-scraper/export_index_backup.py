#!/usr/bin/env python
"""
Export all documents from an Azure Search index to a local JSON backup.

Exports every document (CPR, Court Guides, etc.) with all fields except
the embedding vector (which is large and can be regenerated).

Usage:
    python export_index_backup.py --index legal-court-rag-index-v2
    python export_index_backup.py --index legal-court-rag-index
    python export_index_backup.py --index legal-court-rag-index-v2 --include-embeddings

Output:
    data/legal-scraper/backup/<index-name>_backup.json
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def export_index(index_name: str, include_embeddings: bool = False) -> list[dict]:
    """Export all documents from the given index."""
    from azure.search.documents import SearchClient
    from azure.identity import AzureDeveloperCliCredential

    service_name = Config.AZURE_SEARCH_SERVICE
    endpoint = f"https://{service_name}.search.windows.net"
    tenant_id = os.getenv("AZURE_TENANT_ID")
    cred = AzureDeveloperCliCredential(tenant_id=tenant_id)
    client = SearchClient(endpoint, index_name, cred)

    # Determine fields to select (exclude embedding by default)
    exclude = set() if include_embeddings else {"embedding"}
    from azure.search.documents.indexes import SearchIndexClient

    idx_client = SearchIndexClient(endpoint, cred)
    idx = idx_client.get_index(index_name)
    all_fields = [f.name for f in idx.fields if f.name not in exclude]

    logger.info("Index: %s", index_name)
    logger.info("Fields to export: %s", all_fields)
    logger.info("Embeddings: %s", "included" if include_embeddings else "excluded (use --include-embeddings to include)")

    # Use search("*") to page through all documents
    documents = []
    results = client.search("*", select=all_fields, top=1000, include_total_count=True)
    total = results.get_count()
    logger.info("Total documents in index: %d", total)

    for doc in results:
        # Remove @search metadata keys
        clean = {k: v for k, v in doc.items() if not k.startswith("@search.")}
        documents.append(clean)

    logger.info("Exported %d documents", len(documents))

    if len(documents) != total:
        logger.warning(
            "Document count mismatch: expected %d, got %d. "
            "Index may have more than 1000 docs — using paging.",
            total,
            len(documents),
        )
        # Re-export with proper paging using search after
        documents = []
        batch_size = 1000
        skip = 0
        while True:
            results = client.search("*", select=all_fields, top=batch_size, skip=skip)
            batch = []
            for doc in results:
                clean = {k: v for k, v in doc.items() if not k.startswith("@search.")}
                batch.append(clean)
            if not batch:
                break
            documents.extend(batch)
            skip += len(batch)
            logger.info("  Paged: %d / %d", len(documents), total)
            if len(batch) < batch_size:
                break

        logger.info("Final export count: %d", len(documents))

    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Export all documents from an Azure Search index to a local JSON backup"
    )
    parser.add_argument(
        "--index",
        required=True,
        help="Name of the index to export",
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        default=False,
        help="Include embedding vectors in the backup (warning: very large files)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: data/legal-scraper/backup/<index>_backup.json)",
    )
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        backup_dir = Path(script_dir).parent.parent / "data" / "legal-scraper" / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        output_path = backup_dir / f"{args.index}_backup.json"

    logger.info("Exporting index '%s' to '%s'", args.index, output_path)

    documents = export_index(args.index, include_embeddings=args.include_embeddings)

    if not documents:
        logger.error("No documents exported — check index name and credentials")
        sys.exit(1)

    # Build backup metadata
    backup = {
        "metadata": {
            "index_name": args.index,
            "document_count": len(documents),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "includes_embeddings": args.include_embeddings,
            "fields": list(documents[0].keys()) if documents else [],
        },
        "documents": documents,
    }

    # Category summary
    categories = {}
    for doc in documents:
        cat = doc.get("category", "(no category)")
        categories[cat] = categories.get(cat, 0) + 1
    backup["metadata"]["categories"] = categories

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(backup, f, ensure_ascii=False, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Backup saved: %s (%.1f MB)", output_path, file_size_mb)
    logger.info("Documents: %d", len(documents))
    logger.info("Categories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        logger.info("  %-55s %d", cat, count)

    print(f"\n✅ Backup complete: {output_path} ({len(documents)} docs, {file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
