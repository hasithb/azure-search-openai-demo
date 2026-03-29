#!/usr/bin/env python
"""
Export Court Guides from the v2 Azure Search index as a backup.

Fetches all non-CPR documents (Court Guides) from legal-court-rag-index-v2
and saves them to data/legal-scraper/backup/court_guides_backup.json.

This backup is a CRITICAL prerequisite before deleting or recreating the
index, since Court Guides are NOT in the Upload folder.

Usage:
    python export_court_guides_from_v2.py [--all] [--index INDEX_NAME]

Options:
    --all           Export ALL documents (not just Court Guides)
    --index         Source index name (default: legal-court-rag-index-v2)
"""
import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Known Court Guide categories — expand if more are added
COURT_GUIDE_CATEGORIES = [
    "Commercial Court Guide",
    "Kings Bench Division Guide",
    "Chancery Guide",
    "Patents Court Guide",
    "Technology and Construction Court Guide",
    "Technology & Construction Court Guide",    # alternate form
    "Kings Bench Guide",
    "King's Bench Division Guide",
    "King's Bench Guide",
    "Commercial Court",
    "Technology and Construction Court",
    "King's Bench Division",
    "Kings Bench Division",
    "Patents Court",
    "Chancery Division",
    "Chancery",
]

# CPR categories — used to identify non-Court-Guide documents when exporting all
CPR_CATEGORIES = [
    "Civil Procedure Rules",
    "CPR",
    "Legal Document",  # default upstream category
]


def build_credential():
    from azure.identity import DefaultAzureCredential
    from azure.core.credentials import AzureKeyCredential
    key = Config.AZURE_SEARCH_KEY
    if key:
        return AzureKeyCredential(key)
    logger.info("Using DefaultAzureCredential (no AZURE_SEARCH_KEY set)")
    return DefaultAzureCredential()


def get_endpoint() -> str:
    endpoint = Config.AZURE_SEARCH_SERVICE
    if not endpoint:
        raise ValueError("AZURE_SEARCH_SERVICE is not configured")
    if not endpoint.startswith("https://"):
        endpoint = f"https://{endpoint}.search.windows.net"
    return endpoint


def export_documents(
    index_name: str,
    export_all: bool = False,
    output_path: str | None = None,
) -> list[dict]:
    """
    Export documents from the given index.

    When export_all=False (default), only Court Guide categories are fetched.
    When export_all=True, all documents are fetched.
    """
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.core.exceptions import ResourceNotFoundError

    endpoint = get_endpoint()
    credential = build_credential()

    # Verify the source index exists
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    try:
        idx = index_client.get_index(index_name)
        logger.info(f"✅ Source index '{idx.name}' found")
    except ResourceNotFoundError:
        logger.error(f"❌ Index '{index_name}' does not exist — cannot export")
        return []

    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    # ── Discover what categories actually exist ──────────────────────────────
    logger.info("Discovering categories in index...")
    try:
        facet_results = client.search(
            search_text="*",
            facets=["category,count:100"],
            top=0,
            include_total_count=True,
        )
        total_docs = facet_results.get_count()
        logger.info(f"Total documents in index: {total_docs}")

        facets = facet_results.get_facets()
        if facets and "category" in facets:
            discovered_categories = {f["value"]: f["count"] for f in facets["category"]}
            logger.info("Categories found in index:")
            for cat, count in sorted(discovered_categories.items(), key=lambda x: -x[1]):
                logger.info(f"  {cat!r}: {count} documents")
        else:
            discovered_categories = {}
            logger.warning("No category facets returned — exporting without category filter")
    except Exception as e:
        logger.warning(f"Facet query failed ({e}), proceeding without category discovery")
        discovered_categories = {}
        total_docs = None

    # ── Determine which categories to export ─────────────────────────────────
    if export_all:
        filter_expr = None
        desc = "ALL documents"
    else:
        # Match any known Court Guide category (case-insensitive via lower check)
        matched = [
            cat for cat in discovered_categories
            if any(
                known.lower() in cat.lower() or cat.lower() in known.lower()
                for known in COURT_GUIDE_CATEGORIES
            )
        ]

        if not matched:
            # Fallback: export everything that is NOT a known CPR category
            logger.warning(
                "No Court Guide categories matched — will export everything "
                "except known CPR categories as a safety net"
            )
            cpr_filters = " and ".join(
                [f"category ne '{c.replace(chr(39), chr(39)+chr(39))}'" for c in CPR_CATEGORIES]
            )
            filter_expr = cpr_filters if cpr_filters else None
            desc = "non-CPR documents (fallback)"
        else:
            logger.info(f"Exporting Court Guide categories: {matched}")
            # OData string literals require single quotes to be escaped as ''
            cat_filters = " or ".join(
                [f"category eq '{c.replace(chr(39), chr(39)+chr(39))}'" for c in matched]
            )
            filter_expr = f"({cat_filters})"
            desc = f"Court Guides ({', '.join(matched)})"

    # ── Detect available retrievable fields ──────────────────────────────────
    # The v2 index may not have subsection_id/subsections/embedding as retrievable.
    # Probe the index schema to find which fields are retrievable.
    retrievable_fields = None  # None = return all defaults
    try:
        idx_def = index_client.get_index(index_name)
        retrievable_fields = [
            f.name for f in idx_def.fields
            if getattr(f, 'retrievable', True)
        ]
        # Always exclude embedding (write-only / not retrievable in v2)
        # Always exclude subsection_id and subsections (v3-only)
        skip_fields = {"embedding", "subsection_id", "subsections"}
        retrievable_fields = [f for f in retrievable_fields if f not in skip_fields]
        logger.info(f"Retrievable fields for export: {retrievable_fields}")
    except Exception as e:
        logger.warning(f"Could not probe index schema ({e}) — using safe defaults")
        retrievable_fields = ["id", "content", "category", "sourcepage", "sourcefile", "storageUrl", "updated"]

    # ── Paginate through all results ──────────────────────────────────────────
    logger.info(f"Exporting: {desc}")

    documents: list[dict] = []
    page_size = 100
    skip = 0
    page_num = 0

    while True:
        try:
            results = client.search(
                search_text="*",
                filter=filter_expr,
                select=retrievable_fields,
                top=page_size,
                skip=skip,
                include_total_count=(skip == 0),
            )

            if skip == 0:
                count = results.get_count()
                if count is not None:
                    logger.info(f"Matched document count: {count}")

            batch = list(results)
            if not batch:
                break

            documents.extend(batch)
            skip += len(batch)
            page_num += 1

            logger.info(
                f"⬇️  Page {page_num}: fetched {len(batch)} docs "
                f"(total so far: {len(documents)})"
            )

            if len(batch) < page_size:
                break

            # Brief pause to avoid throttling
            time.sleep(0.2)

        except Exception as e:
            logger.error(f"Error on page {page_num + 1} (skip={skip}): {e}")
            if documents:
                logger.warning("Partial export — saving what we have so far")
            break

    logger.info(f"Export complete: {len(documents)} documents fetched")

    # ── Save to disk ──────────────────────────────────────────────────────────
    if output_path is None:
        backup_dir = os.path.join(Config.SCRAPER_DATA_DIR, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        output_path = os.path.join(backup_dir, "court_guides_backup.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(output_path) / 1024
    logger.info(f"✅ Backup saved to: {output_path}")
    logger.info(f"   File size: {size_kb:.1f} KB")
    logger.info(f"   Documents: {len(documents)}")

    # Print category breakdown
    cat_counts: dict[str, int] = {}
    for doc in documents:
        c = doc.get("category", "(none)")
        cat_counts[c] = cat_counts.get(c, 0) + 1
    logger.info("Category breakdown:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat!r}: {cnt}")

    return documents


def main():
    parser = argparse.ArgumentParser(description="Export Court Guides from Azure Search v2 index")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export ALL documents (not just Court Guides)",
    )
    parser.add_argument(
        "--index",
        default=os.getenv("AZURE_SEARCH_INDEX", "legal-court-rag-index-v2"),
        help="Source index name (default: legal-court-rag-index-v2)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: data/legal-scraper/backup/court_guides_backup.json)",
    )
    args = parser.parse_args()

    valid, errors = Config.validate_azure_config()
    if not valid:
        logger.error("Azure configuration invalid:")
        for err in errors:
            logger.error(f"  - {err}")
        sys.exit(1)

    docs = export_documents(
        index_name=args.index,
        export_all=args.all,
        output_path=args.output,
    )

    if not docs:
        logger.error("No documents exported — check configuration and index name")
        sys.exit(1)

    logger.info("Export successful. Run validate_backup.py to verify.")


if __name__ == "__main__":
    main()
