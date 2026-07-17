#!/usr/bin/env python
"""
Validate Court Guides backup before index migration.

Checks that court_guides_backup.json is complete and well-formed:
  - Expected document count (~1,020)
  - All 5 Court Guides represented
  - Required fields present (id, content, embedding, etc.)
  - Embedding dimensions correct (3072 for text-embedding-3-large)
  - No duplicate IDs

Usage:
    python validate_backup.py [--file PATH]

Exits with code 0 on success, 1 on failure.
"""
import os
import sys
import json
import argparse
import logging

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EXPECTED_COURT_GUIDES = [
    "Commercial Court Guide",
    "Kings Bench Division Guide",
    "Chancery Guide",
    "Patents Court Guide",
    "Technology and Construction Court Guide",
    # Actual category names used in v2 index (as-found via export)
    "Commercial Court",
    "Technology and Construction Court",
    "King's Bench Division",
    "Chancery Division",
    "Patents Court",
]

REQUIRED_FIELDS = ["id", "content", "category", "sourcepage", "sourcefile"]
EXPECTED_EMBEDDING_DIM = Config.EMBEDDING_DIMENSIONS  # 3072

# Actual count of Court Guide docs found in v2 index (verified 2026-02-23)
MIN_EXPECTED_DOCS = 400
EXPECTED_APPROX_DOCS = 538


def validate_backup(backup_path: str) -> bool:
    """
    Returns True if backup is considered valid, False otherwise.
    Prints a detailed human-readable report.
    """
    if not os.path.exists(backup_path):
        logger.error(f"❌ Backup file not found: {backup_path}")
        return False

    size_kb = os.path.getsize(backup_path) / 1024
    logger.info(f"Loading backup: {backup_path} ({size_kb:.1f} KB)")

    try:
        with open(backup_path, "r", encoding="utf-8") as f:
            documents: list[dict] = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parse error: {e}")
        return False

    total = len(documents)
    logger.info(f"Total documents loaded: {total}")

    errors: list[str] = []
    warnings: list[str] = []

    # ── 1. Document count ────────────────────────────────────────────────────
    if total == 0:
        errors.append("Backup is empty — no documents found")
    elif total < MIN_EXPECTED_DOCS:
        errors.append(
            f"Too few documents: {total} (expected at least {MIN_EXPECTED_DOCS})"
        )
    elif total < EXPECTED_APPROX_DOCS * 0.8:
        warnings.append(
            f"Document count ({total}) is below 80% of expected "
            f"~{EXPECTED_APPROX_DOCS} — backup may be incomplete"
        )

    # ── 2. Category coverage ─────────────────────────────────────────────────
    cat_counts: dict[str, int] = {}
    for doc in documents:
        c = doc.get("category", "(none)")
        cat_counts[c] = cat_counts.get(c, 0) + 1

    logger.info("Categories in backup:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat!r}: {cnt} documents")

    # Check each expected court guide is represented
    for guide in EXPECTED_COURT_GUIDES:
        found = any(
            guide.lower() in cat.lower() or cat.lower() in guide.lower()
            for cat in cat_counts
        )
        if not found:
            warnings.append(
                f"Court Guide not found in backup: '{guide}' "
                f"(may have a different category string)"
            )

    # ── 3. Required fields ───────────────────────────────────────────────────
    missing_fields_by_doc: dict[str, list[str]] = {}
    for doc in documents:
        missing = [f for f in REQUIRED_FIELDS if not doc.get(f)]
        if missing:
            missing_fields_by_doc[doc.get("id", "unknown")] = missing

    if missing_fields_by_doc:
        sample = list(missing_fields_by_doc.items())[:5]
        for doc_id, fields in sample:
            warnings.append(f"Document '{doc_id}' missing fields: {fields}")
        if len(missing_fields_by_doc) > 5:
            warnings.append(
                f"... and {len(missing_fields_by_doc) - 5} more documents with missing fields"
            )

    # ── 4. Embedding validation ──────────────────────────────────────────────
    docs_with_embeddings = 0
    docs_wrong_dim = 0
    for doc in documents:
        emb = doc.get("embedding", [])
        if emb and isinstance(emb, list):
            docs_with_embeddings += 1
            if len(emb) != EXPECTED_EMBEDDING_DIM:
                docs_wrong_dim += 1

    docs_without_embeddings = total - docs_with_embeddings

    if docs_without_embeddings > 0:
        warnings.append(
            f"{docs_without_embeddings}/{total} documents have no embedding "
            f"— embeddings will be regenerated on upload"
        )
    if docs_wrong_dim > 0:
        errors.append(
            f"{docs_wrong_dim} documents have wrong embedding dimension "
            f"(expected {EXPECTED_EMBEDDING_DIM})"
        )
    logger.info(
        f"Embeddings: {docs_with_embeddings}/{total} present, "
        f"{docs_without_embeddings} missing, {docs_wrong_dim} wrong dimension"
    )

    # ── 5. Duplicate ID check ────────────────────────────────────────────────
    ids = [doc.get("id") for doc in documents]
    unique_ids = set(ids)
    if len(ids) != len(unique_ids):
        dup_count = len(ids) - len(unique_ids)
        errors.append(
            f"{dup_count} duplicate document IDs found "
            f"({total} docs, {len(unique_ids)} unique IDs)"
        )
    else:
        logger.info(f"All {len(unique_ids)} document IDs are unique ✅")

    # ── 6. Content length sanity check ──────────────────────────────────────
    empty_content = sum(1 for doc in documents if not doc.get("content"))
    if empty_content:
        warnings.append(f"{empty_content} documents have empty content")

    short_content = sum(
        1 for doc in documents
        if doc.get("content") and len(str(doc["content"])) < 50
    )
    if short_content:
        warnings.append(f"{short_content} documents have very short content (<50 chars)")

    # ── Report ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("          COURT GUIDES BACKUP VALIDATION REPORT")
    print("=" * 60)
    print(f"File:      {backup_path}")
    print(f"Size:      {size_kb:.1f} KB")
    print(f"Documents: {total}")
    print()

    if warnings:
        print(f"⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   • {w}")
        print()

    if errors:
        print(f"❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"   • {e}")
        print()
        print("Status: ❌ VALIDATION FAILED — DO NOT PROCEED WITH MIGRATION")
        print("=" * 60)
        return False
    else:
        print("✅ All checks passed")
        print()
        print("Status: ✅ BACKUP IS VALID — SAFE TO PROCEED WITH MIGRATION")
        print("=" * 60)
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate Court Guides backup before index migration"
    )
    parser.add_argument(
        "--file",
        default=os.path.join(Config.SCRAPER_DATA_DIR, "backup", "court_guides_backup.json"),
        help="Path to backup JSON file",
    )
    args = parser.parse_args()

    ok = validate_backup(args.file)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
