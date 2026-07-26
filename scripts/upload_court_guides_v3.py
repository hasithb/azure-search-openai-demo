#!/usr/bin/env python3
"""
Upload court guides from Azure DI pipeline to index v3.

Steps:
  1. Load JSON files from outputs_azure_di/
  2. Query index for existing court-guide documents → collect IDs to delete
  3. Delete orphaned old docs
  4. Map new docs to index schema (subsections, headers, IDs)
  5. Generate embeddings via Azure OpenAI
  6. Upload to Azure AI Search index v3

Usage:
    # Dry run — show what would happen, no changes:
    python scripts/upload_court_guides_v3.py --dry-run

    # Live upload:
    python scripts/upload_court_guides_v3.py

    # Skip specific guides:
    python scripts/upload_court_guides_v3.py --skip "Court of Appeal" --skip "Senior Courts Costs Office"
"""
import os
import sys
import json
import glob
import re
import time
import logging
import argparse
import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from load_azd_env import load_azd_env
from customizations.subsection_extractor import SubsectionExtractor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("upload_court_guides_v3")

# Suppress verbose Azure SDK logging
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Guide metadata — mirrors extract_court_guides_azure_di.py
# ---------------------------------------------------------------------------
GUIDE_FILES = {
    "Commercial Court": {
        "file": "14.341_JO_Commercial_Court_Guide_FINAL_processed.json",
        "sourcefile": "Commercial Court Guide",
        "category": "Commercial Court",
    },
    "King's Bench Division": {
        "file": "35.16_JO_Kings_Bench_Division_Guide_2025_WEB4_processed.json",
        "sourcefile": "King's Bench Division Guide",
        "category": "King's Bench Division",
    },
    "Chancery Division": {
        "file": "Chancery-Guide-2024-web_processed.json",
        "sourcefile": "Chancery Guide",
        "category": "Chancery Division",
    },
    "Patents Court": {
        "file": "Patents-Court-Guide-Updated-February-2025_processed.json",
        "sourcefile": "Patents Court Guide",
        "category": "Patents Court",
    },
    "Technology and Construction Court": {
        "file": "The-Technology-and-Construction-Court-Guide_processed.json",
        "sourcefile": "Technology and Construction Court Guide",
        "category": "Technology and Construction Court",
    },
    "Court of Appeal Civil Division": {
        "file": "35.67_JO_Court-of-Appeal-Civil-Division-Guide_FINAL_WEB_processed.json",
        "sourcefile": "Court of Appeal Civil Division Guide",
        "category": "Court of Appeal Civil Division",
    },
    "Intellectual Property Enterprise Court": {
        "file": "Intellectual-Property-Enterprise-Court-IPEC-Guide-revised-November-2024_processed.json",
        "sourcefile": "Intellectual Property Enterprise Court Guide",
        "category": "Intellectual Property Enterprise Court",
    },
    "Senior Courts Costs Office": {
        "file": "Senior-Courts-Costs-Office-Guide_processed.json",
        "sourcefile": "Senior Courts Costs Office Guide",
        "category": "Senior Courts Costs Office",
    },
}

# Old sourcefiles that may exist in the index for the same guides
# (needed to find orphaned docs to delete)
OLD_SOURCEFILES = {
    "Commercial Court Guide",
    "35.16_JO_Kings_Bench_Division_Guide_2025_WEB4.pdf",
    "King's Bench Division Guide",
    "Chancery Guide",
    "Patents Court Guide",
    "Technology and Construction Court Guide",
    "Court of Appeal Civil Division Guide",
    "Senior Courts Costs Office Guide",
}

INPUT_DIR = SCRIPT_DIR / "court_guides_processing_pipeline" / "outputs_azure_di"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_id(doc_id: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-=]", "_", doc_id)
    s = re.sub(r"_{2,}", "___", s)
    return s.strip("_")


def has_existing_header(text: str) -> bool:
    if not text:
        return False
    head = [line.strip() for line in text.splitlines()[:6] if line.strip()]
    return any(
        line.startswith(("SOURCE:", "SOURCEPAGE:", "SECTION:"))
        or (line.startswith("[") and ">" in line)
        for line in head
    )


def extract_subsection_from_sourcepage(value: str) -> str:
    if not value:
        return ""
    for pat in [
        r"\b([A-Z]\.\d+(?:\.\d+)?)\b",
        r"\b([A-Z]\d+\.\d+(?:\.\d+)?)\b",
        r"\b(\d+\.\d+(?:\.\d+)?)\b",
        r"\b([A-Z]\d+)\b",
    ]:
        m = re.search(pat, value, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def extract_parent_section(value: str) -> str:
    if not value:
        return ""
    raw = value.strip()
    first_segment = raw.split(",", 1)[0].strip()
    if re.match(r"^[A-Z]\.", first_segment) or re.match(
        r"^(Section|Appendix|Part|Practice Direction)\b", first_segment, re.IGNORECASE
    ):
        return first_segment
    for pat in [
        r"\b(Practice Direction\s+[0-9A-Z]+)\b",
        r"\b(Part\s+\d+[A-Z]?)\b",
        r"\b(Section\s+\d+)\b",
        r"\b(Appendix\s+[A-Z])\b",
        r"\b([A-Z]\.\s+[^,]+)\b",
    ]:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def map_doc(doc: dict, id_prefix: str = "") -> dict:
    """Map a raw JSON doc to the index v3 schema.
    id_prefix is prepended to the sanitized doc ID to avoid cross-guide collisions."""
    content = doc.get("content", "") or ""
    sourcepage = doc.get("sourcepage", "") or ""
    sourcefile = doc.get("sourcefile", "") or ""
    category = doc.get("category", "") or ""

    # Subsection extraction
    extracted_sub = SubsectionExtractor.extract_first_subsection(content)
    all_subs = list(SubsectionExtractor.extract_all_subsections(content))
    derived_sub = extract_subsection_from_sourcepage(sourcepage)
    parent = extract_parent_section(sourcepage)

    subsection_id = extracted_sub or derived_sub or parent or ""
    subsections = list(all_subs)
    if subsection_id and subsection_id not in subsections:
        subsections.insert(0, subsection_id)

    # Inject header into content
    if content and not has_existing_header(content):
        hdr = []
        if sourcefile:
            hdr.append(f"SOURCE: {sourcefile}")
        if sourcepage:
            hdr.append(f"SOURCEPAGE: {sourcepage}")
        if category:
            hdr.append(f"CATEGORY: {category}")
        if parent and parent != subsection_id:
            hdr.append(f"SECTION: {parent}")
        if subsection_id:
            hdr.append(f"## {subsection_id}")
        if hdr:
            content = "\n".join(hdr) + "\n\n" + content

    raw_id = doc.get("id", "")
    full_id = f"{id_prefix}___{raw_id}" if id_prefix else raw_id

    result = {
        "id": sanitize_id(full_id),
        "content": content,
        "embedding": [],  # filled later
        "category": category,
        "sourcepage": sourcepage,
        "sourcefile": sourcefile,
        "storageUrl": doc.get("storageUrl", "") or "",
        "oids": [],
        "groups": [],
    }
    # Only include extended fields if the target index has them
    if index_has_field("parent_id"):
        result["parent_id"] = ""
    if index_has_field("subsection_id"):
        result["subsection_id"] = subsection_id
    if index_has_field("subsections"):
        result["subsections"] = subsections
    if index_has_field("updated"):
        result["updated"] = doc.get("updated", "") or ""
    return result


# ---------------------------------------------------------------------------
# Index field detection
# ---------------------------------------------------------------------------
_INDEX_FIELDS: set[str] | None = None


def load_index_fields(endpoint: str, index_name: str):
    """Fetch the field names from the deployed index schema."""
    global _INDEX_FIELDS
    from azure.identity import DefaultAzureCredential
    from azure.search.documents.indexes import SearchIndexClient

    idx_client = SearchIndexClient(endpoint=endpoint, credential=DefaultAzureCredential())
    index = idx_client.get_index(index_name)
    _INDEX_FIELDS = {f.name for f in index.fields}
    logger.info("Index fields: %s", sorted(_INDEX_FIELDS))


def index_has_field(name: str) -> bool:
    if _INDEX_FIELDS is None:
        return False  # conservative: omit unknown fields
    return name in _INDEX_FIELDS


# ---------------------------------------------------------------------------
# Azure clients
# ---------------------------------------------------------------------------

def get_search_client(endpoint: str, index: str):
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient

    return SearchClient(
        endpoint=endpoint,
        index_name=index,
        credential=DefaultAzureCredential(),
    )


def get_openai_client():
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AzureOpenAI

    service = os.environ.get("AZURE_OPENAI_SERVICE", "")
    if not service:
        raise RuntimeError("AZURE_OPENAI_SERVICE not set")
    ep = f"https://{service}.openai.azure.com" if not service.startswith("https://") else service

    key = os.environ.get("AZURE_OPENAI_KEY", "")
    if key:
        return AzureOpenAI(
            api_key=key,
            api_version="2023-05-15",
            azure_endpoint=ep,
            max_retries=3,
            timeout=120.0,
        )

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version="2023-05-15",
        azure_endpoint=ep,
        max_retries=3,
        timeout=120.0,
    )


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def load_guide_docs(guide_name: str, meta: dict) -> list[dict]:
    path = INPUT_DIR / meta["file"]
    if not path.exists():
        logger.warning("File not found: %s", path)
        return []
    with open(path) as f:
        raw_docs = json.load(f)
    # Create a short prefix from the sourcefile to prevent cross-guide ID collisions
    id_prefix = sanitize_id(meta["sourcefile"])
    mapped = [map_doc(d, id_prefix=id_prefix) for d in raw_docs]
    logger.info("  %s: loaded %d docs from %s", guide_name, len(mapped), meta["file"])
    return mapped


def query_existing_ids(client, sourcefiles: set[str]) -> dict[str, set[str]]:
    """Query the index for all docs matching any of the court-guide sourcefiles.
    Returns {sourcefile: {id, ...}}."""
    result: dict[str, set[str]] = {}
    for sf in sorted(sourcefiles):
        ids: set[str] = set()
        try:
            # OData filter — exact match on sourcefile
            safe_sf = sf.replace("'", "''")
            results = client.search(
                search_text="*",
                filter=f"sourcefile eq '{safe_sf}'",
                select=["id"],
                top=5000,
            )
            for doc in results:
                ids.add(doc["id"])
        except Exception as e:
            logger.error("  Error querying sourcefile '%s': %s", sf, e)
        if ids:
            result[sf] = ids
            logger.info("  Index has %d docs for sourcefile='%s'", len(ids), sf)
    return result


def delete_docs(client, ids_to_delete: list[str], dry_run: bool) -> int:
    if not ids_to_delete:
        return 0
    if dry_run:
        logger.info("  [DRY RUN] Would delete %d documents", len(ids_to_delete))
        return 0
    deleted = 0
    batch_size = 1000
    for i in range(0, len(ids_to_delete), batch_size):
        batch = ids_to_delete[i : i + batch_size]
        payload = [{"id": doc_id} for doc_id in batch]
        try:
            client.delete_documents(documents=payload)
            deleted += len(batch)
            logger.info("  Deleted batch %d (%d docs)", i // batch_size + 1, len(batch))
        except Exception as e:
            logger.error("  Delete error: %s", e)
    return deleted


def generate_embeddings(docs: list[dict], dry_run: bool) -> list[dict]:
    """Generate embeddings for all docs. Modifies in-place and returns docs."""
    if dry_run:
        logger.info("  [DRY RUN] Would generate embeddings for %d docs", len(docs))
        return docs

    from openai import RateLimitError, APIConnectionError, APIError
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

    client = get_openai_client()
    deployment = os.environ.get("AZURE_OPENAI_EMB_DEPLOYMENT", "text-embedding-3-large")

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
    )
    def _embed(texts):
        return client.embeddings.create(input=texts, model=deployment)

    batch_size = 50  # conservative to avoid token limits
    success = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        texts = [d["content"].replace("\n", " ")[:8000] for d in batch]
        try:
            resp = _embed(texts)
            for j, data in enumerate(resp.data):
                batch[j]["embedding"] = data.embedding
            success += len(batch)
            logger.info(
                "  Embeddings: %d / %d (%.0f%%)",
                success,
                len(docs),
                success / len(docs) * 100,
            )
            if i + batch_size < len(docs):
                time.sleep(0.5)
        except Exception as e:
            logger.error("  Embedding batch %d failed: %s", i // batch_size + 1, e)
            raise
    return docs


def upload_docs(client, docs: list[dict], dry_run: bool) -> int:
    if dry_run:
        logger.info("  [DRY RUN] Would upload %d documents", len(docs))
        return 0
    uploaded = 0
    failed = 0
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        try:
            results = client.upload_documents(documents=batch)
            for r in results:
                if r.succeeded:
                    uploaded += 1
                else:
                    failed += 1
                    logger.error("  Upload failed for %s: %s", r.key, r.error_message)
            logger.info(
                "  Uploaded batch %d (%d ok, %d failed)",
                i // batch_size + 1,
                uploaded,
                failed,
            )
            if i + batch_size < len(docs):
                time.sleep(0.3)
        except Exception as e:
            logger.error("  Upload batch error: %s", e)
            failed += len(batch)
    return uploaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Upload court guides to index v3")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without making changes")
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Skip a guide by name (can repeat). E.g. --skip 'Court of Appeal Civil Division'",
    )
    args = parser.parse_args()

    load_azd_env()

    search_service = os.environ.get("AZURE_SEARCH_SERVICE", "")
    index_name = os.environ.get("AZURE_SEARCH_INDEX", "legal-court-rag-index-v3")
    if not search_service:
        logger.error("AZURE_SEARCH_SERVICE not set. Run `azd env` or configure .env")
        return 1

    endpoint = f"https://{search_service}.search.windows.net"
    logger.info("Target: %s / %s", endpoint, index_name)
    if args.dry_run:
        logger.info("*** DRY RUN — no changes will be made ***")

    # Load index schema so we only send fields that exist
    load_index_fields(endpoint, index_name)

    # ── 1. Load new docs ────────────────────────────────────────────────
    logger.info("── Step 1: Loading new docs from %s", INPUT_DIR)
    all_new_docs: list[dict] = []
    new_ids: set[str] = set()
    guides_processed = []

    for guide_name, meta in GUIDE_FILES.items():
        if guide_name in args.skip:
            logger.info("  Skipping %s (--skip)", guide_name)
            continue
        docs = load_guide_docs(guide_name, meta)
        if not docs:
            continue
        all_new_docs.extend(docs)
        new_ids.update(d["id"] for d in docs)
        guides_processed.append(guide_name)

    logger.info("Total new docs: %d across %d guides", len(all_new_docs), len(guides_processed))

    # ── 2. Query existing docs in index ─────────────────────────────────
    logger.info("── Step 2: Querying index for existing court-guide docs")
    client = get_search_client(endpoint, index_name)
    existing_by_sf = query_existing_ids(client, OLD_SOURCEFILES)

    all_existing_ids: set[str] = set()
    for ids in existing_by_sf.values():
        all_existing_ids.update(ids)

    orphan_ids = sorted(all_existing_ids - new_ids)
    overlap_ids = all_existing_ids & new_ids
    logger.info(
        "Existing: %d | Overlap (will update): %d | Orphans (will delete): %d",
        len(all_existing_ids),
        len(overlap_ids),
        len(orphan_ids),
    )

    # ── 3. Delete orphans ───────────────────────────────────────────────
    logger.info("── Step 3: Deleting %d orphaned documents", len(orphan_ids))
    deleted = delete_docs(client, orphan_ids, args.dry_run)
    if not args.dry_run and deleted:
        logger.info("Deleted %d orphaned docs. Waiting 2s for index propagation...", deleted)
        time.sleep(2)

    # ── 4. Generate embeddings ──────────────────────────────────────────
    logger.info("── Step 4: Generating embeddings for %d docs", len(all_new_docs))
    all_new_docs = generate_embeddings(all_new_docs, args.dry_run)

    # ── 5. Upload ───────────────────────────────────────────────────────
    logger.info("── Step 5: Uploading %d docs to index", len(all_new_docs))
    uploaded = upload_docs(client, all_new_docs, args.dry_run)

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  UPLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info("  Guides processed: %s", ", ".join(guides_processed))
    logger.info("  Docs loaded:      %d", len(all_new_docs))
    if args.dry_run:
        logger.info("  Would delete:     %d orphans", len(orphan_ids))
        logger.info("  Would upload:     %d docs", len(all_new_docs))
        logger.info("  *** DRY RUN — no changes were made ***")
    else:
        logger.info("  Deleted:          %d orphans", deleted)
        logger.info("  Uploaded:         %d docs", uploaded)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
