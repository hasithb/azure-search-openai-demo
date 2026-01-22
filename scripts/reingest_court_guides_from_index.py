#!/usr/bin/env python3
"""Reingest court guide documents from the v2 Azure Search index.

Downloads court guide docs from the index, injects subsection headers into
content (no PDF re-parse), and updates documents back into the same index.

By default this preserves existing embeddings.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from typing import Any

from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient

from load_azd_env import load_azd_env
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from customizations.subsection_extractor import SubsectionExtractor


def has_existing_header(text: str) -> bool:
    if not text:
        return False
    head = [line.strip() for line in text.splitlines()[:6] if line.strip()]
    if any(line.startswith("SOURCE:") or line.startswith("SOURCEPAGE:") or line.startswith("SECTION:") for line in head):
        return True
    if any(line.startswith("[PART") or (line.startswith("[") and ">" in line) for line in head):
        return True
    return False


def extract_parent_section_from_sourcepage(value: str) -> str:
    if not value:
        return ""
    raw = value.strip()
    first_segment = raw.split(",", 1)[0].strip()
    if re.match(r"^[A-Z]\.", first_segment) or re.match(r"^(Section|Appendix|Part|Practice Direction)\b", first_segment, re.IGNORECASE):
        return first_segment

    patterns = [
        r"\b(Practice Direction\s+[0-9A-Z]+)\b",
        r"\b(Part\s+\d+[A-Z]?)\b",
        r"\b(Section\s+\d+)\b",
        r"\b(Appendix\s+[A-Z])\b",
        r"\b([A-Z]\.\s+[^,]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def extract_subsection_from_sourcepage(value: str) -> str:
    if not value:
        return ""
    raw = value.strip()
    patterns = [
        r"\b([A-Z]\.\d+(?:\.\d+)?)\b",  # C.2, F.1, A.1.1
        r"\b([A-Z]\d+\.\d+(?:\.\d+)?)\b",  # A4.1, B2.3
        r"\b(\d+\.\d+(?:\.\d+)?)\b",  # 8.4, 35.1
        r"\b([A-Z]\d+)\b",  # A1, B2
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def build_header(sourcefile: str, sourcepage: str, category: str, subsection_id: str, parent_section: str) -> str:
    header_lines = []
    if sourcefile:
        header_lines.append(f"SOURCE: {sourcefile}")
    if sourcepage:
        header_lines.append(f"SOURCEPAGE: {sourcepage}")
    if category:
        header_lines.append(f"CATEGORY: {category}")
    if parent_section and parent_section != subsection_id:
        header_lines.append(f"SECTION: {parent_section}")
    if subsection_id:
        header_lines.append(f"## {subsection_id}")
    return "\n".join(header_lines)


def filter_fieldnames(index_fields: set[str], doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k in index_fields}


def build_filter(scope: str) -> str:
    if scope == "all":
        return ""
    categories = [
        "Commercial Court",
        "Chancery Division",
        "Patents Court",
        "Technology and Construction Court",
        "King's Bench Division",
    ]
    parts = [f"category eq '{c.replace("'", "''")}'" for c in categories]
    return " or ".join(parts)


async def get_index_fields(endpoint: str, index_name: str, credential: DefaultAzureCredential) -> set[str]:
    async with SearchIndexClient(endpoint=endpoint, credential=credential) as index_client:
        index = await index_client.get_index(index_name)
        return {field.name for field in index.fields}


async def reingest(max_docs: int, dry_run: bool, batch_size: int, scope: str) -> None:
    load_azd_env()
    service_name = os.environ.get("AZURE_SEARCH_SERVICE")
    index_name = os.environ.get("AZURE_SEARCH_INDEX")

    if not service_name or not index_name:
        raise RuntimeError("AZURE_SEARCH_SERVICE or AZURE_SEARCH_INDEX not set")

    endpoint = f"https://{service_name}.search.windows.net"
    credential = DefaultAzureCredential()

    index_fields = await get_index_fields(endpoint, index_name, credential)

    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    updated = 0
    to_upload: list[dict[str, Any]] = []

    try:
        async with search_client:
            results = await search_client.search(
                search_text="*",
                filter=build_filter(scope) or None,
                top=max_docs,
            )

            async for doc in results:
                content = (doc.get("content") or "").strip()
                sourcepage = doc.get("sourcepage") or ""
                sourcefile = doc.get("sourcefile") or ""
                category = doc.get("category") or ""

                extracted_subsection = SubsectionExtractor.extract_first_subsection(content)
                derived_subsection = extract_subsection_from_sourcepage(sourcepage)
                parent_section = extract_parent_section_from_sourcepage(sourcepage)
                subsection_id = extracted_subsection or derived_subsection or parent_section or ""
                subsections = SubsectionExtractor.extract_all_subsections(content)
                if subsection_id and subsection_id not in subsections:
                    subsections.insert(0, subsection_id)

                new_content = content
                if content and not has_existing_header(content):
                    header = build_header(sourcefile, sourcepage, category, subsection_id, parent_section)
                    if header:
                        new_content = header + "\n\n" + content

                update_doc = {
                    "id": doc.get("id"),
                    "content": new_content,
                    "subsection_id": subsection_id,
                    "subsections": subsections,
                }

                update_doc = filter_fieldnames(index_fields, update_doc)

                if update_doc.get("content") != content or "subsection_id" in update_doc or "subsections" in update_doc:
                    to_upload.append(update_doc)
                    updated += 1

                if len(to_upload) >= batch_size:
                    if not dry_run:
                        await search_client.upload_documents(documents=to_upload)
                    to_upload = []

            if to_upload:
                if not dry_run:
                    await search_client.upload_documents(documents=to_upload)

    finally:
        await credential.close()

    print(f"Matched and updated documents: {updated}")
    print("Dry-run enabled, no changes written." if dry_run else "Reingest complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reingest court guides from index")
    parser.add_argument("--max-docs", type=int, default=1000000, help="Maximum documents to process")
    parser.add_argument("--batch-size", type=int, default=500, help="Upload batch size")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without uploading")
    parser.add_argument("--scope", choices=["court-guides", "all"], default="court-guides", help="Documents to reingest")
    args = parser.parse_args()

    asyncio.run(reingest(args.max_docs, args.dry_run, args.batch_size, args.scope))


if __name__ == "__main__":
    main()
