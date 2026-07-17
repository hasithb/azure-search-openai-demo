#!/usr/bin/env python3
"""
Produce the final reviewable JSON with ALL sections parsed from HTML.

Format: same as index, but content field has each \\n\\n-separated block
on its own line for easier review.

Each document in the output represents one chunk with:
- All original index fields preserved
- subsection_id and subsections from the fixed extractor
- content_lines: array where each element is a \\n\\n-separated block
- page_all_sections: ALL sections found in the HTML page (ground truth)
- page_tier: which extraction tier the page uses

Output:
  data/legal-scraper/processed/final_all_sections.json
"""

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "legal-scraper" / "processed"
V3_FILE = PROCESSED_DIR / "v3_full_corrected.json"
HTML_CACHE_DIR = PROCESSED_DIR / "html_cache"
OUTPUT_FILE = PROCESSED_DIR / "final_all_sections.json"

sys.path.insert(0, str(SCRIPT_DIR))
from html_section_extractor import extract_sections


def cache_path_for_url(url: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", url.split("/")[-1])[:60]
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    return HTML_CACHE_DIR / f"{slug}_{h}.html"


def natural_sort_key(value: str):
    parts = re.split(r"(\d+)", value)
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return key


def main():
    print("Loading v3_full_corrected.json...")
    with open(V3_FILE, encoding="utf-8") as f:
        docs = json.load(f)

    # Pre-compute page-level section info from HTML cache
    print("Extracting page-level sections from HTML cache...")
    url_page_info: dict = {}
    unique_urls = sorted(set(d["storageUrl"] for d in docs))

    for url in unique_urls:
        cache_file = cache_path_for_url(url)
        if cache_file.exists():
            html = cache_file.read_text(encoding="utf-8", errors="ignore")
            page_sections = extract_sections(html)
            url_page_info[url] = {
                "page_tier": page_sections.tier,
                "page_tier_reason": page_sections.tier_reason,
                "page_all_sections": sorted(page_sections.all_section_ids, key=natural_sort_key),
                "page_section_count": len(page_sections.all_section_ids),
            }
        else:
            url_page_info[url] = {
                "page_tier": None,
                "page_tier_reason": "html_not_cached",
                "page_all_sections": [],
                "page_section_count": 0,
            }

    # Build final output
    print("Building final JSON...")
    results = []

    for doc in docs:
        url = doc["storageUrl"]
        content = doc.get("content", "")

        # Split content by \n\n into lines for easier review
        content_lines = [block.strip() for block in content.split("\n\n") if block.strip()]

        # Build output record
        record = {
            "id": doc["id"],
            "category": doc.get("category", ""),
            "sourcepage": doc.get("sourcepage", ""),
            "sourcefile": doc.get("sourcefile", ""),
            "storageUrl": url,
            "subsection_id": doc.get("subsection_id", "-"),
            "subsections": doc.get("subsections", []),
            # Page-level ground truth from HTML
            "page_tier": url_page_info[url]["page_tier"],
            "page_all_sections": url_page_info[url]["page_all_sections"],
            "page_section_count": url_page_info[url]["page_section_count"],
            # Content split for review
            "content_lines": content_lines,
            "content_line_count": len(content_lines),
        }

        results.append(record)

    # Write output
    print(f"Writing {len(results)} records to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    file_size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"Done. File size: {file_size_kb:.1f} KB")

    # Summary stats
    total_sections = sum(r["page_section_count"] for r in results)
    tier_counts = defaultdict(int)
    for r in results:
        tier_counts[r["page_tier"]] += 1

    pages_with_sections = sum(1 for url in unique_urls if url_page_info[url]["page_section_count"] > 0)

    print(f"\nSummary:")
    print(f"  Total chunks: {len(results)}")
    print(f"  Unique pages: {len(unique_urls)}")
    print(f"  Pages with sections: {pages_with_sections}")
    print(f"  Pages without sections (tier-3): {len(unique_urls) - pages_with_sections}")
    print(f"  Chunks by tier: {dict(sorted(tier_counts.items()))}")
    print(f"  Avg content lines per chunk: {sum(r['content_line_count'] for r in results) / len(results):.1f}")


if __name__ == "__main__":
    main()
