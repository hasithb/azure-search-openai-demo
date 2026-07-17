#!/usr/bin/env python3
"""
Generate full V3 corrected index JSON using html_section_extractor.

Reads all 215 Upload JSONs, fetches live HTML for each unique URL once,
applies extract_sections_for_chunk() per chunk, and writes:
  data/legal-scraper/processed/v3_full_corrected.json

Run from project root:
  python3 scripts/legal-scraper/generate_v3_full.py
"""

import json
import glob
import sys
import time
import traceback
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import requests

# Add scripts dir to path for extractor import
sys.path.insert(0, str(Path(__file__).parent))
from html_section_extractor import extract_sections_for_chunk

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data/legal-scraper/processed/Upload"
OUTPUT_FILE = PROJECT_ROOT / "data/legal-scraper/processed/v3_full_corrected.json"
HTML_CACHE_DIR = PROJECT_ROOT / "data/legal-scraper/processed/html_cache"
FETCH_DELAY = 0.3          # seconds between HTTP requests (be polite)
REQUEST_TIMEOUT = 20       # seconds per request
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def cache_path_for_url(url: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", url.split("/")[-1])[:60]
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    return HTML_CACHE_DIR / f"{slug}_{h}.html"

# ── Load all Upload JSONs ────────────────────────────────────────────────────
print("Loading Upload JSONs...")
all_files = sorted(f for f in UPLOAD_DIR.glob("*.json") if not f.name.endswith(".md5"))
print(f"  Found {len(all_files)} document chunks")

docs = []
for fpath in all_files:
    with open(fpath, encoding="utf-8") as fh:
        docs.append(json.load(fh))

# ── Discover unique URLs ─────────────────────────────────────────────────────
url_to_docs: dict[str, list[dict]] = defaultdict(list)
for d in docs:
    url_to_docs[d["storageUrl"]].append(d)

unique_urls = sorted(url_to_docs.keys())
print(f"  {len(unique_urls)} unique URLs to fetch")

# ── Fetch HTML for each URL ──────────────────────────────────────────────────
print("\nFetching HTML pages...")
url_to_html: dict[str, str | None] = {}
fetch_errors = 0
cache_hits = 0
HTML_CACHE_DIR.mkdir(parents=True, exist_ok=True)

for i, url in enumerate(unique_urls, 1):
    cache_file = cache_path_for_url(url)
    if cache_file.exists():
        try:
            url_to_html[url] = cache_file.read_text(encoding="utf-8")
            cache_hits += 1
            status = "✓ cache"
            print(f"  [{i:3d}/{len(unique_urls)}] {url.split('/')[-1][:60]}  {status}")
            continue
        except Exception:
            pass

    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        url_to_html[url] = resp.text
        try:
            cache_file.write_text(resp.text, encoding="utf-8")
        except Exception:
            pass
        status = "✓"
    except Exception as e:
        url_to_html[url] = None
        fetch_errors += 1
        status = f"✗ {e!r}"

    print(f"  [{i:3d}/{len(unique_urls)}] {url.split('/')[-1][:60]}  {status}")
    time.sleep(FETCH_DELAY)

print(
    f"\n  HTML ready: {len(unique_urls) - fetch_errors}/{len(unique_urls)}"
    f"  (cache hits: {cache_hits}, fetch errors: {fetch_errors})"
)

# ── Apply extractor to each chunk ────────────────────────────────────────────
print("\nExtracting sections for each chunk...")
results = []
tier_counts = defaultdict(int)
error_count = 0

for doc in docs:
    url = doc["storageUrl"]
    html = url_to_html.get(url)
    chunk_text = doc.get("content", "")

    subsection_id = None
    subsections: list[str] = []

    if html:
        try:
            subsection_id, subsections = extract_sections_for_chunk(html, chunk_text)
        except Exception as e:
            error_count += 1
            print(f"  ERROR on {doc['id']!r}: {e}")
            traceback.print_exc()

    # Build corrected document (all original fields preserved, new fields added)
    corrected = dict(doc)              # copy all original fields
    corrected["subsection_id"] = subsection_id or "-"
    corrected["subsections"] = subsections

    results.append(corrected)

print(f"  Processed {len(results)} chunks  Extraction errors: {error_count}")

# ── Analysis ─────────────────────────────────────────────────────────────────
print("\n── Analysis ────────────────────────────────────────────────────────────")
has_sub_id   = sum(1 for r in results if r["subsection_id"] not in ("-", None, ""))
has_subsecs  = sum(1 for r in results if r["subsections"])
multi_subsec = sum(1 for r in results if len(r["subsections"]) > 1)
no_sub_id    = sum(1 for r in results if r["subsection_id"] in ("-", None, ""))

print(f"  Total chunks:               {len(results)}")
print(f"  With subsection_id:         {has_sub_id} ({has_sub_id/len(results)*100:.1f}%)")
print(f"  Without subsection_id:      {no_sub_id}")
print(f"  With subsections list:      {has_subsecs} ({has_subsecs/len(results)*100:.1f}%)")
print(f"  Multi-section subsections:  {multi_subsec}")

# Show sample of resolved and unresolved
print("\n  Sample RESOLVED chunks:")
for r in results:
    if r["subsection_id"] not in ("-", None, ""):
        print(f"    {r['id']!r}")
        print(f"      subsection_id: {r['subsection_id']!r}")
        print(f"      subsections:   {r['subsections'][:5]}")
        break

print("\n  Sample UNRESOLVED chunks (subsection_id='-'):")
unresolved_samples = [r for r in results if r["subsection_id"] in ("-", None, "")]
for r in unresolved_samples[:3]:
    url_short = r["storageUrl"].split("/")[-1]
    print(f"    {r['id']!r}  url={url_short!r}")

# ── Write output ─────────────────────────────────────────────────────────────
print(f"\nWriting {len(results)} documents to {OUTPUT_FILE} ...")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
    json.dump(results, fh, ensure_ascii=False, indent=2)

print(f"Done. Output: {OUTPUT_FILE}")
print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
