#!/usr/bin/env python3
"""
Comprehensive audit of HTML anchor structure across ALL CPR source pages.

Fetches every unique storageUrl from the Upload JSON files and analyzes:
1. Whether <a id="..."> anchors exist inside headings
2. What format the anchor IDs use (e.g., "1.1", "rule44.1", "sectionI", etc.)
3. Whether the page uses wp-block-heading CSS classes
4. What heading structure exists (h1, h2, h3, h4)
5. Any pages that have NO meaningful anchors

Output: JSON report + console summary
"""

import json
import glob
import os
import re
import sys
import time
from collections import Counter, defaultdict
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "legal-scraper", "processed", "Upload")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "legal-scraper", "processed")


def collect_urls():
    """Collect all unique storageUrl values from Upload JSON files."""
    files = sorted(glob.glob(os.path.join(UPLOAD_DIR, "*.json")))
    files = [f for f in files if not f.endswith(".md5")]

    url_map = {}  # url -> list of filenames
    for f in files:
        with open(f) as fp:
            doc = json.load(fp)
        url = doc.get("storageUrl", "")
        fname = os.path.basename(f)
        if url:
            if url not in url_map:
                url_map[url] = []
            url_map[url].append(fname)
    return url_map


def classify_anchor_id(anchor_id):
    """Classify an anchor ID into a category."""
    if not anchor_id:
        return "empty"

    # Footnote anchors
    if re.match(r'^fn\d+$', anchor_id):
        return "footnote"

    # Text anchors
    if re.match(r'^text\d+$', anchor_id):
        return "text_anchor"

    # Rule-prefixed: rule44.1, rule44.2
    if re.match(r'^rule\d+\.\d+', anchor_id):
        return "rule_prefixed"

    # Section Roman: sectionI, sectionII
    if re.match(r'^section[IVX]+$', anchor_id):
        return "section_roman"

    # Clean dotted rule number: 1.1, 29.3, 2A.1
    if re.match(r'^\d+[A-Z]?\.\d+', anchor_id):
        return "dotted_rule"

    # Single number: 1, 2, 3
    if re.match(r'^\d+$', anchor_id):
        return "single_number"

    # Named sections: Annex, Schedule
    if re.match(r'^[A-Z][a-z]+', anchor_id):
        return "named_section"

    # Legacy auto-generated IDs
    if re.match(r'^ID[A-Z0-9]+$', anchor_id):
        return "legacy_autogen"

    # Paragraph references: para1.1
    if re.match(r'^para', anchor_id):
        return "para_prefixed"

    return "other"


def analyze_page(url, html_content):
    """Analyze a single page's HTML structure."""
    soup = BeautifulSoup(html_content, 'html.parser')

    result = {
        "url": url,
        "status": "ok",
        "title": "",
        "has_wp_block_headings": False,
        "heading_counts": {"h1": 0, "h2": 0, "h3": 0, "h4": 0},
        "total_anchors_in_headings": 0,
        "total_anchors_anywhere": 0,
        "anchor_ids": [],
        "anchor_categories": {},
        "content_div_found": None,
        "sample_heading_html": [],
    }

    # Find title
    h1 = soup.find('h1')
    if h1:
        result["title"] = h1.get_text(strip=True)[:100]

    # Check for wp-block-heading class
    wp_headings = soup.find_all(class_='wp-block-heading')
    result["has_wp_block_headings"] = len(wp_headings) > 0

    # Find content div
    content_div = None
    for selector in ['article-content', 'content', 'main']:
        content_div = soup.find('div', class_=selector) or soup.find(selector)
        if content_div:
            result["content_div_found"] = selector
            break
    if not content_div:
        content_div = soup.find('body') or soup
        result["content_div_found"] = "body_fallback"

    # Count headings
    for tag in ['h1', 'h2', 'h3', 'h4']:
        result["heading_counts"][tag] = len(content_div.find_all(tag))

    # Find ALL <a> tags with id attribute
    all_anchors = content_div.find_all('a', id=True)
    result["total_anchors_anywhere"] = len(all_anchors)

    # Find anchors specifically inside headings
    anchors_in_headings = []
    for tag in ['h1', 'h2', 'h3', 'h4']:
        for heading in content_div.find_all(tag):
            for a in heading.find_all('a', id=True):
                anchors_in_headings.append({
                    "id": a['id'],
                    "heading_tag": tag,
                    "heading_text": heading.get_text(strip=True)[:80],
                    "category": classify_anchor_id(a['id']),
                })

    result["total_anchors_in_headings"] = len(anchors_in_headings)
    result["anchor_ids"] = [a["id"] for a in anchors_in_headings]

    # Categorize
    cats = Counter(a["category"] for a in anchors_in_headings)
    result["anchor_categories"] = dict(cats)

    # Also check for anchors NOT in headings (standalone <a id="...">)
    standalone_anchors = []
    for a in all_anchors:
        parent = a.parent
        if parent and parent.name not in ['h1', 'h2', 'h3', 'h4']:
            standalone_anchors.append({
                "id": a['id'],
                "parent_tag": parent.name if parent else "none",
                "category": classify_anchor_id(a['id']),
            })

    result["standalone_anchor_count"] = len(standalone_anchors)
    result["standalone_categories"] = dict(Counter(a["category"] for a in standalone_anchors))

    # Also check for <a name="..."> (older HTML pattern)
    name_anchors = content_div.find_all('a', attrs={'name': True})
    name_only = [a for a in name_anchors if not a.get('id')]
    result["name_only_anchors"] = len(name_only)

    # Sample first 3 heading HTML
    for tag in ['h2', 'h3']:
        for heading in content_div.find_all(tag)[:3]:
            result["sample_heading_html"].append(str(heading)[:200])

    return result


def main():
    url_map = collect_urls()
    print(f"Collected {len(url_map)} unique URLs from Upload JSONs")

    results = []
    errors = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    for i, (url, fnames) in enumerate(sorted(url_map.items())):
        short_url = url.replace("https://www.justice.gov.uk/courts/procedure-rules/civil/rules/", "")
        print(f"[{i+1}/{len(url_map)}] {short_url} ...", end=" ", flush=True)

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}")
                errors.append({"url": url, "files": fnames, "error": f"HTTP {resp.status_code}"})
                results.append({
                    "url": url,
                    "files": fnames,
                    "status": f"http_{resp.status_code}",
                })
                continue

            page_result = analyze_page(url, resp.text)
            page_result["files"] = fnames
            results.append(page_result)

            in_h = page_result["total_anchors_in_headings"]
            anywhere = page_result["total_anchors_anywhere"]
            standalone = page_result["standalone_anchor_count"]
            cats = page_result["anchor_categories"]
            print(f"anchors: {in_h} in headings, {standalone} standalone, {anywhere} total | cats: {cats}")

        except Exception as e:
            print(f"ERROR: {e}")
            errors.append({"url": url, "files": fnames, "error": str(e)})
            results.append({
                "url": url,
                "files": fnames,
                "status": "error",
                "error": str(e),
            })

        # Be polite
        time.sleep(0.3)

    # Save full results
    output_path = os.path.join(OUTPUT_DIR, "html_anchor_audit.json")
    with open(output_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\nFull results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results)
    ok = [r for r in results if r.get("status") == "ok"]
    http_errors = [r for r in results if r.get("status", "").startswith("http_")]
    other_errors = [r for r in results if r.get("status") == "error"]

    print(f"\nTotal URLs: {total}")
    print(f"  Successfully fetched: {len(ok)}")
    print(f"  HTTP errors: {len(http_errors)}")
    print(f"  Other errors: {len(other_errors)}")

    # Pages with anchors in headings
    has_heading_anchors = [r for r in ok if r["total_anchors_in_headings"] > 0]
    no_heading_anchors = [r for r in ok if r["total_anchors_in_headings"] == 0]

    print(f"\nPages with anchors in headings: {len(has_heading_anchors)} / {len(ok)}")
    print(f"Pages WITHOUT anchors in headings: {len(no_heading_anchors)} / {len(ok)}")

    # Pages with standalone anchors
    has_standalone = [r for r in ok if r.get("standalone_anchor_count", 0) > 0]
    print(f"Pages with standalone anchors (not in headings): {len(has_standalone)} / {len(ok)}")

    # Pages with name-only anchors
    has_name_only = [r for r in ok if r.get("name_only_anchors", 0) > 0]
    print(f"Pages with <a name=''> only (no id): {len(has_name_only)} / {len(ok)}")

    # Aggregate anchor categories
    print(f"\nAnchor category breakdown (in headings):")
    all_cats = Counter()
    for r in ok:
        for cat, count in r.get("anchor_categories", {}).items():
            all_cats[cat] += count
    for cat, count in all_cats.most_common():
        print(f"  {cat}: {count}")

    print(f"\nStandalone anchor categories:")
    sa_cats = Counter()
    for r in ok:
        for cat, count in r.get("standalone_categories", {}).items():
            sa_cats[cat] += count
    for cat, count in sa_cats.most_common():
        print(f"  {cat}: {count}")

    # wp-block-heading usage
    has_wp = [r for r in ok if r.get("has_wp_block_headings")]
    print(f"\nPages with wp-block-heading CSS class: {len(has_wp)} / {len(ok)}")

    # List pages WITHOUT heading anchors
    if no_heading_anchors:
        print(f"\n--- Pages WITHOUT anchors in headings ({len(no_heading_anchors)}) ---")
        for r in no_heading_anchors:
            short = r["url"].replace("https://www.justice.gov.uk/courts/procedure-rules/civil/rules/", "")
            standalone = r.get("standalone_anchor_count", 0)
            name_only = r.get("name_only_anchors", 0)
            h_counts = r.get("heading_counts", {})
            print(f"  {short}")
            print(f"    title: {r.get('title', 'N/A')[:60]}")
            print(f"    headings: {h_counts}")
            print(f"    standalone anchors: {standalone}, name-only: {name_only}")
            print(f"    files: {r.get('files', [])}")

    # List HTTP error pages
    if http_errors:
        print(f"\n--- HTTP Error Pages ({len(http_errors)}) ---")
        for r in http_errors:
            short = r["url"].replace("https://www.justice.gov.uk/courts/procedure-rules/civil/rules/", "")
            print(f"  {short} -> {r['status']} | files: {r.get('files', [])}")

    # List pages with ONLY legacy autogen IDs
    legacy_only = [r for r in ok
                   if r["total_anchors_in_headings"] > 0
                   and set(r.get("anchor_categories", {}).keys()) == {"legacy_autogen"}]
    if legacy_only:
        print(f"\n--- Pages with ONLY legacy auto-generated IDs ({len(legacy_only)}) ---")
        for r in legacy_only:
            short = r["url"].replace("https://www.justice.gov.uk/courts/procedure-rules/civil/rules/", "")
            print(f"  {short} ({r['total_anchors_in_headings']} anchors)")
            print(f"    sample IDs: {r['anchor_ids'][:5]}")

    print(f"\n{'=' * 80}")
    print("AUDIT COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
