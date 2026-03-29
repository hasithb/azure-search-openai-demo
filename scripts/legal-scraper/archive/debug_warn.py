#!/usr/bin/env python3
"""Check warn samples and HTML structure to classify them."""
import json, re, hashlib
from pathlib import Path
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent.parent.parent
AUDIT = ROOT / "data/legal-scraper/processed/section_audit.json"
CACHE = ROOT / "data/legal-scraper/processed/html_cache"

with open(AUDIT) as f:
    audit = json.load(f)

warns = audit["anchor_warnings"][:5]
print("=== SAMPLE WARN CASES ===")
for w in warns:
    print(f"  Page: {w['slug']}")
    print(f"  Section: {w['section_id']}")
    print(f"  Heading: {w['heading_text']}")
    print(f"  Source: {w['source']}")
    print(f"  Reason: {w['anchor_reason']}")
    print()

# Load HTML for first WARN page
w = warns[0]
url = w["url"]
h = hashlib.md5(url.encode()).hexdigest()[:8]
clean = re.sub(r"[^a-zA-Z0-9_-]", "_", url.split("/")[-1])[:60]
cache_file = CACHE / f"{clean}_{h}.html"
if cache_file.exists():
    html = cache_file.read_text(errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    content = (
        soup.find("div", class_="entry-content")
        or soup.find("div", class_="article-content")
        or soup.find("main")
        or soup.find("body")
    )
    print(f"=== RAW HTML HEADINGS for {w['slug']} ===")
    for h_tag in content.find_all(["h2", "h3", "h4"])[:10]:
        print(f"  {h_tag.name}: {str(h_tag)[:250]}")

# Also check the warn reason breakdown
print()
print("=== WARN REASON CATEGORIES ===")
from collections import Counter
reasons = Counter()
for page in audit["pages"]:
    for sec in page.get("section_details", []):
        if "WARN" in sec.get("anchor_check", ""):
            r = sec.get("anchor_reason", "")
            key = r[:50]
            reasons[key] += 1
for r, c in reasons.most_common(10):
    print(f"  {c:4d}  {r}")
