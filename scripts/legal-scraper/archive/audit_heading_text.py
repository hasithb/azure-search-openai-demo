#!/usr/bin/env python3
"""Check heading TEXT on legacy/no-anchor pages for parseable rule numbers."""
import json
import re
import time
import requests
from bs4 import BeautifulSoup

DATA = "/Users/HasithB/Downloads/PROJECTS/azure-search-openai-demo-2/data/legal-scraper/processed/html_anchor_audit.json"

with open(DATA) as f:
    audit = json.load(f)

# Category C: standalone anchors + headings (many standalone, decent headings)
cat_c = [r for r in audit
         if r.get("status") == "ok"
         and r.get("total_anchors_in_headings", 0) == 0
         and r.get("standalone_anchor_count", 0) > 5
         and sum(r.get("heading_counts", {}).get(t, 0) for t in ["h2", "h3", "h4"]) > 3]

# Category B: no anchors + headings
cat_b = [r for r in audit
         if r.get("status") == "ok"
         and r.get("total_anchors_in_headings", 0) == 0
         and r.get("standalone_anchor_count", 0) == 0
         and sum(r.get("heading_counts", {}).get(t, 0) for t in ["h2", "h3", "h4"]) > 3]

headers = {"User-Agent": "Mozilla/5.0"}

samples = cat_c[:5] + cat_b[:5]
rule_re = re.compile(r"^\d+[A-Z]?\.\d+")
section_re = re.compile(r"^[IVX]+\s|^Section\s|^SECTION\s")

for r in samples:
    short = r["url"].replace("https://www.justice.gov.uk/courts/procedure-rules/civil/rules/", "")
    resp = requests.get(r["url"], headers=headers, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")
    content = (
        soup.find("div", class_="article-content")
        or soup.find("div", class_="content")
        or soup.find("main")
        or soup.find("body")
    )

    print(f"\n== {short} ==")
    rule_headings = []
    for tag in ["h2", "h3", "h4"]:
        for h in content.find_all(tag):
            text = h.get_text(strip=True)
            if rule_re.match(text):
                rule_headings.append((tag, text[:40]))
            elif section_re.match(text):
                rule_headings.append((tag, text[:40]))

    print(f"  Rule-number headings found: {len(rule_headings)}")
    for tag, text in rule_headings[:8]:
        print(f"    <{tag}> {text}")
    if len(rule_headings) > 8:
        print(f"    ... and {len(rule_headings) - 8} more")

    section_headings = [
        (h.name, h.get_text(strip=True)[:60])
        for h in content.find_all(["h2", "h3"])
        if not rule_re.match(h.get_text(strip=True))
    ]
    print(f"  Non-rule headings: {len(section_headings)}")
    for tag, text in section_headings[:5]:
        print(f"    <{tag}> {text}")

    time.sleep(0.3)

# Part 25 (h4 only) and Part 45 (h2 only) are special - check them too
print("\n\n== SPECIAL CASES ==")
special = [
    "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part25",
    "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part45-fixed-costs",
    "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-81-applications-and-proceedings-in-relation-to-contempt-of-court",
]
for url in special:
    short = url.replace("https://www.justice.gov.uk/courts/procedure-rules/civil/rules/", "")
    resp = requests.get(url, headers=headers, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")
    content = (
        soup.find("div", class_="article-content")
        or soup.find("div", class_="content")
        or soup.find("main")
        or soup.find("body")
    )

    print(f"\n== {short} ==")
    for tag in ["h2", "h3", "h4"]:
        hlist = content.find_all(tag)
        if hlist:
            print(f"  {tag}: {len(hlist)} headings")
            for h in hlist[:6]:
                text = h.get_text(strip=True)[:60]
                has_a = bool(h.find("a", id=True))
                print(f"    <{tag}> anchor={has_a} '{text}'")
            if len(hlist) > 6:
                print(f"    ... and {len(hlist) - 6} more")
    time.sleep(0.3)
