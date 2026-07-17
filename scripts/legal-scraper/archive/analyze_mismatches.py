#!/usr/bin/env python3
"""Categorize all mismatches to understand what fixes are needed."""
import json
import re
from collections import defaultdict

with open("data/legal-scraper/processed/section_review_against_html.json") as f:
    d = json.load(f)

pages = [p for p in d["pages"] if p["status"] == "mismatch"]

# Categorize mismatches
categories = defaultdict(list)

for p in pages:
    slug = p["url"].split("/")[-1]
    missing = p["missing_in_extracted"]
    extra = p["unexpected_in_extracted"]
    tier = p.get("tier", "?")

    if extra and not missing:
        # Extra sections in extracted that HTML doesn't have
        # These are things the chunks have that the extractor doesn't find in HTML
        for e in extra:
            if re.match(r"^(PART|PRACTICE DIRECTION|CYFARWYDDYD|NON-DISCLOSURE|Pre-Action Protocol|Pre-action protocol).*", e):
                categories["tier3_doc_title"].append((slug, e))
            elif re.match(r"^\d+$", e):
                categories["bare_number"].append((slug, e))
            elif re.match(r"^\d+[A-Z]?\.\d+[A-Z]?$", e):
                categories["rule_not_in_html"].append((slug, e))
            elif re.match(r"^[IVX]+$", e):
                categories["roman_not_in_html"].append((slug, e))
            elif re.match(r"^(Annex|Appendix|Schedule|Table|APPENDIX|TABLE|Footnotes|TEMPLATES)", e):
                categories["structural_label"].append((slug, e))
            elif re.match(r"^[A-Z][a-z]", e):
                categories["heading_word"].append((slug, e))
            else:
                categories["other_extra"].append((slug, e))

    elif missing and not extra:
        # Sections found in HTML but not in extracted chunks
        for m in missing:
            if re.match(r"^\d+[A-Z]?\.\d+", m):
                categories["rule_missing_from_chunks"].append((slug, m))
            elif re.match(r"^[IVX]+$", m):
                categories["roman_missing"].append((slug, m))
            elif re.match(r"^(annex|Annex|appendix|Appendix|Schedule)", m, re.I):
                categories["annex_missing"].append((slug, m))
            elif re.match(r"^\d+$", m):
                categories["num_missing"].append((slug, m))
            else:
                categories["other_missing"].append((slug, m))
    else:
        # Both missing and extra
        categories["both_missing_and_extra"].append({
            "slug": slug,
            "missing": missing[:5],
            "extra": extra[:5],
            "tier": tier
        })

print("=" * 70)
print("MISMATCH CATEGORY ANALYSIS")
print("=" * 70)
for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
    print(f"\n{cat} ({len(items)} items):")
    for item in items[:8]:
        print(f"  {item}")
    if len(items) > 8:
        print(f"  ... and {len(items) - 8} more")

# Also show a full summary of tier-3 doc titles
print("\n" + "=" * 70)
print("TIER-3 DOC TITLE EXTRAS (these are subsection_ids for tier-3 pages):")
print("=" * 70)
tier3_pages = [p for p in d["pages"] if p.get("tier") == 3]
print(f"Total tier-3 pages: {len(tier3_pages)}")
print(f"Tier-3 mismatches: {sum(1 for p in tier3_pages if p['status'] == 'mismatch')}")
print(f"Tier-3 matches: {sum(1 for p in tier3_pages if p['status'] == 'match')}")

# Check tier-3 matches — how come they match?
for p in tier3_pages:
    if p["status"] == "match":
        print(f"  MATCH tier3: {p['url'].split('/')[-1]} | html_sections={p['html_section_count']} extracted={p['extracted_section_count']}")
    else:
        print(f"  MISMATCH tier3: {p['url'].split('/')[-1]} | extra={p['unexpected_in_extracted'][:2]} missing={p['missing_in_extracted'][:2]}")

# Show what missing-only pages look like
print("\n" + "=" * 70)
print("MISSING-ONLY PAGES (HTML has sections chunks don't reference):")
print("=" * 70)
missing_only_pages = [p for p in pages if p["missing_in_extracted"] and not p["unexpected_in_extracted"]]
for p in missing_only_pages:
    slug = p["url"].split("/")[-1]
    print(f"  {slug} tier={p.get('tier')} html={p['html_section_count']} extracted={p['extracted_section_count']} missing={p['missing_in_extracted'][:6]}")
