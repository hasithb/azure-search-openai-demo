#!/usr/bin/env python3
"""
Final verification: read final_all_sections.json and check for coverage gaps.
Group chunks by page URL, check if all page_all_sections are claimed by at least one chunk.
Also check for any invalid claims (subsections not in page_all_sections).
"""
import json
from pathlib import Path
from collections import defaultdict

FINAL = Path("data/legal-scraper/processed/final_all_sections.json")

docs = json.load(open(FINAL))
print(f"Loaded {len(docs)} records from final_all_sections.json\n")

# Group by storageUrl
by_url = defaultdict(list)
for d in docs:
    by_url[d["storageUrl"]].append(d)

total_unclaimed = 0
total_invalid   = 0

unclaimed_pages = []
invalid_pages   = []

for url, chunks in sorted(by_url.items()):
    page_secs = set(chunks[0].get("page_all_sections", []))
    all_claimed = set()
    for c in chunks:
        all_claimed.update(c.get("subsections", []))

    unclaimed = page_secs - all_claimed
    invalid   = all_claimed - page_secs  # claimed but not in page_all_sections

    if unclaimed:
        total_unclaimed += len(unclaimed)
        unclaimed_pages.append({
            "url": url.split("/")[-1],
            "chunks": len(chunks),
            "page_secs": len(page_secs),
            "unclaimed": sorted(unclaimed),
        })
    if invalid:
        total_invalid += len(invalid)
        invalid_pages.append({
            "url": url.split("/")[-1],
            "chunks": len(chunks),
            "invalid": sorted(invalid),
        })

print(f"=== UNCLAIMED SECTIONS (page sections claimed by no chunk) ===")
if unclaimed_pages:
    for p in unclaimed_pages:
        print(f"  {p['url']}: {p['unclaimed']}")
else:
    print("  NONE ✓")

print(f"\n=== INVALID CLAIMS (chunk claims section not in page_all_sections) ===")
if invalid_pages:
    for p in invalid_pages:
        print(f"  {p['url']}: {p['invalid']}")
else:
    print("  NONE ✓")

print(f"\n=== SUMMARY ===")
print(f"  Total unclaimed section slots: {total_unclaimed}")
print(f"  Total invalid claims: {total_invalid}")

# Also run the anomaly detector
print(f"\n=== ANOMALY CHECK (tier mismatch etc) ===")
tiered_empty = [(d['id'][:70], d['page_tier'], d.get('page_section_count',0)) for d in docs
                if d.get('page_tier') in (1,2) and d.get('page_section_count',0) > 0 and not d.get('subsections')]
print(f"  tier-1/2 chunks with >0 page sections but empty subsections: {len(tiered_empty)}")
for item in tiered_empty:
    print(f"    {item}")

tier_none = [d['id'][:70] for d in docs if d.get('page_tier') is None]
print(f"  tier=None chunks: {len(tier_none)}")
for t in tier_none:
    print(f"    {t}")
