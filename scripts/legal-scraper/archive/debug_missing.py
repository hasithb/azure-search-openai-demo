#!/usr/bin/env python3
"""Quick check: Why are sections missing from chunk text?"""
import json, re
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
V3 = ROOT / "data/legal-scraper/processed/v3_full_corrected.json"

with open(V3) as f:
    docs = json.load(f)

# Check 'prot_hou' (missing 8.1, 9.1, 10.1, 11.1, 12.1)
url_slug = "prot_hou"
chunks = [d for d in docs if url_slug in d["storageUrl"]]
print(f"=== {url_slug}: {len(chunks)} chunks ===")
for c in chunks:
    content = c["content"][:300]
    print(f"\n  ID: {c['id']}")
    print(f"  subsection_id: {c['subsection_id']}")
    print(f"  subsections: {c.get('subsections', [])}")
    # Check if 8.1 appears in text
    for rule in ["8.1", "9.1", "10.1", "11.1", "12.1"]:
        if re.search(rf"\b{re.escape(rule)}\b", c["content"]):
            print(f"  >>> FOUND {rule} in text!")
    print(f"  content[:200]: {content[:200]}")

# Also check pre-action-protocol with 44 HTML sections, 4 extracted
print("\n\n=== low-value PI protocol ===")
chunks2 = [d for d in docs if "low-value-personal-injury-employers" in d["storageUrl"]]
print(f"Chunks: {len(chunks2)}")
for c in chunks2:
    print(f"\n  ID: {c['id']}")
    print(f"  subsection_id: {c['subsection_id']}")
    print(f"  subsections: {c.get('subsections', [])}")
    # Check for any dotted rules
    found = set(re.findall(r"\b(\d+\.\d+)\b", c["content"]))
    print(f"  dotted rules in text: {sorted(found)[:15]}")

# Also check part-44 (missing 2.2, 2.3, 2.9, 2.10, 2.11, 2.12 etc.)
print("\n\n=== part-44-general-rules-about-costs2 ===")
chunks3 = [d for d in docs if "part-44-general-rules-about-costs2" in d["storageUrl"]]
print(f"Chunks: {len(chunks3)}")
for c in chunks3:
    print(f"\n  ID: {c['id']}")
    print(f"  subsection_id: {c['subsection_id']}")
    print(f"  subsections: {c.get('subsections', [])}")
    for rule in ["2.2", "2.3", "2.9", "2.10", "2.11", "2.12"]:
        if re.search(rf"\b{re.escape(rule)}\b", c["content"]):
            print(f"  >>> FOUND {rule} in text!")
