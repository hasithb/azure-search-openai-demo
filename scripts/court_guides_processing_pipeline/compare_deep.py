#!/usr/bin/env python3
"""Deep structural comparison between old and new court guide data."""
import json

OLD_DIR = "data/legal-scraper/processed/Upload"
NEW_DIR = "scripts/court_guides_processing_pipeline/outputs_azure_di"

# Load Commercial Court as representative
old = json.load(open(f"{OLD_DIR}/14.341_JO_Commercial_Court_Guide_FINAL_processed.json"))
new = json.load(open(f"{NEW_DIR}/14.341_JO_Commercial_Court_Guide_FINAL_processed.json"))

print("=== OLD SAMPLE DOC (first) ===")
d = old[0]
for k in sorted(d.keys()):
    v = d[k]
    if k == "content":
        print(f"  {k}: ({len(v)} chars) {v[:150]!r}...")
    elif k == "embedding":
        print(f"  {k}: [{len(v)} floats]")
    else:
        print(f"  {k}: {v!r}")

print()
print("=== NEW SAMPLE DOC (first) ===")
d = new[0]
for k in sorted(d.keys()):
    v = d[k]
    if k == "content":
        print(f"  {k}: ({len(v)} chars) {v[:150]!r}...")
    elif k == "embedding":
        print(f"  {k}: [{len(v)} floats]")
    else:
        print(f"  {k}: {v!r}")

print()
print("=== OLD ID PATTERNS (first 5) ===")
for d in old[:5]:
    print(f"  id={d['id']!r}  sp={d['sourcepage']!r}")

print()
print("=== NEW ID PATTERNS (first 5) ===")
for d in new[:5]:
    print(f"  id={d['id']!r}  sp={d['sourcepage']!r}")

# All keys
old_all_keys = set()
for d in old:
    old_all_keys.update(d.keys())
new_all_keys = set()
for d in new:
    new_all_keys.update(d.keys())
print(f"\nOLD all keys: {sorted(old_all_keys)}")
print(f"NEW all keys: {sorted(new_all_keys)}")

# Sourcepage format differences
print("\n=== SOURCEPAGE FORMAT COMPARISON ===")
print("OLD sourcepages (first 10):")
for d in old[:10]:
    print(f"  {d['sourcepage']}")
print("NEW sourcepages (first 10):")
for d in new[:10]:
    print(f"  {d['sourcepage']}")

# Content length distribution
import statistics
old_lens = [len(d.get("content", "")) for d in old]
new_lens = [len(d.get("content", "")) for d in new]
print(f"\n=== CONTENT LENGTH STATS ===")
print(f"OLD: min={min(old_lens)} max={max(old_lens)} median={statistics.median(old_lens):.0f} stdev={statistics.stdev(old_lens):.0f}")
print(f"NEW: min={min(new_lens)} max={max(new_lens)} median={statistics.median(new_lens):.0f} stdev={statistics.stdev(new_lens):.0f}")

# Check for content overlap with fuzzy matching
# Normalize and find closest matching old/new pairs
from difflib import SequenceMatcher
print("\n=== CONTENT OVERLAP (sampling 5 old docs, finding best new match) ===")
import random
random.seed(42)
samples = random.sample(range(len(old)), min(5, len(old)))
for i in samples:
    old_content = old[i]["content"][:500]
    old_sp = old[i]["sourcepage"]
    best_ratio = 0
    best_new_sp = ""
    for nd in new:
        new_content = nd["content"][:500]
        ratio = SequenceMatcher(None, old_content, new_content).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_new_sp = nd["sourcepage"]
    print(f"  OLD: {old_sp[:70]}")
    print(f"  BEST NEW: {best_new_sp[:70]}  similarity={best_ratio:.1%}")
    print()

# Check King's Bench sourcefile difference
print("=== KING'S BENCH SOURCEFILE CHECK ===")
kb_old = json.load(open(f"{OLD_DIR}/35.16_JO_Kings_Bench_Division_Guide_2025_WEB4_processed.json"))
kb_new = json.load(open(f"{NEW_DIR}/35.16_JO_Kings_Bench_Division_Guide_2025_WEB4_processed.json"))
print(f"OLD sourcefile: {kb_old[0].get('sourcefile', 'MISSING')}")
print(f"NEW sourcefile: {kb_new[0].get('sourcefile', 'MISSING')}")

# Check NEW fields for storageUrl, updated
print("\n=== NEW DOC METADATA SAMPLE ===")
for guide_name, fname in [("Commercial", "14.341_JO_Commercial_Court_Guide_FINAL_processed.json"),
                           ("Court of Appeal", "35.67_JO_Court-of-Appeal-Civil-Division-Guide_FINAL_WEB_processed.json"),
                           ("SCCO", "Senior-Courts-Costs-Office-Guide_processed.json")]:
    docs = json.load(open(f"{NEW_DIR}/{fname}"))
    d = docs[0]
    print(f"\n  {guide_name}:")
    print(f"    storageUrl: {d.get('storageUrl', 'MISSING')}")
    print(f"    updated: {d.get('updated', 'MISSING')}")
    print(f"    parent_id: {d.get('parent_id', 'MISSING')}")
    print(f"    oids: {d.get('oids', 'MISSING')}")
    print(f"    groups: {d.get('groups', 'MISSING')}")
