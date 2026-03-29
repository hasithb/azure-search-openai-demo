#!/usr/bin/env python3
"""Check ID overlap between old and new court guide docs."""
import json

OLD_DIR = "data/legal-scraper/processed/Upload"
NEW_DIR = "scripts/court_guides_processing_pipeline/outputs_azure_di"

guides = [
    ("Commercial", "14.341_JO_Commercial_Court_Guide_FINAL_processed.json"),
    ("Kings Bench", "35.16_JO_Kings_Bench_Division_Guide_2025_WEB4_processed.json"),
    ("Chancery", "Chancery-Guide-2024-web_processed.json"),
    ("Patents", "Patents-Court-Guide-Updated-February-2025_processed.json"),
    ("TCC", "The-Technology-and-Construction-Court-Guide_processed.json"),
]

print("=== ID OVERLAP CHECK ===")
print("If old IDs don't match new IDs, uploading new docs won't replace old ones.\n")

total_orphaned = 0
for name, fname in guides:
    old = json.load(open(f"{OLD_DIR}/{fname}"))
    new = json.load(open(f"{NEW_DIR}/{fname}"))
    
    old_ids = set(d["id"] for d in old)
    new_ids = set(d["id"] for d in new)
    
    common = old_ids & new_ids
    only_old = old_ids - new_ids
    only_new = new_ids - old_ids
    total_orphaned += len(only_old)
    
    print(f"{name}:")
    print(f"  OLD IDs: {len(old_ids)}  NEW IDs: {len(new_ids)}")
    print(f"  Matching IDs: {len(common)}")
    print(f"  OLD-only IDs (ORPHANED): {len(only_old)}")
    print(f"  NEW-only IDs: {len(only_new)}")
    if common:
        samples = sorted(common)[:3]
        print(f"  Sample matching: {samples}")
    if only_old:
        samples = sorted(only_old)[:5]
        print(f"  Sample orphaned: {samples}")
    print()

print(f"=== TOTAL ORPHANED: {total_orphaned} documents would remain as duplicates ===")
print("These old docs would stay in the index alongside new ones unless explicitly deleted.")

# Check how push_court_guides.py handles this
print("\n=== NEW DOCS MISSING 'embedding' FIELD ===")
for name, fname in [("Commercial", "14.341_JO_Commercial_Court_Guide_FINAL_processed.json")]:
    new = json.load(open(f"{NEW_DIR}/{fname}"))
    has_emb = sum(1 for d in new if "embedding" in d)
    no_emb = sum(1 for d in new if "embedding" not in d)
    print(f"  {name}: {has_emb} with embedding, {no_emb} without embedding")
    print(f"  (Embeddings will need to be generated during upload)")
