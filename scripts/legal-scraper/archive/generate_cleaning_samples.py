#!/usr/bin/env python
"""
Generate diverse before/after cleaning samples for visual inspection.
Picks 6 representative CPR documents covering different content types.
"""
import sys, json, re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "app" / "backend"))

from content_cleaner import clean_content

UPLOAD_DIR = PROJECT_ROOT / "data" / "legal-scraper" / "processed" / "Upload"

# Pick 6 diverse samples
SAMPLE_FILES = [
    # 1. Simple single-chunk CPR Part (short, with breadcrumbs + markdown)
    "Part_1___Overriding_Objective.json",
    # 2. CPR Part with multi-subsections (medium length, numbered rules)
    "Part_15___Defence_And_Reply.json",
    # 3. Multi-chunk CPR Part (chunk_001 — has chunk headers)
    "Part_44___General_Rules_About_Costs_chunk_001.json",
    # 4. Practice Direction (different breadcrumb prefix)
    "Practice_Direction_3E___Costs_Management.json",
    # 5. Practice Direction with sub-sections
    "Practice_Direction_31A___Disclosure_And_Inspection.json",
    # 6. Annex / form-style content (if exists, else another PD)
    "Part_35___Experts_And_Assessors.json",
]

samples = []
for filename in SAMPLE_FILES:
    filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        # Try to find a close match
        matches = list(UPLOAD_DIR.glob(filename.replace(".json", "*.json")))
        if matches:
            filepath = matches[0]
            filename = filepath.name
        else:
            continue

    with open(filepath) as f:
        doc = json.load(f)

    content = doc.get("content", "")
    cleaned = clean_content(content)

    samples.append({
        "filename": filename,
        "category": doc.get("category", ""),
        "sourcepage": doc.get("sourcepage", ""),
        "sourcefile": doc.get("sourcefile", ""),
        "subsection_id": doc.get("subsection_id", ""),
        "original_content": content,
        "cleaned_content": cleaned,
        "original_chars": len(content),
        "cleaned_chars": len(cleaned),
        "reduction_pct": round((1 - len(cleaned) / len(content)) * 100, 1) if content else 0,
    })

# Write output
output_path = PROJECT_ROOT / "data" / "legal-scraper" / "processed" / "cleaning_samples.json"
with open(output_path, "w") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print(f"Generated {len(samples)} samples → {output_path}")
for s in samples:
    print(f"  {s['filename'][:60]:62s} {s['original_chars']:6d} → {s['cleaned_chars']:6d} chars ({s['reduction_pct']:4.1f}% reduction)")
