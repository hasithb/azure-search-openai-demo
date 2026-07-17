#!/usr/bin/env python
"""
Generate V3 Index Document Preview

Produces 20+ diverse CPR documents showing the EXACT format that will be
uploaded to legal-court-rag-index-v3 — all index fields populated, content
cleaned, subsections extracted. No embeddings (replaced with placeholder).

Usage:
    python generate_v3_document_preview.py

Output:
    data/legal-scraper/processed/v3_document_preview.json
"""

import os
import sys
import json
import re
import hashlib
import glob

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(script_dir, '../../app/backend')
sys.path.insert(0, script_dir)
sys.path.insert(0, backend_dir)

from content_cleaner import clean_content
from customizations.subsection_extractor import SubsectionExtractor


# ── ID sanitizer (same as upload_with_embeddings.py) ──

def sanitize_id(doc_id: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9_\-=]', '_', doc_id)
    s = re.sub(r'_{2,}', '___', s)
    return s.strip('_')


# ── Parent / subsection extraction from sourcepage (same as upload_with_embeddings.py) ──

def extract_parent_section_from_sourcepage(value: str) -> str:
    if not value:
        return ""
    raw = value.strip()
    first_segment = raw.split(",", 1)[0].strip()
    if re.match(r"^[A-Z]\.", first_segment) or re.match(
        r"^(Section|Appendix|Part|Practice Direction)\b", first_segment, re.IGNORECASE
    ):
        return first_segment

    patterns = [
        r"\b(Practice Direction\s+[0-9A-Z]+)\b",
        r"\b(Part\s+\d+[A-Z]?)\b",
        r"\b(Section\s+\d+)\b",
        r"\b(Appendix\s+[A-Z])\b",
        r"\b([A-Z]\.\s+[^,]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def extract_subsection_from_sourcepage(value: str) -> str:
    if not value:
        return ""
    raw = value.strip()
    patterns = [
        r"\b([A-Z]\.\d+(?:\.\d+)?)\b",
        r"\b([A-Z]\d+\.\d+(?:\.\d+)?)\b",
        r"\b(\d+\.\d+(?:\.\d+)?)\b",
        r"\b([A-Z]\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def has_existing_header(text: str) -> bool:
    if not text:
        return False
    head = [line.strip() for line in text.splitlines()[:6] if line.strip()]
    if any(
        line.startswith("SOURCE:") or line.startswith("SOURCEPAGE:") or line.startswith("SECTION:")
        for line in head
    ):
        return True
    if any(line.startswith("[PART") or (line.startswith("[") and ">" in line) for line in head):
        return True
    return False


# ── V3 document builder ──

def build_v3_document(doc: dict) -> dict:
    """Build a V3 index document from a source Upload JSON.
    
    Pipeline:
    1. Clean content (remove metadata headers, breadcrumbs, markdown, chunk headers)
    2. Extract subsections from CLEANED content
    3. DO NOT prepend headers (V3 relies on index fields, not in-content headers)
    4. Return full schema
    """
    doc_id = doc.get("id", "")
    sanitized_id = sanitize_id(doc_id)

    # Handle content that might be a list
    content = doc.get("content", "")
    if isinstance(content, list):
        content = "\n".join(content)

    sourcepage = doc.get("sourcepage", "")
    sourcefile = doc.get("sourcefile", "")
    category = doc.get("category", "Legal Document")

    # ── V3 CHANGE: Clean content first ──
    cleaned_content = clean_content(content)

    # ── Extract subsections from CLEANED content ──
    extracted_subsection = SubsectionExtractor.extract_first_subsection(cleaned_content)
    extracted_subsections = SubsectionExtractor.extract_all_subsections(cleaned_content)

    derived_subsection = extract_subsection_from_sourcepage(sourcepage)
    parent_section = extract_parent_section_from_sourcepage(sourcepage)

    subsection_id = extracted_subsection or derived_subsection or parent_section or ""
    subsections = list(extracted_subsections)
    if subsection_id and subsection_id not in subsections:
        subsections.insert(0, subsection_id)

    # ── V3 CHANGE: Do NOT prepend SOURCE:/SOURCEPAGE: headers ──
    # (V2 prepended these; V3 removes them since they're in dedicated fields)

    return {
        "id": sanitized_id,
        "content": cleaned_content,
        "embedding": "[3072-dimensional vector — omitted for preview]",
        "category": category,
        "sourcepage": sourcepage,
        "sourcefile": sourcefile,
        "storageUrl": doc.get("storageUrl", ""),
        "oids": doc.get("oids", []) if doc.get("oids") else [],
        "groups": doc.get("groups", []) if doc.get("groups") else [],
        "parent_id": doc.get("parent_id", ""),
        "subsection_id": subsection_id,
        "subsections": subsections,
        "updated": doc.get("updated", ""),
    }


# ── Diverse sample selection ──

# Hand-picked files covering all document types / edge cases
DIVERSE_PICKS = [
    # -- Short single-chunk Parts --
    "Part_1___Overriding_Objective.json",               # Shortest Part, foundational CPR
    "Part_11___Disputing_The_Court_S_Jurisdiction.json", # Small Part
    "Part_15___Defence_And_Reply.json",                  # Small Part

    # -- Multi-chunk Parts (chunk_000 + chunk_001) --
    "Part_21___Children_And_Protected_Parties_chunk_000.json",
    "Part_21___Children_And_Protected_Parties_chunk_001.json",
    "Part_44___General_Rules_About_Costs_chunk_000.json",
    "Part_44___General_Rules_About_Costs_chunk_001.json",

    # -- Higher-numbered Parts --
    "Part_29___The_Multi-Track.json",
    "Part_35___Experts_And_Assessors.json",
    "Part_52___Appeals_chunk_001.json",

    # -- Practice Directions (various types) --
    "Practice_Direction_1A___Participation_Of_Vulnerable_Parties_Or_Witnesses.json",
    "Practice_Direction_22___Statements_Of_Truth.json",
    "Practice_Direction_31A___Disclosure_And_Inspection.json",
    "Practice_Direction_36___Offers_To_Settle.json",

    # -- Practice Direction with sub-direction letter --
    "Practice_Direction_2A___Court_Offices.json",
    "Practice_Direction_19B___Group_Litigation.json",

    # -- Practice Direction 27B (truncated filename edge case) --
    "Practice_Direction__27B___Claims_Under_The_Pre-Action_Protocol_For_Personal_Injury_Claims_Below_The_.json",

    # -- Long multi-chunk Practice Direction --
    "Practice_Direction_29___The_Multi-Track_chunk_001.json",

    # -- Welsh language documents --
    "Cyfarwyddyd_Ymarfer_54A___Adolygiad_Barnwrol_chunk_001.json",

    # -- Part 3 chunk_002 (high chunk number) --
    "Part_3___The_Court_S_Case_Management_Powers_chunk_002.json",

    # -- Large Part with costs focus --
    "Part_36___Offers_To_Settle_chunk_000.json",

    # -- Part with specialist list --
    "Part_49___Specialist_Proceedings.json",
]


def load_diverse_samples(upload_dir: str) -> list:
    """Load the diverse sample files."""
    samples = []
    available = {os.path.basename(f): f for f in glob.glob(os.path.join(upload_dir, "*.json"))}

    for filename in DIVERSE_PICKS:
        if filename in available:
            with open(available[filename], 'r', encoding='utf-8') as f:
                doc = json.load(f)
                samples.append({
                    "source_file": filename,
                    "document": doc,
                })
        else:
            print(f"⚠️  Not found: {filename}")

    # If any picks are missing, fill up to 22 from remaining files
    used = set(s["source_file"] for s in samples)
    remaining = sorted(set(available.keys()) - used)
    # Pick varied remaining files
    import random
    random.seed(42)
    extras = random.sample(remaining, min(max(0, 22 - len(samples)), len(remaining)))
    for filename in extras:
        with open(available[filename], 'r', encoding='utf-8') as f:
            doc = json.load(f)
            samples.append({
                "source_file": filename,
                "document": doc,
            })

    return samples


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    upload_dir = os.path.join(base, "../../data/legal-scraper/processed/Upload")
    output_path = os.path.join(base, "../../data/legal-scraper/processed/v3_document_preview.json")

    print(f"Loading diverse samples from: {upload_dir}")
    samples = load_diverse_samples(upload_dir)
    print(f"Loaded {len(samples)} diverse documents")

    results = []
    for i, sample in enumerate(samples):
        filename = sample["source_file"]
        doc = sample["document"]

        # Build V3 document
        v3_doc = build_v3_document(doc)

        # Calculate content stats
        original_content = doc.get("content", "")
        if isinstance(original_content, list):
            original_content = "\n".join(original_content)
        cleaned_len = len(v3_doc["content"])
        original_len = len(original_content)
        reduction = ((original_len - cleaned_len) / original_len * 100) if original_len else 0

        results.append({
            "_meta": {
                "sample_number": i + 1,
                "source_filename": filename,
                "original_content_chars": original_len,
                "cleaned_content_chars": cleaned_len,
                "reduction_percent": round(reduction, 1),
            },
            "index_document": v3_doc,
        })

        print(f"  [{i+1:2d}] {filename}")
        print(f"       id: {v3_doc['id']}")
        print(f"       subsection_id: {v3_doc['subsection_id']!r}")
        print(f"       subsections: {v3_doc['subsections'][:5]}{'...' if len(v3_doc['subsections']) > 5 else ''}")
        print(f"       content: {original_len} → {cleaned_len} chars ({reduction:.1f}% reduction)")

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Wrote {len(results)} V3 document previews to:")
    print(f"   {output_path}")
    print(f"\nEach entry has:")
    print(f"  _meta: stats about the transformation")
    print(f"  index_document: the EXACT document that will be uploaded to V3 (minus embedding vector)")


if __name__ == "__main__":
    main()
