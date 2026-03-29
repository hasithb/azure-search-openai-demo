#!/usr/bin/env python3
"""
Build per-page section JSON from extracted chunks and review it against cached HTML pages.

Outputs:
  - data/legal-scraper/processed/all_sections_by_url.json
  - data/legal-scraper/processed/section_review_against_html.json
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import sys

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "legal-scraper" / "processed"
V3_FILE = PROCESSED_DIR / "v3_full_corrected.json"
HTML_CACHE_DIR = PROCESSED_DIR / "html_cache"
SECTIONS_OUTPUT = PROCESSED_DIR / "all_sections_by_url.json"
REVIEW_OUTPUT = PROCESSED_DIR / "section_review_against_html.json"

sys.path.insert(0, str(SCRIPT_DIR))
from html_section_extractor import extract_sections


ROMAN_RE = re.compile(r"^[IVX]+$")
RULE_RE = re.compile(r"^\d+[A-Z]?\.\d+[A-Z]?$")
PARA_RE = re.compile(r"^\d+[A-Z]?$")


def cache_path_for_url(url: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", url.split("/")[-1])[:60]
    digest = hashlib.md5(url.encode()).hexdigest()[:8]
    return HTML_CACHE_DIR / f"{slug}_{digest}.html"


def natural_sort_key(value: str):
    parts = re.split(r"(\d+)", value)
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return key


def sort_sections(sections: Set[str]) -> List[str]:
    return sorted(sections, key=natural_sort_key)


def load_extracted_sections() -> Dict[str, List[str]]:
    with open(V3_FILE, encoding="utf-8") as f:
        docs = json.load(f)

    url_to_sections: Dict[str, Set[str]] = defaultdict(set)
    for doc in docs:
        url = doc["storageUrl"]
        for sec in doc.get("subsections") or []:
            if sec and sec != "-":
                url_to_sections[url].add(sec)
        sid = doc.get("subsection_id")
        if sid and sid != "-":
            url_to_sections[url].add(sid)

    return {url: sort_sections(sections) for url, sections in url_to_sections.items()}


def classification(value: str) -> str:
    if RULE_RE.match(value):
        return "rule"
    if PARA_RE.match(value):
        return "paragraph"
    if ROMAN_RE.match(value):
        return "roman"
    return "textual"


def main():
    extracted = load_extracted_sections()

    with open(SECTIONS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    pages = []
    html_missing = 0
    perfect_matches = 0

    for url in sorted(extracted.keys()):
        html_file = cache_path_for_url(url)
        extracted_set = set(extracted[url])

        if not html_file.exists():
            html_missing += 1
            pages.append(
                {
                    "url": url,
                    "html_file": str(html_file.relative_to(PROJECT_ROOT)),
                    "status": "html_missing",
                    "extracted_sections": extracted[url],
                    "extracted_count": len(extracted_set),
                }
            )
            continue

        html = html_file.read_text(encoding="utf-8", errors="ignore")
        page_sections = extract_sections(html)
        html_set = set(page_sections.all_section_ids or [])

        missing_in_extracted = sort_sections(html_set - extracted_set)
        unexpected_in_extracted = sort_sections(extracted_set - html_set)

        if not missing_in_extracted and not unexpected_in_extracted:
            status = "match"
            perfect_matches += 1
        else:
            status = "mismatch"

        overlap = len(html_set & extracted_set)
        union = len(html_set | extracted_set)
        jaccard = (overlap / union) if union else 1.0

        class_counts = defaultdict(int)
        for sec in extracted_set:
            class_counts[classification(sec)] += 1

        pages.append(
            {
                "url": url,
                "html_file": str(html_file.relative_to(PROJECT_ROOT)),
                "status": status,
                "tier": page_sections.tier,
                "tier_reason": page_sections.tier_reason,
                "html_section_count": len(html_set),
                "extracted_section_count": len(extracted_set),
                "intersection_count": overlap,
                "jaccard": round(jaccard, 4),
                "missing_in_extracted": missing_in_extracted,
                "unexpected_in_extracted": unexpected_in_extracted,
                "sample_extracted": extracted[url][:20],
                "extracted_classification": dict(sorted(class_counts.items())),
            }
        )

    mismatches = [p for p in pages if p["status"] == "mismatch"]
    summary = {
        "urls_total": len(extracted),
        "html_missing": html_missing,
        "pages_compared": len(extracted) - html_missing,
        "perfect_matches": perfect_matches,
        "mismatches": len(mismatches),
        "match_rate": round(perfect_matches / (len(extracted) - html_missing), 4)
        if (len(extracted) - html_missing)
        else 0.0,
    }

    review = {
        "summary": summary,
        "mismatch_examples": sorted(
            [
                {
                    "url": p["url"],
                    "jaccard": p["jaccard"],
                    "missing_in_extracted": p["missing_in_extracted"][:10],
                    "unexpected_in_extracted": p["unexpected_in_extracted"][:10],
                }
                for p in mismatches
            ],
            key=lambda x: x["jaccard"],
        )[:50],
        "pages": pages,
    }

    with open(REVIEW_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {SECTIONS_OUTPUT}")
    print(f"Wrote: {REVIEW_OUTPUT}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
