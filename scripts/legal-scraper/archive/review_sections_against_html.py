#!/usr/bin/env python3
"""
Build per-page section JSON from extracted chunks and review against cached HTML.

Accuracy is measured as: chunks NEVER claim sections that don't exist in HTML.
Coverage is measured as: what % of HTML sections are referenced by at least one chunk.

Outputs:
  - data/legal-scraper/processed/all_sections_by_url.json
  - data/legal-scraper/processed/section_review_against_html.json
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

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


def sort_sections(sections) -> List[str]:
    return sorted(sections, key=natural_sort_key)


def load_extracted_sections() -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """Load sections from chunks. Only includes HTML-verified section IDs."""
    with open(V3_FILE, encoding="utf-8") as f:
        docs = json.load(f)

    url_to_sections: Dict[str, Set[str]] = defaultdict(set)
    url_chunk_count: Dict[str, int] = defaultdict(int)

    for doc in docs:
        url = doc["storageUrl"]
        url_chunk_count[url] += 1
        for sec in doc.get("subsections") or []:
            if sec and sec != "-":
                url_to_sections[url].add(sec)
        # NOTE: We do NOT add subsection_id to the comparison set.
        # subsection_id is a contextual label (may be doc title for tier-3);
        # subsections is the HTML-verified list used for accuracy checking.

    return (
        {url: sort_sections(sections) for url, sections in url_to_sections.items()},
        url_chunk_count,
    )


def classification(value: str) -> str:
    if RULE_RE.match(value):
        return "rule"
    if PARA_RE.match(value):
        return "paragraph"
    if ROMAN_RE.match(value):
        return "roman"
    return "textual"


def main():
    extracted, chunk_counts = load_extracted_sections()

    # Include URLs that have chunks but 0 extracted sections (tier-3)
    with open(V3_FILE, encoding="utf-8") as f:
        docs = json.load(f)
    all_urls = sorted(set(d["storageUrl"] for d in docs))

    # Ensure all URLs are in extracted (even if empty)
    for url in all_urls:
        if url not in extracted:
            extracted[url] = []

    with open(SECTIONS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    pages = []
    html_missing = 0
    accurate_pages = 0    # No extras (invented sections)
    perfect_pages = 0     # No extras AND no missing
    total_html_sections = 0
    total_covered_sections = 0

    for url in sorted(all_urls):
        html_file = cache_path_for_url(url)
        extracted_set = set(extracted.get(url, []))

        if not html_file.exists():
            html_missing += 1
            pages.append({
                "url": url,
                "html_file": str(html_file.relative_to(PROJECT_ROOT)),
                "status": "html_missing",
                "extracted_sections": extracted.get(url, []),
                "extracted_count": len(extracted_set),
            })
            continue

        html = html_file.read_text(encoding="utf-8", errors="ignore")
        page_sections = extract_sections(html)
        html_set = set(page_sections.all_section_ids or [])

        # Accuracy: no invented sections (extras)
        extras = sort_sections(extracted_set - html_set)
        # Coverage: HTML sections found in chunks
        missing = sort_sections(html_set - extracted_set)
        covered = html_set & extracted_set

        is_accurate = len(extras) == 0
        is_perfect = is_accurate and len(missing) == 0

        if is_accurate:
            accurate_pages += 1
        if is_perfect:
            perfect_pages += 1

        total_html_sections += len(html_set)
        total_covered_sections += len(covered)

        # Status based on accuracy (no extras)
        if is_perfect:
            status = "perfect"
        elif is_accurate:
            status = "accurate"  # No extras, but some sections uncovered
        else:
            status = "inaccurate"  # Has invented sections

        overlap = len(covered)
        union = len(html_set | extracted_set)
        jaccard = (overlap / union) if union else 1.0
        coverage = (overlap / len(html_set)) if html_set else 1.0

        class_counts = defaultdict(int)
        for sec in extracted_set:
            class_counts[classification(sec)] += 1

        pages.append({
            "url": url,
            "html_file": str(html_file.relative_to(PROJECT_ROOT)),
            "status": status,
            "tier": page_sections.tier,
            "tier_reason": page_sections.tier_reason,
            "html_section_count": len(html_set),
            "extracted_section_count": len(extracted_set),
            "covered_count": overlap,
            "coverage": round(coverage, 4),
            "jaccard": round(jaccard, 4),
            "extras_invented": extras,
            "uncovered_in_html": missing,
            "chunks_for_url": chunk_counts.get(url, 0),
            "sample_extracted": extracted.get(url, [])[:20],
            "html_sections": sort_sections(html_set)[:30],
            "extracted_classification": dict(sorted(class_counts.items())),
        })

    compared = len(all_urls) - html_missing
    accuracy_rate = round(accurate_pages / compared, 4) if compared else 0.0
    perfect_rate = round(perfect_pages / compared, 4) if compared else 0.0
    coverage_rate = round(total_covered_sections / total_html_sections, 4) if total_html_sections else 1.0

    summary = {
        "urls_total": len(all_urls),
        "html_missing": html_missing,
        "pages_compared": compared,
        "accurate_pages": accurate_pages,
        "accuracy_rate": accuracy_rate,
        "perfect_pages": perfect_pages,
        "perfect_rate": perfect_rate,
        "total_html_sections": total_html_sections,
        "total_covered_sections": total_covered_sections,
        "section_coverage_rate": coverage_rate,
        "inaccurate_pages": compared - accurate_pages,
    }

    # Show inaccurate pages (with extras)
    inaccurate = [p for p in pages if p["status"] == "inaccurate"]

    review = {
        "summary": summary,
        "inaccurate_pages": [
            {
                "url": p["url"],
                "tier": p["tier"],
                "extras_invented": p["extras_invented"],
                "html_section_count": p["html_section_count"],
                "extracted_section_count": p["extracted_section_count"],
            }
            for p in inaccurate
        ],
        "coverage_gaps": [
            {
                "url": p["url"],
                "tier": p.get("tier"),
                "html_sections": p["html_section_count"],
                "covered": p["covered_count"],
                "coverage": p["coverage"],
                "uncovered": p["uncovered_in_html"][:10],
                "chunks": p["chunks_for_url"],
            }
            for p in pages
            if p.get("uncovered_in_html") and p["status"] != "html_missing"
        ][:50],
        "pages": pages,
    }

    with open(REVIEW_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {SECTIONS_OUTPUT}")
    print(f"Wrote: {REVIEW_OUTPUT}")
    print()
    print("=" * 60)
    print("SECTION EXTRACTION REVIEW")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print()
    if inaccurate:
        print(f"WARNING: {len(inaccurate)} pages have INVENTED sections (extras):")
        for p in inaccurate[:10]:
            slug = p["url"].split("/")[-1]
            print(f"  {slug}: {p['extras_invented'][:5]}")
    else:
        print("100% ACCURACY: No pages have invented sections!")
    print()
    print(f"Section coverage: {total_covered_sections}/{total_html_sections} "
          f"({coverage_rate*100:.1f}%) of HTML sections covered by chunks")


if __name__ == "__main__":
    main()
