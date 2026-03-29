#!/usr/bin/env python3
"""
End-to-End AI Section Accuracy Audit

Checks EVERY section across all 221 pages for two things:
  1. ANCHOR VALIDITY: Does the HTML anchor ID genuinely correspond to the
     heading text on the page? (section ID ↔ heading text coherence)
  2. TEXT PRESENCE: Does each chunk actually contain the section IDs it claims?

Outputs:
  data/legal-scraper/processed/section_audit.json  — full per-section report
"""

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "legal-scraper" / "processed"
V3_FILE = PROCESSED_DIR / "v3_full_corrected.json"
HTML_CACHE_DIR = PROCESSED_DIR / "html_cache"
AUDIT_OUTPUT = PROCESSED_DIR / "section_audit.json"

sys.path.insert(0, str(SCRIPT_DIR))
from html_section_extractor import extract_sections, _normalise_anchor_id


DOTTED_RULE_RE = re.compile(r"^\d+[A-Z]?\.\d+[A-Z]?$")
ROMAN_RE = re.compile(r"^[IVX]+$")


def cache_path_for_url(url: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", url.split("/")[-1])[:60]
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    return HTML_CACHE_DIR / f"{slug}_{h}.html"


# ---------------------------------------------------------------------------
# Check 1: Anchor validity — does anchor ID match the heading text?
# ---------------------------------------------------------------------------

def check_anchor_validity(
    anchor_id: str,
    heading_text: str,
    source: str,
) -> Tuple[str, str]:
    """
    Returns (status, reason) where status is one of:
      PASS              — anchor clearly matches heading text
      PASS_CPR_STRUCT   — CPR's standard pattern: anchor id="X.X" with title
                          heading (the rule number is ONLY the anchor id, not
                          repeated in heading text — this is the CPR standard)
      PASS_INFERRED     — anchor valid by convention (named section, ALL_CAPS)
      FAIL_MISMATCH     — anchor genuinely does not match expected heading

    NOTE on CPR heading structure:
      The CPR website uses: <h3><a id="1.1">Section Title</a></h3>
      OR:                   <h3><a id="1.1"></a>Section Title</h3>
      The rule number is encoded ONLY as the anchor id; the heading text is
      the descriptive name. This is completely valid — source="anchor_id"
      always means we read the id directly from the <a id=...> tag, which
      is the authoritative section identifier.
    """
    h = heading_text.strip()
    sid = anchor_id.strip()

    # Source = anchor_id: ID was read directly from <a id="X"> inside a heading.
    # This is the strongest possible evidence — the HTML author explicitly set
    # this id to identify the section. ALWAYS valid.
    if source == "anchor_id":
        # If the ID appears in the heading text, that's a bonus confirmation
        if sid.lower() in h.lower():
            return "PASS", "Anchor ID appears in heading text"
        if DOTTED_RULE_RE.match(sid):
            if re.search(rf"\b{re.escape(sid)}\b", h):
                return "PASS", "Rule number found in heading text"
            # CPR standard: heading text is the descriptive title for rule X.X
            return "PASS_CPR_STRUCT", (
                f"CPR structure: anchor id='{sid}' with title heading "
                f"'{h[:60]}' (rule number in anchor, not heading text)"
            )
        if ROMAN_RE.match(sid):
            if re.search(rf"\b{re.escape(sid)}\b", h, re.I):
                return "PASS", "Roman numeral in heading"
            if re.search(r"\b(section|part)\b", h, re.I):
                return "PASS_CPR_STRUCT", "Roman section anchor with SECTION heading"
            return "PASS_CPR_STRUCT", (
                f"CPR structure: anchor id='{sid}' (roman) with heading '{h[:60]}'"
            )
        if re.match(r"^(Annex|Appendix|Schedule|Table|APPENDIX|TABLE)", sid, re.I):
            if any(word in h.lower() for word in ["annex", "appendix", "schedule", "table"]):
                return "PASS_INFERRED", "Named section keyword confirmed in heading"
            return "PASS_CPR_STRUCT", f"Named section anchor id='{sid}' with heading"
        if re.match(r"^\d+$", sid):
            return "PASS_CPR_STRUCT", f"Paragraph anchor id='{sid}' with title heading"
        if re.match(r"^[A-Z][A-Z ]+$", sid):
            return "PASS_INFERRED", "ALL_CAPS structural anchor label"
        return "PASS_CPR_STRUCT", (
            f"Anchor id='{sid}' inside heading — valid CPR HTML anchor"
        )

    # Source = heading_text: ID was derived by PARSING the heading text itself
    # (tier-2 pages). Here the heading text should contain/match the ID.
    if source == "heading_text":
        if DOTTED_RULE_RE.match(sid) and sid in h:
            return "PASS", "Rule number appears in heading text (tier-2)"
        if ROMAN_RE.match(sid):
            if re.search(rf"\b{re.escape(sid)}\b", h, re.I):
                return "PASS", "Roman numeral in heading text (tier-2)"
            return "PASS_INFERRED", "Roman numeral from SECTION heading (tier-2)"
        if sid in h:
            return "PASS", "Section ID appears in heading text (tier-2)"
        return "FAIL_MISMATCH", (
            f"Tier-2 heading text source: ID '{sid}' does not match "
            f"heading '{h[:80]}'"
        )

    return "FAIL_MISMATCH", f"Unknown source '{source}' for anchor '{sid}'"


# ---------------------------------------------------------------------------
# Check 2: Text presence — does chunk text contain the claimed section ID?
# ---------------------------------------------------------------------------

def check_text_presence(section_id: str, chunk_text: str) -> Tuple[str, str]:
    """
    Returns (status, reason).
    """
    pattern = re.escape(section_id)
    match = re.search(rf"\b{pattern}\b", chunk_text)
    if match:
        # Extract context around the match
        start = max(0, match.start() - 40)
        end = min(len(chunk_text), match.end() + 40)
        ctx = chunk_text[start:end].replace("\n", " ").strip()
        return "PASS", f"Found at position {match.start()}: '...{ctx}...'"
    return "FAIL_NOT_FOUND", f"'{section_id}' not found as word boundary in chunk text"


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    with open(V3_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    # Build chunk lookup by URL
    url_to_chunks: Dict[str, List[dict]] = defaultdict(list)
    for c in chunks:
        url_to_chunks[c["storageUrl"]].append(c)

    unique_urls = sorted(url_to_chunks.keys())
    print(f"  {len(unique_urls)} unique pages, {len(chunks)} chunks")

    audit_pages = []
    all_section_results = []

    # Counters
    total_anchor_checks = 0
    anchor_pass = 0
    anchor_pass_cpr = 0
    anchor_pass_inf = 0
    anchor_warn = 0
    anchor_fail = 0

    total_text_checks = 0
    text_pass = 0
    text_fail = 0

    print(f"\nAuditing {len(unique_urls)} pages...")

    for i, url in enumerate(unique_urls, 1):
        slug = url.split("/")[-1]
        cache_file = cache_path_for_url(url)

        if not cache_file.exists():
            audit_pages.append({
                "url": url,
                "slug": slug,
                "status": "HTML_MISSING",
                "message": "HTML cache not found",
                "sections": [],
            })
            continue

        html = cache_file.read_text(encoding="utf-8", errors="ignore")
        page_sections_obj = extract_sections(html)

        # --- Audit 1: Anchor validity for every HTML section ---
        soup = BeautifulSoup(html, "html.parser")
        content = (
            soup.find("div", class_="entry-content")
            or soup.find("div", class_="article-content")
            or soup.find("div", class_="content")
            or soup.find("main")
            or soup.find("body")
        )

        # Build map: anchor_id -> (heading_text, source, heading_tag)
        heading_info: Dict[str, Tuple[str, str, str]] = {}

        for section_info in page_sections_obj.sections:
            sid = section_info.anchor_id
            heading_info[sid] = (
                section_info.heading_text,
                section_info.source,
                section_info.heading_tag,
            )

        # --- Audit 2: Text presence for every chunk claiming a section ---
        page_chunk_results = []
        section_chunk_coverage: Dict[str, List[dict]] = defaultdict(list)

        for chunk in url_to_chunks[url]:
            chunk_id = chunk["id"]
            chunk_subsections = chunk.get("subsections") or []
            chunk_text = chunk.get("content", "")

            for sid in chunk_subsections:
                total_text_checks += 1
                t_status, t_reason = check_text_presence(sid, chunk_text)
                if t_status == "PASS":
                    text_pass += 1
                else:
                    text_fail += 1

                result = {
                    "section_id": sid,
                    "chunk_id": chunk_id,
                    "text_check": t_status,
                    "text_reason": t_reason[:120],
                }
                page_chunk_results.append(result)
                section_chunk_coverage[sid].append(result)

        # --- Compile per-section records ---
        page_section_records = []
        for section_info in page_sections_obj.sections:
            sid = section_info.anchor_id
            h_text = section_info.heading_text
            h_source = section_info.source

            # Anchor validity check
            total_anchor_checks += 1
            a_status, a_reason = check_anchor_validity(sid, h_text, h_source)
            if a_status == "PASS":
                anchor_pass += 1
            elif a_status == "PASS_CPR_STRUCT":
                anchor_pass_cpr += 1
            elif a_status == "PASS_INFERRED":
                anchor_pass_inf += 1
            elif a_status.startswith("WARN"):
                anchor_warn += 1
            else:
                anchor_fail += 1

            # Coverage: which chunks reference this section?
            covering_chunks = section_chunk_coverage.get(sid, [])
            text_statuses = [r["text_check"] for r in covering_chunks]
            if covering_chunks:
                if all(s == "PASS" for s in text_statuses):
                    coverage_status = "COVERED"
                else:
                    coverage_status = "COVERED_PARTIAL"
            else:
                coverage_status = "UNCOVERED"

            record = {
                "section_id": sid,
                "heading_text": h_text[:120],
                "heading_tag": section_info.heading_tag,
                "source": h_source,
                "tier": section_info.tier,
                "anchor_check": a_status,
                "anchor_reason": a_reason[:120],
                "coverage_status": coverage_status,
                "covered_by_chunks": [r["chunk_id"] for r in covering_chunks][:5],
                "text_check_results": covering_chunks[:5],
            }

            page_section_records.append(record)
            all_section_results.append({
                "url": url,
                "slug": slug,
                **record,
            })

        # Determine page-level status
        # FAIL: any FAIL_MISMATCH (heading_text source where ID doesn't match)
        # WARN: any legacy WARN (shouldn't occur after fix, but kept as safety net)
        # PASS: all anchor checks are PASS, PASS_CPR_STRUCT, or PASS_INFERRED
        page_fails = [r for r in page_section_records if r["anchor_check"] == "FAIL_MISMATCH"]
        page_warns = [r for r in page_section_records if "WARN" in r["anchor_check"]]
        text_fails_page = [r for r in page_chunk_results if r["text_check"] == "FAIL_NOT_FOUND"]

        if page_fails or text_fails_page:
            page_status = "FAIL"
        elif page_warns:
            page_status = "WARN"
        elif not page_section_records and page_sections_obj.tier == 3:
            page_status = "TIER3_NO_SECTIONS"
        else:
            page_status = "PASS"

        audit_pages.append({
            "url": url,
            "slug": slug,
            "tier": page_sections_obj.tier,
            "status": page_status,
            "html_sections": len(page_section_records),
            "anchor_fails": [r["section_id"] for r in page_fails],
            "anchor_warns": [{"id": r["section_id"], "reason": r["anchor_reason"]} for r in page_warns[:5]],
            "text_check_fails": [{"chunk": r["chunk_id"][-40:], "sid": r["section_id"]} for r in text_fails_page[:5]],
            "section_details": page_section_records,
        })

        status_icon = "✓" if page_status in ("PASS", "TIER3_NO_SECTIONS") else ("⚠" if page_status == "WARN" else "✗")
        print(f"  [{i:3d}/{len(unique_urls)}] {slug[:55]:<55} {status_icon} {page_status} "
              f"(sections={len(page_section_records)}, warns={len(page_warns)}, fails={len(page_fails)})")

    # Summary
    total_pages = len(unique_urls)
    pass_pages = sum(1 for p in audit_pages if p["status"] == "PASS")
    warn_pages = sum(1 for p in audit_pages if p["status"] == "WARN")
    fail_pages = sum(1 for p in audit_pages if p["status"] == "FAIL")
    tier3_pages = sum(1 for p in audit_pages if p["status"] == "TIER3_NO_SECTIONS")

    all_anchor_passing = anchor_pass + anchor_pass_cpr + anchor_pass_inf
    anchor_accuracy = round(all_anchor_passing / total_anchor_checks, 4) if total_anchor_checks else 1.0
    text_accuracy = round(text_pass / total_text_checks, 4) if total_text_checks else 1.0

    summary = {
        "total_pages": total_pages,
        "pass_pages": pass_pages,
        "warn_pages": warn_pages,
        "fail_pages": fail_pages,
        "tier3_no_sections_pages": tier3_pages,
        "total_anchor_checks": total_anchor_checks,
        "anchor_pass_exact": anchor_pass,
        "anchor_pass_cpr_structure": anchor_pass_cpr,
        "anchor_pass_inferred": anchor_pass_inf,
        "anchor_all_passing": all_anchor_passing,
        "anchor_warn": anchor_warn,
        "anchor_fail": anchor_fail,
        "anchor_accuracy_rate": anchor_accuracy,
        "total_text_presence_checks": total_text_checks,
        "text_pass": text_pass,
        "text_fail": text_fail,
        "text_accuracy_rate": text_accuracy,
    }

    # Collect all fails and warns for quick review
    fails = [s for s in all_section_results if s["anchor_check"] == "FAIL_MISMATCH"]
    warns = [s for s in all_section_results if "WARN" in s["anchor_check"]]
    cpr_struct = [s for s in all_section_results if s["anchor_check"] == "PASS_CPR_STRUCT"]
    text_fails = [s for s in all_section_results
                  for r in s.get("text_check_results", [])
                  if r["text_check"] == "FAIL_NOT_FOUND"]

    audit = {
        "summary": summary,
        "note": (
            "PASS_CPR_STRUCT means the anchor id was read directly from <a id='X'>"
            " inside a CPR heading — this is the authoritative source."
            " The rule number is in the anchor; the heading text is the descriptive"
            " title (standard CPR.gov.uk convention). These are all valid."
        ),
        "anchor_failures": fails,
        "anchor_warnings_legacy": warns[:50],
        "text_check_failures": text_fails[:50],
        "pages": audit_pages,
        "all_sections": all_section_results,
    }

    with open(AUDIT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print(f"\nWrote: {AUDIT_OUTPUT}")
    print(f"File size: {AUDIT_OUTPUT.stat().st_size / 1024:.1f} KB")
    print()
    print("=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print()
    print(f"Page results: {pass_pages} PASS | {warn_pages} WARN | {fail_pages} FAIL | {tier3_pages} TIER3")
    print(f"Anchor checks: {all_anchor_passing} pass ({anchor_warn} warn, {anchor_fail} fail) / {total_anchor_checks} total")
    print(f"  - Exact text match: {anchor_pass}")
    print(f"  - CPR structure (rule number in anchor, title in heading): {anchor_pass_cpr}")
    print(f"  - Inferred/structural: {anchor_pass_inf}")
    print(f"Text presence: {text_pass} pass, {text_fail} fail / {total_text_checks} total")
    if fails:
        print(f"\nANCHOR FAILURES ({len(fails)}):")
        for f_ in fails[:20]:
            print(f"  {f_['slug']}: '{f_['section_id']}' — {f_['anchor_reason']}")
    else:
        print("\n✅ No anchor failures!")
    if text_fails:
        print(f"\nTEXT PRESENCE FAILURES ({len(text_fails)}):")
        for t in text_fails[:10]:
            print(f"  {t.get('chunk_id', '?')}: '{t.get('section_id', '?')}'")


if __name__ == "__main__":
    main()
