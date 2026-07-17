#!/usr/bin/env python3
"""
Validate live justice.gov.uk HTML against the Azure AI Search index.

For each entry in ACTION_LIST (or a specified subset via --sections), the script:
  1. Scrapes the live page (same logic as update_cpr_index_v3.py)
  2. Runs verify_scrape_target (redirect / renamed-doc check)
  3. Checks boundary detection quality
  4. Queries the index for docs with matching sourcefile
  5. Compares live content vs indexed content:
       - char-count delta (volume)
       - key rule numbers present in live HTML but absent from indexed content
       - DC.date.modified mismatch
  6. Produces a report table + JSON artifact

Usage:
    python scripts/validate_html_vs_index.py               # all sections
    python scripts/validate_html_vs_index.py --sections A C  # specific sections
    python scripts/validate_html_vs_index.py --dry-run     # skip index queries
    python scripts/validate_html_vs_index.py --json-out /tmp/report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup (mirrors update_cpr_index_v3.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "app" / "backend"
SCRAPER_DIR = SCRIPT_DIR / "legal-scraper"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SCRAPER_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from load_azd_env import load_azd_env  # noqa: E402
import update_cpr_index_v3 as updater  # noqa: E402
from token_chunker import LegalDocumentChunker  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("html_vs_index")
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
VOLUME_DROP_PCT = 20   # flag if indexed content is <80% of live content chars
MAX_MISSING_RULES = 3  # flag if > N key rules from HTML are absent in index
CHUNKER = LegalDocumentChunker(max_tokens=8000, overlap_tokens=200)


# ---------------------------------------------------------------------------
# Index query helpers
# ---------------------------------------------------------------------------
def get_search_client(endpoint: str, index_name: str):
    from azure.search.documents import SearchClient
    from azure.identity import DefaultAzureCredential
    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=DefaultAzureCredential(),
    )


def fetch_indexed_docs(client, sourcefile: str) -> list[dict]:
    """Return all index docs whose sourcefile equals sourcefile."""
    try:
        escaped = sourcefile.replace("'", "''")
        results = client.search(
            search_text="*",
            filter=f"sourcefile eq '{escaped}'",
            select=["id", "sourcefile", "sourcepage", "content", "updated"],
            top=50,
        )
        return list(results)
    except Exception as e:
        logger.warning("Index query failed for %s: %s", sourcefile, e)
        return []


# ---------------------------------------------------------------------------
# Content comparison helpers
# ---------------------------------------------------------------------------

# Matches subsection headings like "## 2.1", "## 2.1(3)", "## 10.2A"
_SUBSECTION_HEADING_RE = re.compile(r"^#{1,4}\s+(\d+[A-Z]?\.\d+[A-Z]?(?:\([^)]*\))?)", re.MULTILINE)
# Bodies that are trivially short by design (omitted / reserved provisions)
_TRIVIAL_BODY_RE = re.compile(
    r"^\s*(?:omitted|reserved|not\s+used|\[omitted\]|\(omitted\)|revoked)\.?\s*$",
    re.IGNORECASE,
)


def extract_subsection_headings(text: str) -> list[str]:
    """Return unique subsection IDs (e.g. '2.1', '4.1(2)') found as ## headings."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in _SUBSECTION_HEADING_RE.finditer(text):
        label = m.group(1).strip()
        if label not in seen_set:
            seen_set.add(label)
            seen.append(label)
    return seen


def check_heading_coverage(
    live_content: str, indexed_docs: list[dict]
) -> dict:
    """
    Check that every subsection heading found in the live HTML appears in at
    least one indexed chunk for the same sourcefile.

    Returns a dict with:
      - live_headings: list of heading labels from the HTML
      - missing_headings: headings absent from ALL indexed chunks
      - stranded_headings: headings that sit alone at the end of a chunk
        (body of < 80 non-whitespace chars before the next chunk)
    """
    live_headings = extract_subsection_headings(live_content)

    # Build a single combined text and a per-chunk list for seam checks
    chunk_contents: list[str] = [d.get("content", "") for d in indexed_docs]
    all_indexed_text = "\n\n".join(chunk_contents)

    # ── Heading presence: does the heading number appear anywhere in the index? ──
    missing: list[str] = []
    for heading in live_headings:
        # Use word-boundary aware search so "2.1" doesn't match "12.1".
        # Lookahead excludes digits/letters only — allows trailing punctuation like
        # "19.1(1)." where the provision number is followed by a period in-text.
        escaped = re.escape(heading)
        pattern = rf"(?<![\d.A-Z]){escaped}(?![\dA-Z])"
        if not re.search(pattern, all_indexed_text):
            missing.append(heading)

    # ── Seam detection: last heading in a chunk with < 80 non-ws chars of body ──
    # Only flag where the body looks substantive but incomplete (excludes Omitted etc.)
    stranded: list[str] = []
    heading_body_re = re.compile(
        r"^(#{1,4}\s+\d+[A-Z]?\.\d+[A-Z]?(?:\([^)]*\))?.*)$", re.MULTILINE
    )
    for chunk in chunk_contents:
        positions = list(heading_body_re.finditer(chunk))
        # Only check the LAST heading in each chunk — that's where seams occur
        if not positions:
            continue
        last = positions[-1]
        body = chunk[last.end():]
        body_stripped = body.strip()
        # Skip trivially short provisions (Omitted, Reserved, etc.)
        if _TRIVIAL_BODY_RE.match(body_stripped):
            continue
        non_ws = len(re.sub(r"\s", "", body_stripped))
        # Flag only if the body after the last heading is very short (< 40 non-ws chars)
        # relative to the heading itself implying the content spilled into the next chunk
        if non_ws < 40:
            label = last.group(0).strip()[:60]
            stranded.append(label)

    return {
        "live_headings": live_headings,
        "missing_headings": missing,
        "stranded_headings": stranded,
    }


def extract_rule_numbers(text: str) -> set[str]:
    """Extract CPR rule numbers (e.g. 44.1, 44.2A) and PD para refs from text."""
    return set(re.findall(r"\b\d+[A-Z]?\.\d+[A-Z]?\b", text))


def compare_content(live_content: str, indexed_content: str) -> dict:
    """Return a comparison summary between live and indexed content."""
    live_chars = len(live_content)
    idx_chars = len(indexed_content)

    live_rules = extract_rule_numbers(live_content)
    idx_rules = extract_rule_numbers(indexed_content)
    missing_rules = sorted(live_rules - idx_rules)

    pct_coverage = (idx_chars / live_chars * 100) if live_chars else 100

    return {
        "live_chars": live_chars,
        "indexed_chars": idx_chars,
        "coverage_pct": round(pct_coverage, 1),
        "live_rules": sorted(live_rules),
        "indexed_rules": sorted(idx_rules),
        "missing_rules": missing_rules[:10],  # cap at 10
        "volume_flag": pct_coverage < (100 - VOLUME_DROP_PCT),
        "rules_flag": len(missing_rules) > MAX_MISSING_RULES,
    }


# ---------------------------------------------------------------------------
# Per-entry validation
# ---------------------------------------------------------------------------
def validate_entry(
    entry: dict,
    session,
    search_client,
    dry_run: bool,
) -> dict:
    sourcefile = entry["sourcefile"]
    url = entry["url"]

    result: dict = {
        "sourcefile": sourcefile,
        "url": url,
        "section": entry.get("section", "?"),
        "verify_ok": None,
        "verify_reason": "",
        "redirect_url": url,
        "h1": "",
        "live_chars": 0,
        "boundary_detection": "skipped",
        "boundary_count": 0,
        "indexed_doc_count": 0,
        "indexed_chars": 0,
        "coverage_pct": None,
        "missing_rules": [],
        "volume_flag": False,
        "rules_flag": False,
        "date_mismatch": False,
        "live_updated": "",
        "indexed_updated": "",
        "live_heading_count": 0,
        "missing_headings": [],
        "stranded_headings": [],
        "headings_flag": False,
        "issues": [],
        "status": "UNKNOWN",
    }

    if url == "DISCOVER_FROM_PROTOCOL_PAGE":
        logger.info("[%s] Skipping — URL requires discovery", sourcefile)
        result["status"] = "SKIPPED"
        return result

    # ── 1. Scrape live page ──────────────────────────────────────────────────
    logger.info("[%s] Fetching %s", sourcefile, url)
    scraped = updater.scrape_page(session, entry)
    if scraped is None:
        result["issues"].append("SCRAPE_FAILED: scrape_page returned None")
        result["status"] = "FAIL"
        return result

    final_url = scraped.get("_final_url", url)
    result["redirect_url"] = final_url
    result["h1"] = scraped.get("title", "")
    result["live_chars"] = len(scraped.get("content", ""))
    result["live_updated"] = scraped.get("updated", "")

    # ── 2. verify_scrape_target ──────────────────────────────────────────────
    ok, reason = updater.verify_scrape_target(entry, final_url, result["h1"])
    result["verify_ok"] = ok
    result["verify_reason"] = reason
    if not ok:
        result["issues"].append(f"VERIFY_FAIL: {reason}")

    # ── 3. Boundary detection ────────────────────────────────────────────────
    content = scraped.get("content", "")
    from token_chunker import LegalDocumentChunker
    chunker = LegalDocumentChunker(max_tokens=8000, overlap_tokens=200)
    token_count = chunker.count_tokens(content)
    if token_count <= chunker.max_tokens:
        result["boundary_detection"] = "not_chunked"
        result["boundary_count"] = 0
    else:
        boundaries = chunker.find_legal_boundaries(content)
        if boundaries:
            result["boundary_detection"] = "success"
            result["boundary_count"] = len(boundaries)
        else:
            result["boundary_detection"] = "fallback_sentence"
            result["boundary_count"] = 0
            result["issues"].append("BOUNDARY_FALLBACK: no legal boundaries found — will use sentence chunking")

    # ── 4. Index comparison (skipped in dry_run) ─────────────────────────────
    if dry_run or search_client is None:
        result["status"] = "OK" if ok else "FAIL"
        return result

    indexed_docs = fetch_indexed_docs(search_client, sourcefile)
    result["indexed_doc_count"] = len(indexed_docs)

    if not indexed_docs:
        result["issues"].append("INDEX_MISSING: no docs found in index for this sourcefile")
        result["status"] = "FAIL"
        return result

    # Concatenate all indexed content for this sourcefile
    indexed_combined = "\n\n".join(d.get("content", "") for d in indexed_docs)
    result["indexed_chars"] = len(indexed_combined)

    # Date comparison (use most recent indexed updated)
    updated_dates = [d.get("updated", "") for d in indexed_docs if d.get("updated")]
    if updated_dates:
        result["indexed_updated"] = max(updated_dates)

    # ── 5. Content comparison ────────────────────────────────────────────────
    cmp = compare_content(content, indexed_combined)
    result["coverage_pct"] = cmp["coverage_pct"]
    result["missing_rules"] = cmp["missing_rules"]
    result["volume_flag"] = cmp["volume_flag"]
    result["rules_flag"] = cmp["rules_flag"]

    if cmp["volume_flag"]:
        result["issues"].append(
            f"VOLUME_DROP: indexed {cmp['indexed_chars']} chars vs live {cmp['live_chars']} "
            f"({cmp['coverage_pct']}% coverage)"
        )
    if cmp["rules_flag"]:
        result["issues"].append(
            f"MISSING_RULES: {len(cmp['missing_rules'])} rule numbers in live HTML absent from index: "
            + ", ".join(cmp["missing_rules"][:5])
        )

    # ── 6. Subsection heading presence + seam check ───────────────────────────
    hcov = check_heading_coverage(content, indexed_docs)
    result["live_heading_count"] = len(hcov["live_headings"])
    result["missing_headings"] = hcov["missing_headings"][:20]
    result["stranded_headings"] = hcov["stranded_headings"][:20]
    result["headings_flag"] = len(hcov["missing_headings"]) > 0

    # MISSING_HEADINGS → real failure: heading in live HTML absent from all indexed chunks
    if hcov["missing_headings"]:
        result["issues"].append(
            f"MISSING_HEADINGS ({len(hcov['missing_headings'])}/{len(hcov['live_headings'])}): "
            + ", ".join(hcov["missing_headings"][:8])
        )
    # STRANDED_HEADINGS → informational warning only (does not cause FAIL)
    if hcov["stranded_headings"]:
        result["warnings"] = result.get("warnings", [])
        result["warnings"].append(
            f"STRANDED_HEADINGS ({len(hcov['stranded_headings'])}): "
            + "; ".join(hcov["stranded_headings"][:3])
        )

    # ── 7. Final status ───────────────────────────────────────────────────────
    if result["issues"] or not ok:
        result["status"] = "FAIL"
    else:
        result["status"] = "OK"

    return result


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------
COL_SF = 42
COL_ST = 6
COL_BD = 18
COL_CV = 9


def print_report(rows: list[dict]) -> None:
    total = len(rows)
    fails = sum(1 for r in rows if r["status"] == "FAIL")
    skipped = sum(1 for r in rows if r["status"] == "SKIPPED")
    ok = total - fails - skipped

    hr = "=" * 100
    print(f"\n{hr}")
    print(f"  HTML vs INDEX VALIDATION REPORT   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total: {total}   OK: {ok}   FAIL: {fails}   SKIPPED: {skipped}")
    print(hr)
    print(
        f"  {'SOURCEFILE':{COL_SF}}  {'STATUS':{COL_ST}}  {'BOUNDARY':{COL_BD}}  "
        f"{'COVERAGE':>{COL_CV}}  ISSUES"
    )
    print("-" * 100)

    for r in rows:
        sf = r["sourcefile"][:COL_SF]
        st = r["status"]
        bd = r["boundary_detection"]
        cv = f"{r['coverage_pct']}%" if r["coverage_pct"] is not None else "n/a"
        miss_h = len(r.get("missing_headings", []))
        warn_h = len(r.get("stranded_headings", []))
        hd_info = f" [{miss_h}miss]" if miss_h else (f" [{warn_h}warn]" if warn_h else "")
        issues_summary = "; ".join(r["issues"])[:40] if r["issues"] else ""
        print(
            f"  {sf:{COL_SF}}  {st:{COL_ST}}  {bd:{COL_BD}}  "
            f"{cv:>{COL_CV}}{hd_info}  {issues_summary}"
        )

    print(hr)

    # Detail section for failures
    failures = [r for r in rows if r["status"] == "FAIL"]
    if failures:
        print(f"\n  FAILURE DETAILS ({len(failures)} entries):")
        for r in failures:
            print(f"\n  ── {r['sourcefile']} ──")
            print(f"     URL:       {r['url']}")
            if r["redirect_url"] != r["url"]:
                print(f"     REDIRECTED: {r['redirect_url']}")
            print(f"     H1:        {r['h1']!r}")
            print(f"     Verify:    {r['verify_reason']}")
            print(f"     Boundary:  {r['boundary_detection']} (count={r['boundary_count']})")
            if r["coverage_pct"] is not None:
                print(f"     Coverage:  {r['coverage_pct']}% ({r['indexed_chars']} / {r['live_chars']} chars)")
            if r["missing_rules"]:
                print(f"     Missing rules: {', '.join(r['missing_rules'])}")
            if r.get("missing_headings"):
                print(f"     Missing headings ({len(r['missing_headings'])}/{r.get('live_heading_count','?')}): "
                      + ", ".join(r["missing_headings"][:10]))
            if r.get("stranded_headings"):
                print(f"     WARN stranded headings: " + "; ".join(r["stranded_headings"][:5]))
            for issue in r["issues"]:
                print(f"     ⚠  {issue}")
            for warn in r.get("warnings", []):
                print(f"     ~  {warn}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Validate live HTML vs Azure Search index")
    parser.add_argument("--sections", nargs="+", help="Sections to validate (A B C D DEBT). Default: all")
    parser.add_argument("--dry-run", action="store_true", help="Skip index queries, only check scrape quality")
    parser.add_argument("--json-out", type=Path, default=None, help="Write JSON report to this path")
    args = parser.parse_args()

    load_azd_env()

    # Build section filter
    sections_filter = set(args.sections) if args.sections else None

    entries = [
        e for e in updater.ACTION_LIST
        if sections_filter is None or e.get("section", "") in sections_filter
    ]

    logger.info("Validating %d ACTION_LIST entries (dry_run=%s)", len(entries), args.dry_run)

    # Set up search client
    search_client = None
    if not args.dry_run:
        search_svc = os.environ.get("AZURE_SEARCH_SERVICE", "")
        index_name = os.environ.get("AZURE_SEARCH_INDEX", "legal-court-rag-index-v3")
        if search_svc:
            endpoint = f"https://{search_svc}.search.windows.net"
            try:
                search_client = get_search_client(endpoint, index_name)
                logger.info("Connected to index: %s @ %s", index_name, endpoint)
            except Exception as e:
                logger.warning("Could not connect to index (%s) — running in dry-run mode", e)
        else:
            logger.warning("AZURE_SEARCH_SERVICE not set — running in dry-run mode")

    session = updater.make_session()
    rows: list[dict] = []

    for entry in entries:
        row = validate_entry(entry, session, search_client, dry_run=args.dry_run or search_client is None)
        rows.append(row)
        time.sleep(0.3)  # light throttle between entries

    print_report(rows)

    # Write JSON report
    out_path = args.json_out
    if out_path is None:
        reports_dir = PROJECT_ROOT / "data" / "legal-scraper" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = reports_dir / f"html_vs_index_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"entries": rows}, f, indent=2, ensure_ascii=False)
    logger.info("JSON report written to %s", out_path)

    fails = sum(1 for r in rows if r["status"] == "FAIL")
    return 1 if fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
