#!/usr/bin/env python3
"""Read-only source-to-index fidelity diagnostic for the legal RAG index.

This module is intentionally *additive* and *read-only*. It never queries,
uploads to, deletes from, re-embeds, or otherwise mutates Azure AI Search.
It reuses the existing audit primitives (normalization, n-gram extraction,
reconciliation) and writes a *separate* diagnostic report so the canonical
``reports/source_document_accuracy.*`` contract is left untouched.

Purpose
-------
Raw six-gram coverage is a triage signal, not a completeness target. A low
score can be caused by benign parser transformations (navigation removal,
breadcrumb injection, table flattening, whitespace/markdown normalization)
*or* by genuine loss of substantive legal text. This diagnostic separates the
two by reporting three coverage layers and the actual unmatched substantive
blocks, then proposes an evidence-based provisional cause instead of assuming
every low score means the source changed.

Layers
------
1. ``raw_coverage``          — existing normalized six-gram coverage.
2. ``substantive_coverage``  — coverage after dropping non-substantive source
                               blocks (navigation, breadcrumbs, ToC, running
                               headers, publication boilerplate, short
                               headings/fragments). This is the legally
                               meaningful number.
3. ``identifier_coverage``   — fraction of legal identifiers (rules, nested
                               paragraphs, PD/annex/schedule labels) present in
                               the index, with the ordered missing ranges.

Usage
-----
    python scripts/diagnose_source_fidelity.py \
        --index-snapshot reports/source_document_index_snapshot.json \
        --html-cache reports/source_document_http_cache.json \
        --json-output reports/source_fidelity_diagnostic.json

    # Sibling-chunk investigation only (no HTML fetch), for a single source:
    python scripts/diagnose_source_fidelity.py \
        --index-snapshot reports/source_document_index_snapshot.json \
        --source "Practice Direction 27B" --no-html
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import tiktoken

try:
    from scripts.audit_source_documents import (  # type: ignore[import-not-found]
        DEFAULT_HTML_CACHE,
        ROOT,
        CanonicalSource,
        HtmlAuditCache,
        compare_text_content,
        load_pdf_sources,
        load_web_scraper,
        load_web_sources,
        normalize_label,
        normalize_url,
        reconcile_sources,
        scrape_with_cache,
    )
    from scripts.court_guides_processing_pipeline.verify_extraction_completeness import (  # type: ignore[import-not-found]
        MIN_NGRAM_COVERAGE,
        classify_missing_block,
        extract_ngrams,
        normalize,
    )
except ModuleNotFoundError:
    from audit_source_documents import (  # type: ignore[import-not-found]
        DEFAULT_HTML_CACHE,
        ROOT,
        CanonicalSource,
        HtmlAuditCache,
        compare_text_content,
        load_pdf_sources,
        load_web_scraper,
        load_web_sources,
        normalize_label,
        normalize_url,
        reconcile_sources,
        scrape_with_cache,
    )
    from court_guides_processing_pipeline.verify_extraction_completeness import (  # type: ignore[import-not-found]
        MIN_NGRAM_COVERAGE,
        classify_missing_block,
        extract_ngrams,
        normalize,
    )

DEFAULT_DIAGNOSTIC_REPORT = ROOT / "reports" / "source_fidelity_diagnostic.json"
EMBEDDING_ENCODING = tiktoken.encoding_for_model("text-embedding-3-large")
EMBEDDING_MAX_TOKENS = 8191
HISTORICAL_CHARACTER_LIMIT = 8000

# Per-block substantive coverage threshold: a substantive source block whose
# six-grams are covered below this fraction is reported as unmatched evidence.
MIN_BLOCK_COVERAGE = 0.60

# Block classifications that are safe to exclude from substantive coverage.
# These mirror the documented, semantically-neutral transformations already
# performed by the production scraper and content_cleaner.
NON_SUBSTANTIVE_CLASSIFICATIONS = {
    "NAVIGATION",
    "BREADCRUMB_ONLY",
    "TOC_DOTS_LINE",
    "TOC_ENTRY",
    "TOC_WITH_PAGE_NUMBERS",
    "OGL_BOILERPLATE",
    "PUBLISHER_INFO",
    "BLANK_PAGE",
    "SHORT_HEADING",
    "SHORT_FRAGMENT",
    "RUNNING_HEADER",
}

# Leading breadcrumb prefix injected by the scraper, e.g. "[PART 3 > 3.1] text".
BREADCRUMB_PREFIX = re.compile(r"^\s*\[[^\]]*\]\s*")

# Navigation / site-furniture heuristics for live HTML text blocks.
NAVIGATION_MARKERS = (
    "skip to main content",
    "back to top",
    "you are here",
    "cookie",
    "print this page",
    "is this page useful",
    "related links",
    "breadcrumb",
)


def strip_breadcrumb(text: str) -> str:
    """Remove a single leading breadcrumb prefix from a scraped block."""
    return BREADCRUMB_PREFIX.sub("", text, count=1)


def split_blocks(text: str) -> list[str]:
    """Split content into paragraph-like blocks (double-newline separated)."""
    if not text:
        return []
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text)]
    return [block for block in blocks if block]


def classify_block(text: str) -> str:
    """Classify a single content block, adding HTML-specific categories.

    Falls back to the PDF pipeline's ``classify_missing_block`` for the shared
    categories (ToC, boilerplate, short heading/fragment, substantive).
    """
    lowered = text.lower()
    if any(marker in lowered for marker in NAVIGATION_MARKERS):
        return "NAVIGATION"

    stripped = strip_breadcrumb(text)
    if not normalize(stripped):
        # Nothing left after removing the breadcrumb: purely navigational.
        return "BREADCRUMB_ONLY"
    return classify_missing_block(stripped)


def extract_identifiers(text: str) -> list[str]:
    """Extract legal identifiers more thoroughly than a bare ``N.N`` regex.

    Covers CPR rule numbers, nested paragraphs, alphanumeric PD paragraph ids,
    and structural labels (Part, Practice Direction, Annex, Appendix,
    Schedule, Section). Returned in first-seen order with duplicates removed.
    """
    patterns = [
        r"\b\d+[A-Z]?\.\d+(?:\.\d+)*[A-Z]?\b",  # 3.1, 45.64, 8A.1, 3.1.2
        r"\bPart\s+\d+[A-Z]?\b",
        r"\bPractice\s+Direction\s+[0-9]+[A-Z]*\b",
        r"\bAnnex\s+[0-9A-Z]+\b",
        r"\bAppendix\s+[0-9A-Z]+\b",
        r"\bSchedule\s+[0-9A-Z]+\b",
        r"\bSection\s+\d+[A-Z]?\b",
    ]
    seen: dict[str, None] = {}
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            key = re.sub(r"\s+", " ", match.group(0)).strip()
            seen.setdefault(key, None)
    return list(seen.keys())


def coverage_layers(source_text: str, index_text: str) -> dict[str, Any]:
    """Compute raw, substantive, and identifier coverage for one source."""
    raw = compare_text_content(source_text, index_text)

    index_ngrams = extract_ngrams(normalize(index_text), 6)

    substantive_blocks: list[str] = []
    excluded: dict[str, int] = {}
    unmatched: list[dict[str, Any]] = []
    for block in split_blocks(source_text):
        classification = classify_block(block)
        if classification in NON_SUBSTANTIVE_CLASSIFICATIONS:
            excluded[classification] = excluded.get(classification, 0) + 1
            continue
        clean = strip_breadcrumb(block)
        substantive_blocks.append(clean)
        block_ngrams = extract_ngrams(normalize(clean), 6)
        if not block_ngrams:
            continue
        block_coverage = len(block_ngrams & index_ngrams) / len(block_ngrams)
        if block_coverage < MIN_BLOCK_COVERAGE:
            unmatched.append(
                {
                    "classification": classification,
                    "coverage": round(block_coverage, 4),
                    "length": len(normalize(clean)),
                    "snippet": re.sub(r"\s+", " ", clean).strip()[:240],
                }
            )

    substantive_text = "\n\n".join(substantive_blocks)
    substantive = compare_text_content(substantive_text, index_text)

    source_ids = extract_identifiers(source_text)
    index_ids = set(extract_identifiers(index_text))
    missing_ids = [identifier for identifier in source_ids if identifier not in index_ids]

    return {
        "raw_coverage": round(raw["source_to_index_coverage"], 4),
        "raw_index_to_source_coverage": round(raw["index_to_source_coverage"], 4),
        "substantive_coverage": round(substantive["source_to_index_coverage"], 4),
        "substantive_block_count": len(substantive_blocks),
        "excluded_block_counts": dict(sorted(excluded.items())),
        "identifier_count": len(source_ids),
        "missing_identifier_count": len(missing_ids),
        "missing_identifiers": missing_ids[:50],
        "unmatched_substantive_blocks": sorted(
            unmatched, key=lambda item: item["coverage"]
        )[:25],
    }


def provisional_cause(
    layers: dict[str, Any],
    reconciliation_issue: str,
    sibling: dict[str, Any],
) -> str:
    """Map diagnostic evidence to a provisional, non-destructive cause bucket.

    Deliberately conservative: this replaces the audit's blanket
    ``FAIL -> SOURCE_CHANGED`` assumption with an evidence-weighted guess that
    still defers to human review for ambiguous cases.
    """
    if "category/sourcefile differs" in reconciliation_issue or "multiple index source" in reconciliation_issue:
        return "METADATA_MAPPING"
    if sibling.get("partial_chunk_sequence"):
        return "PARTIAL_INDEX"
    substantive = layers["substantive_coverage"]
    raw = layers["raw_coverage"]
    unmatched = layers["unmatched_substantive_blocks"]
    missing_ids = layers["missing_identifier_count"]

    if substantive >= MIN_NGRAM_COVERAGE and not unmatched and missing_ids == 0:
        # The shortfall is entirely explained by non-substantive blocks.
        return "BENIGN_TRANSFORMATION"
    if raw < 0.5 and substantive < 0.7 and (unmatched or missing_ids > 0):
        return "SOURCE_CHANGED"
    if unmatched or missing_ids > 0:
        return "NEEDS_REVIEW"
    return "NEEDS_REVIEW"


def sibling_chunk_report(
    documents: Iterable[dict[str, Any]],
    sourcefile: str,
) -> dict[str, Any]:
    """Read-only inspection of every index chunk related to one source.

    Identifies partial chunk sequences (e.g. only ``chunk_011`` present with no
    ``chunk_000``) and metadata variance across siblings, which distinguishes a
    genuine partial index from an audit grouping/mapping artifact.
    """
    target = normalize_label(sourcefile)
    matches: list[dict[str, Any]] = []
    for document in documents:
        doc_sourcefile = normalize_label(str(document.get("sourcefile") or ""))
        parent = str(document.get("parent_id") or "")
        doc_id = str(document.get("id") or "")
        related = doc_sourcefile == target or target in normalize_label(parent) or target in normalize_label(doc_id)
        if not related:
            continue
        matches.append(
            {
                "id": doc_id,
                "parent_id": parent,
                "category": str(document.get("category") or ""),
                "storageUrl": str(document.get("storageUrl") or ""),
                "chunk_index": _chunk_index(doc_id),
            }
        )

    chunk_indices = sorted({m["chunk_index"] for m in matches if m["chunk_index"] is not None})
    partial = bool(chunk_indices) and (min(chunk_indices) > 0 or _has_gaps(chunk_indices))
    categories = sorted({m["category"] for m in matches})
    urls = sorted({normalize_url(m["storageUrl"]) for m in matches if m["storageUrl"]})

    return {
        "matched_document_count": len(matches),
        "chunk_indices": chunk_indices,
        "partial_chunk_sequence": partial,
        "distinct_categories": categories,
        "distinct_storage_urls": urls,
        "metadata_variance": len(categories) > 1 or len(urls) > 1,
        "documents": sorted(matches, key=lambda item: item["id"]),
    }


def _chunk_index(doc_id: str) -> int | None:
    match = re.search(r"_chunk_(\d+)$", doc_id)
    return int(match.group(1)) if match else None


def _has_gaps(indices: list[int]) -> bool:
    return bool(indices) and (indices != list(range(min(indices), max(indices) + 1)))


def index_text_for(result: Any, documents: Iterable[dict[str, Any]]) -> str:
    by_id = {str(document.get("id") or ""): document for document in documents}
    parts = [
        str(by_id[identifier].get("content") or "")
        for identifier in result.index_document_ids
        if identifier in by_id
    ]
    return "\n\n".join(parts)


def embedding_chunk_diagnostics(documents: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Measure local chunk and historical embedding-input risks without Azure."""
    counts = {
        "document_count": 0,
        "over_embedding_token_limit": 0,
        "near_embedding_token_limit": 0,
        "historical_character_truncated": 0,
        "missing_content": 0,
        "missing_source_metadata": 0,
    }
    token_counts: list[int] = []
    examples: list[dict[str, Any]] = []

    for document in documents:
        counts["document_count"] += 1
        content = document.get("content")
        if not content:
            counts["missing_content"] += 1
            continue
        text = " ".join(content) if isinstance(content, list) else str(content)
        normalized_text = text.replace("\n", " ")
        token_count = len(EMBEDDING_ENCODING.encode(normalized_text))
        token_counts.append(token_count)
        historical_text = normalized_text[:HISTORICAL_CHARACTER_LIMIT]
        historical_tokens = len(EMBEDDING_ENCODING.encode(historical_text))
        safe_tokens = min(token_count, EMBEDDING_MAX_TOKENS)

        if token_count > EMBEDDING_MAX_TOKENS:
            counts["over_embedding_token_limit"] += 1
        if token_count >= EMBEDDING_MAX_TOKENS * 0.9:
            counts["near_embedding_token_limit"] += 1
        if len(normalized_text) > HISTORICAL_CHARACTER_LIMIT:
            counts["historical_character_truncated"] += 1
        if not document.get("sourcefile") or not document.get("sourcepage"):
            counts["missing_source_metadata"] += 1

        if len(examples) < 100 and (
            len(normalized_text) > HISTORICAL_CHARACTER_LIMIT
            or token_count >= EMBEDDING_MAX_TOKENS * 0.9
            or token_count > EMBEDDING_MAX_TOKENS
        ):
            examples.append(
                {
                    "id": str(document.get("id") or ""),
                    "sourcefile": str(document.get("sourcefile") or ""),
                    "token_count": token_count,
                    "historical_character_input_tokens": historical_tokens,
                    "token_safe_input_tokens": safe_tokens,
                    "historical_tail_tokens_lost": max(0, token_count - historical_tokens),
                    "token_safe_tail_tokens_lost": max(0, token_count - safe_tokens),
                }
            )

    return {
        "limits": {
            "embedding_model": "text-embedding-3-large",
            "embedding_max_tokens": EMBEDDING_MAX_TOKENS,
            "historical_character_limit": HISTORICAL_CHARACTER_LIMIT,
        },
        "counts": counts,
        "token_count": {
            "minimum": min(token_counts) if token_counts else 0,
            "maximum": max(token_counts) if token_counts else 0,
            "average": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
        },
        "examples": sorted(examples, key=lambda item: item["historical_tail_tokens_lost"], reverse=True),
    }


def remediation_action(cause: str, sibling: dict[str, Any], coverage: dict[str, Any] | None) -> str:
    """Return the next read-only or approval-gated action for a finding."""
    actions = {
        "PARTIAL_INDEX": "Verify source artifact and upload manifest, then prepare a source-scoped reindex for approval.",
        "METADATA_MAPPING": "Reconcile canonical source identity, category, and URL aliases before changing content.",
        "BENIGN_TRANSFORMATION": "No index change indicated; retain as an accepted parser or site-furniture difference.",
        "NEEDS_SOURCE_TEXT": "Fetch or repair the canonical source evidence before classifying the finding.",
        "SOURCE_CHANGED": "Compare the current source against the indexed artifact and review substantive omissions before reindexing.",
        "NEEDS_REVIEW": "Inspect unmatched substantive blocks and sibling chunks; do not reindex from aggregate coverage alone.",
    }
    action = actions.get(cause, "Review evidence before taking any index action.")
    if cause == "PARTIAL_INDEX" and sibling.get("metadata_variance"):
        return "Resolve sibling metadata variance before preparing a source-scoped reindex."
    if coverage and coverage.get("missing_identifier_count", 0) > 0 and cause == "NEEDS_REVIEW":
        return "Review missing legal identifiers and unmatched blocks against the source artifact before reindexing."
    return action


def diagnose(
    documents: list[dict[str, Any]],
    sources: list[CanonicalSource],
    *,
    fetch_html: bool,
    html_cache: Path | None,
) -> dict[str, Any]:
    results = reconcile_sources(sources, documents, include_index_only=False)
    canonical_by_identity = {source.identity: source for source in sources}

    scraper = load_web_scraper() if fetch_html else None
    cache = HtmlAuditCache(html_cache) if fetch_html else None
    session = None
    if fetch_html:
        import requests

        session = requests.Session()
        session.headers.update({"User-Agent": "legal-source-fidelity-diagnostic/1.0 (read-only)"})

    findings: list[dict[str, Any]] = []
    for result in results:
        if not result.index_document_ids:
            continue
        source = canonical_by_identity.get(f"{normalize_label(result.category)}::{normalize_label(result.sourcefile)}")
        sibling = sibling_chunk_report(documents, result.sourcefile)
        index_text = index_text_for(result, documents)
        reconciliation_issue = "; ".join(result.issues)

        layers: dict[str, Any] | None = None
        source_text = ""
        if result.source_type == "html" and fetch_html and source is not None and normalize_url(source.url):
            scraped, requested_url = scrape_with_cache(session, source, scraper, cache)
            if scraped is not None:
                source_text = str(scraped.get("content") or "")
        if source_text and index_text:
            layers = coverage_layers(source_text, index_text)

        cause = (
            provisional_cause(layers, reconciliation_issue, sibling)
            if layers is not None
            else ("PARTIAL_INDEX" if sibling["partial_chunk_sequence"] else "NEEDS_SOURCE_TEXT")
        )

        findings.append(
            {
                "sourcefile": result.sourcefile,
                "category": result.category,
                "source_type": result.source_type,
                "status": result.status,
                "reconciliation_issues": result.issues,
                "provisional_cause": cause,
                "recommended_action": remediation_action(cause, sibling, layers),
                "sibling_chunks": sibling,
                "coverage": layers,
            }
        )

    findings.sort(key=lambda item: (item["provisional_cause"], normalize_label(item["sourcefile"])))
    cause_totals: dict[str, int] = {}
    for finding in findings:
        cause_totals[finding["provisional_cause"]] = cause_totals.get(finding["provisional_cause"], 0) + 1

    return {
        "schema_version": 1,
        "read_only": True,
        "embedding_chunk_diagnostics": embedding_chunk_diagnostics(documents),
        "summary": {
            "finding_count": len(findings),
            "provisional_causes": dict(sorted(cause_totals.items())),
        },
        "findings": findings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-snapshot", type=Path, required=True, help="Cached read-only index JSON array")
    parser.add_argument("--family", choices=("all", "pdf", "html"), default="html")
    parser.add_argument("--source", help="Diagnose one canonical sourcefile (case-insensitive)")
    parser.add_argument("--no-html", action="store_true", help="Skip live HTML fetch; sibling-chunk analysis only")
    parser.add_argument("--html-cache", type=Path, default=DEFAULT_HTML_CACHE, help="Per-URL HTML response checkpoint JSON")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_DIAGNOSTIC_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    documents = json.loads(args.index_snapshot.read_text(encoding="utf-8"))
    sources = load_pdf_sources() + load_web_sources()
    if args.family != "all":
        sources = [source for source in sources if source.source_type == args.family]
    if args.source:
        requested = normalize_label(args.source)
        sources = [source for source in sources if normalize_label(source.sourcefile) == requested]
        if not sources:
            raise SystemExit(f"No canonical sourcefile matched {args.source!r}")

    report = diagnose(
        documents,
        sources,
        fetch_html=not args.no_html,
        html_cache=args.html_cache,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
