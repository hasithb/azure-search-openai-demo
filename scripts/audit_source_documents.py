#!/usr/bin/env python3
"""Audit canonical legal sources against a read-only Azure AI Search inventory.

The initial inventory pass deliberately has no Azure dependency: pass an exported
index snapshot to reconcile canonical PDF and web manifests. Content fidelity
adapters build on the same source/result model.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import tempfile
import os
import re
import sys
import hashlib
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

try:
    from scripts.court_guides_processing_pipeline.verify_extraction_completeness import (
        MIN_NGRAM_COVERAGE,
        MIN_PAGE_COVERAGE,
        classify_missing_block,
        extract_ngrams,
        normalize,
    )
except ModuleNotFoundError:
    from court_guides_processing_pipeline.verify_extraction_completeness import (
        MIN_NGRAM_COVERAGE,
        MIN_PAGE_COVERAGE,
        classify_missing_block,
        extract_ngrams,
        normalize,
    )

ROOT = Path(__file__).resolve().parents[1]
PDF_MANIFEST = ROOT / "scripts" / "court_guides_processing_pipeline" / "scripts" / "extract_court_guides_azure_di.py"
PDF_SOURCE_DIR = ROOT / "scripts" / "court_guides_processing_pipeline" / "sources"
WEB_MANIFEST = ROOT / "scripts" / "update_cpr_index_v3.py"
WEB_CORPUS_DIR = ROOT / "data" / "legal-scraper" / "processed" / "Upload"
DEFAULT_JSON_REPORT = ROOT / "reports" / "source_document_accuracy.json"
DEFAULT_MARKDOWN_REPORT = ROOT / "reports" / "source_document_accuracy.md"
DEFAULT_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE", "gptkb-gz2m4s637t5me")
DEFAULT_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "legal-court-rag-index-v3")
INDEX_SELECT_FIELDS = [
    "id",
    "content",
    "category",
    "sourcepage",
    "sourcefile",
    "storageUrl",
    "updated",
    "parent_id",
    "subsection_id",
    "subsections",
]
SNAPSHOT_SCHEMA_VERSION = 1

TERMINAL_FAILURE_STATUSES = {"FAIL", "MISSING_FROM_INDEX", "UNAVAILABLE", "UNMAPPED"}
REMEDIATION_STATUSES = {
    "SOURCE_CHANGED",
    "INDEX_INCOMPLETE",
    "MANIFEST_DRIFT",
    "SCRAPER_FAILURE",
    "EXTRACTION_FAILURE",
    "INTENTIONAL_EXCLUSION",
    "NEEDS_REVIEW",
}
DEFAULT_HTML_CACHE = ROOT / "reports" / "source_document_http_cache.json"


@dataclass(frozen=True)
class CanonicalSource:
    source_type: str
    sourcefile: str
    category: str
    url: str = ""
    local_path: str = ""
    updated: str = ""
    manifest_key: str = ""

    @property
    def identity(self) -> str:
        return f"{normalize_label(self.category)}::{normalize_label(self.sourcefile)}"


@dataclass(frozen=True)
class IndexSource:
    category: str
    sourcefile: str
    url: str
    document_ids: tuple[str, ...]

    @property
    def identity(self) -> str:
        return f"{normalize_label(self.category)}::{normalize_label(self.sourcefile)}"


@dataclass
class SourceAuditResult:
    source_type: str
    sourcefile: str
    category: str
    canonical_url: str
    index_url: str
    index_document_ids: list[str]
    status: str
    remediation_status: str = ""
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_text_content(source_text: str, index_text: str, ngram_size: int = 6) -> dict[str, Any]:
    source_ngrams = extract_ngrams(normalize(source_text), ngram_size)
    index_ngrams = extract_ngrams(normalize(index_text), ngram_size)
    shared = source_ngrams & index_ngrams
    return {
        "ngram_size": ngram_size,
        "source_ngram_count": len(source_ngrams),
        "index_ngram_count": len(index_ngrams),
        "shared_ngram_count": len(shared),
        "source_to_index_coverage": len(shared) / len(source_ngrams) if source_ngrams else 1.0,
        "index_to_source_coverage": len(shared) / len(index_ngrams) if index_ngrams else 1.0,
    }


def classify_remediation(result: SourceAuditResult) -> str:
    """Map observable audit findings to an actionable remediation bucket."""
    issue_text = " ".join(result.issues).casefold()
    if result.status == "INDEX_ONLY":
        return "MANIFEST_DRIFT"
    if result.status == "UNMAPPED":
        return "MANIFEST_DRIFT"
    if result.status == "MISSING_FROM_INDEX":
        return "INDEX_INCOMPLETE"
    if result.status == "UNAVAILABLE" and "could not be fetched" in issue_text:
        return "SCRAPER_FAILURE"
    if result.status == "UNAVAILABLE" and "local pdf not found" in issue_text:
        return "EXTRACTION_FAILURE"
    if "duplicate canonical" in issue_text or "category/sourcefile differs" in issue_text:
        return "MANIFEST_DRIFT"
    if "could not be fetched" in issue_text or "canonical url" in issue_text:
        return "SCRAPER_FAILURE"
    if "local pdf not found" in issue_text:
        return "EXTRACTION_FAILURE"
    if result.status == "FAIL":
        return "NEEDS_REVIEW"
    if result.status == "UNAVAILABLE":
        return "NEEDS_REVIEW"
    if result.status == "PASS":
        return "VERIFIED_PRESENT"
    return "NEEDS_REVIEW" if result.status == "WARN" else ""


def set_remediation_status(results: Iterable[SourceAuditResult]) -> list[SourceAuditResult]:
    audited = list(results)
    for result in audited:
        result.remediation_status = classify_remediation(result)
    return audited


def apply_block_gate(result: SourceAuditResult, source_label: str) -> None:
    block_result = result.metrics.get("substantive_blocks", {})
    ambiguous_count = int(block_result.get("ambiguous_block_count", 0))
    unmatched_count = int(block_result.get("unmatched_block_count", 0))
    cross_document_overlap_count = int(block_result.get("cross_document_overlap_count", 0))
    if ambiguous_count:
        result.status = "FAIL"
        result.issues.append(
            f"{ambiguous_count} substantive {source_label} block(s) have ambiguous index matches"
        )
    if unmatched_count:
        result.status = "FAIL"
        result.issues.append(
            f"{unmatched_count} substantive {source_label} block(s) are missing from the index"
        )
    if cross_document_overlap_count:
        result.status = "FAIL"
        result.issues.append(
            f"{cross_document_overlap_count} substantive {source_label} block(s) overlap another source family"
        )


def compare_pdf_content(pdf_path: Path, index_documents: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pymupdf

    documents = list(index_documents)
    index_text = "\n\n".join(str(document.get("content") or "") for document in documents)
    index_ngrams = extract_ngrams(normalize(index_text), 6)
    page_texts: list[str] = []
    evidence: list[dict[str, Any]] = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text()
            page_texts.append(page_text)
            page_ngrams = extract_ngrams(normalize(page_text), 6)
            if not page_ngrams:
                continue
            coverage = len(page_ngrams & index_ngrams) / len(page_ngrams)
            if coverage >= MIN_PAGE_COVERAGE:
                continue
            category = classify_missing_block(page_text)
            evidence.append(
                {
                    "kind": "low_pdf_page_coverage",
                    "page": page_number,
                    "coverage": coverage,
                    "classification": category,
                    "snippet": re.sub(r"\s+", " ", page_text).strip()[:240],
                }
            )

    metrics = compare_text_content("\n".join(page_texts), index_text)
    metrics["substantive_blocks"] = compare_substantive_blocks(
        "\n".join(page_texts), index_text, source_type="pdf", index_documents=documents
    )
    metrics.update(
        {
            "pdf_pages": len(page_texts),
            "index_document_count": len(documents),
            "low_coverage_pages": len(evidence),
            "substantive_low_coverage_pages": sum(
                item["classification"] == "POTENTIALLY_SUBSTANTIVE" for item in evidence
            ),
        }
    )
    return metrics, evidence


def apply_pdf_fidelity(
    results: Iterable[SourceAuditResult],
    canonical_sources: Iterable[CanonicalSource],
    index_documents: Iterable[dict[str, Any]],
) -> list[SourceAuditResult]:
    canonical_by_identity = {source.identity: source for source in canonical_sources if source.source_type == "pdf"}
    documents_by_id = {str(document.get("id") or ""): document for document in index_documents}
    audited = list(results)
    for result in audited:
        if result.source_type != "pdf" or not result.index_document_ids:
            continue
        source = canonical_by_identity.get(f"{normalize_label(result.category)}::{normalize_label(result.sourcefile)}")
        if source is None:
            continue
        pdf_path = Path(source.local_path)
        if not pdf_path.exists():
            result.status = "UNAVAILABLE"
            result.issues.append(f"local PDF not found: {pdf_path}")
            continue
        documents = [documents_by_id[identifier] for identifier in result.index_document_ids if identifier in documents_by_id]
        result.metrics, result.evidence = compare_pdf_content(pdf_path, documents)
        block_result = result.metrics["substantive_blocks"]
        result.evidence.extend(block_result["unmatched_blocks"][:100])
        result.evidence.extend(block_result["ambiguous_blocks"][:100])
        if not block_result["source_block_count"]:
            result.status = "UNAVAILABLE"
            result.issues.append("PDF produced no substantive legal blocks")
        else:
            apply_block_gate(result, "PDF")
        result.metrics["ngram_below_threshold"] = (
            result.metrics["source_to_index_coverage"] < MIN_NGRAM_COVERAGE
        )
    return audited


def extract_rule_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+[A-Z]?\.\d+[A-Z]?\b", text))


def load_web_scraper() -> Any:
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import update_cpr_index_v3

    return update_cpr_index_v3


def apply_html_fidelity(
    results: Iterable[SourceAuditResult],
    canonical_sources: Iterable[CanonicalSource],
    index_documents: Iterable[dict[str, Any]],
    scraper: Any | None = None,
    cache_path: Path | None = DEFAULT_HTML_CACHE,
) -> list[SourceAuditResult]:
    import requests

    canonical_by_identity = {source.identity: source for source in canonical_sources if source.source_type == "html"}
    documents_by_id = {str(document.get("id") or ""): document for document in index_documents}
    audited = list(results)
    scraper = scraper or load_web_scraper()
    cache = HtmlAuditCache(cache_path)
    session = requests.Session()
    session.headers.update({"User-Agent": "legal-source-fidelity-audit/1.0 (read-only)"})
    for result in audited:
        if result.source_type != "html" or not result.index_document_ids:
            continue
        source = canonical_by_identity.get(f"{normalize_label(result.category)}::{normalize_label(result.sourcefile)}")
        if source is None or not normalize_url(source.url):
            continue
        scraped, requested_url = scrape_with_cache(session, source, scraper, cache)
        if scraped is None:
            result.status = "UNAVAILABLE"
            result.issues.append(f"official source could not be fetched or parsed: {requested_url}")
            continue
        documents = [documents_by_id[identifier] for identifier in result.index_document_ids if identifier in documents_by_id]
        index_text = "\n\n".join(str(document.get("content") or "") for document in documents)
        live_text = str(scraped.get("content") or "")
        result.metrics = compare_text_content(live_text, index_text)
        result.metrics["substantive_blocks"] = compare_substantive_blocks(
            live_text,
            index_text,
            source_type="html",
            index_documents=documents,
            sourcefile=source.sourcefile,
        )
        live_rules = extract_rule_numbers(live_text)
        indexed_rules = extract_rule_numbers(index_text)
        missing_rules = sorted(live_rules - indexed_rules)
        result.metrics.update(
            {
                "live_character_count": len(live_text),
                "index_character_count": len(index_text),
                "missing_rule_count": len(missing_rules),
                "final_url": scraped.get("_final_url", source.url),
                "redirect_count": scraped.get("_redirect_count", 0),
                "requested_url": requested_url,
            }
        )
        result.evidence = [{"kind": "missing_rule_reference", "rule": rule} for rule in missing_rules[:25]]
        block_result = result.metrics["substantive_blocks"]
        result.evidence.extend(block_result["unmatched_blocks"][:100])
        result.evidence.extend(block_result["ambiguous_blocks"][:100])
        if not block_result["source_block_count"]:
            result.status = "UNAVAILABLE"
            result.issues.append("HTML produced no substantive legal blocks")
        else:
            apply_block_gate(result, "HTML")
        result.metrics["ngram_below_threshold"] = (
            result.metrics["source_to_index_coverage"] < MIN_NGRAM_COVERAGE
        )
    return audited


def normalize_label(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip().casefold()


def normalize_legal_block(value: str) -> str:
    """Normalize legal text without discarding words needed for fidelity checks."""
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("“", '"').replace("”", '"').replace("’", "'")
    value = value.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip()


def compact_legal_block(value: str) -> str:
    """Remove presentation punctuation while retaining every legal word."""
    value = normalize_legal_block(value).casefold()
    value = re.sub(r"^#{1,6}\s*", "", value)
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)).strip()


def count_legal_occurrences(text: str, block: str) -> int:
    if not block:
        return 0
    count = 0
    start = 0
    while True:
        match_start = text.find(block, start)
        if match_start < 0:
            return count
        match_end = match_start + len(block)
        before_is_word = match_start > 0 and (text[match_start - 1] == "_" or text[match_start - 1].isalnum())
        after_is_word = match_end < len(text) and (text[match_end] == "_" or text[match_end].isalnum())
        if not before_is_word and not after_is_word:
            count += 1
        start = match_end


def _is_boilerplate_line(line: str) -> bool:
    normalized = normalize_label(line)
    return not normalized or normalized in {
        "www.justice.gov.uk",
        "www.judiciary.uk",
        "page",
    } or bool(re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", normalized))


def extract_substantive_blocks(
    text: str,
    source_type: str = "text",
    locator_prefix: str = "",
) -> list[dict[str, Any]]:
    """Extract stable, reviewable legal blocks from canonical source text.

    The extractor intentionally keeps headings, numbered provisions, tables,
    footnotes, schedules, annexes, and operative form text. Only blank and
    repeated page-furniture lines are treated as boilerplate.
    """
    lines = [normalize_legal_block(line) for line in (text or "").splitlines()]
    blocks: list[dict[str, Any]] = []
    occurrence_counts: dict[str, int] = defaultdict(int)
    for line_number, line in enumerate(lines, start=1):
        if _is_boilerplate_line(line):
            continue
        if source_type == "pdf" and line_number > 1 and lines[line_number - 2].endswith("-"):
            continue
        normalized = normalize_legal_block(line)
        if len(normalized) < 2:
            continue
        block_hash = hashlib.sha256(normalized.casefold().encode("utf-8")).hexdigest()
        occurrence_counts[block_hash] += 1
        block_kind = "text"
        if re.match(r"^(#{1,6}\s|\[?(part|practice direction|schedule|annex|appendix|section)\b)", normalized, re.I):
            block_kind = "heading"
        elif re.match(r"^(\d+[A-Z]?(?:\.\d+)*|[A-Z][.)]|\([a-z0-9]+\))\s+", normalized):
            block_kind = "numbered_paragraph"
        elif "|" in normalized or "\t" in line:
            block_kind = "table_row"
        elif re.match(r"^(footnote|note)\s*\d*[:.]?\s", normalized, re.I):
            block_kind = "footnote"
        blocks.append(
            {
                "block_id": f"{locator_prefix or source_type}:{line_number}:{block_hash[:16]}",
                "kind": block_kind,
                "locator": {"line": line_number, "source_type": source_type},
                "normalized_hash": block_hash,
                "occurrence_ordinal": occurrence_counts[block_hash],
                "text": line,
                "match_text": (
                    f"{line[:-1]}{lines[line_number]}"
                    if line.endswith("-") and line_number < len(lines) and lines[line_number]
                    else line
                ),
            }
        )
    return blocks


def compare_substantive_blocks(
    source_text: str,
    index_text: str,
    source_type: str = "text",
    index_documents: list[dict[str, Any]] | None = None,
    sourcefile: str = "",
) -> dict[str, Any]:
    """Return block-level evidence; this is stricter than aggregate n-gram scores."""
    blocks = extract_substantive_blocks(source_text, source_type)
    normalized_index = normalize_legal_block(index_text).casefold()
    flattened_index = re.sub(r"\s*[|]+\s*", " ", normalized_index).strip()
    compact_index = compact_legal_block(index_text)
    document_texts = [
        {
            "id": str(document.get("id") or ""),
            "sourcefile": str(document.get("sourcefile") or "").strip(),
            "text": str(document.get("content") or ""),
        }
        for document in (index_documents or [])
        if document.get("content")
    ]
    for document in document_texts:
        document["normalized_text"] = normalize_legal_block(document["text"]).casefold()
        document["flattened_text"] = re.sub(r"\s*[|]+\s*", " ", document["normalized_text"]).strip()
        document["compact_text"] = compact_legal_block(document["text"])
    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    cross_document_overlaps: list[dict[str, Any]] = []
    count_cache: dict[tuple[str, str], int] = {}
    source_occurrence_counts: dict[str, int] = {}
    for block in blocks:
        source_key = normalize_legal_block(block.get("match_text", block["text"])).casefold()
        source_occurrence_counts[source_key] = source_occurrence_counts.get(source_key, 0) + 1

    def cached_count(view_name: str, text: str, block: str) -> int:
        cache_key = (view_name, block)
        if cache_key not in count_cache:
            count_cache[cache_key] = count_legal_occurrences(text, block)
        return count_cache[cache_key]

    for block in blocks:
        normalized_block = normalize_legal_block(block.get("match_text", block["text"])).casefold()
        flattened_block = re.sub(r"\s*[|]+\s*", " ", normalized_block).strip()
        compact_block = compact_legal_block(normalized_block) if normalized_block else ""
        evidence = {
            "block_id": block["block_id"],
            "source_identity": f"{source_type}::{sourcefile}" if sourcefile else source_type,
            "kind": block["kind"],
            "locator": block["locator"],
            "normalized_hash": block["normalized_hash"],
            "occurrence_ordinal": block["occurrence_ordinal"],
            "text": block["text"],
        }
        occurrences = cached_count("index", normalized_index, normalized_block)
        match_method = "normalized_substring"
        if occurrences == 0 and flattened_block != normalized_block:
            occurrences = cached_count("flattened_index", flattened_index, flattened_block)
            match_method = "flattened_table_substring"
        if occurrences == 0 and normalized_block:
            occurrences = cached_count("compact_index", compact_index, compact_block)
            match_method = "compact_formatting_substring"
        raw_occurrences = occurrences
        source_key = normalized_block
        source_has_repeated_occurrences = source_occurrence_counts.get(source_key, 0) > 1
        if source_has_repeated_occurrences and block["occurrence_ordinal"] <= occurrences:
            evidence["match_ordinal"] = block["occurrence_ordinal"]
            occurrences = 1
            match_method = f"{match_method}_occurrence_reconciled"
        document_matches: list[str] = []
        matching_sourcefiles: set[str] = set()
        # Document-level scans only disambiguate duplicate whole-index matches.
        if document_texts and raw_occurrences > 1:
            for document in document_texts:
                document_id = document["id"]
                document_occurrences = cached_count(
                    f"document:{document_id}:normalized", document["normalized_text"], normalized_block
                )
                if document_occurrences == 0 and flattened_block != normalized_block:
                    document_occurrences = cached_count(
                        f"document:{document_id}:flattened", document["flattened_text"], flattened_block
                    )
                if document_occurrences == 0 and normalized_block:
                    document_occurrences = cached_count(
                        f"document:{document_id}:compact", document["compact_text"], compact_block
                    )
                if document_occurrences:
                    document_matches.append(document["id"])
                    matching_sourcefiles.add(str(document.get("sourcefile") or "").strip())
            evidence["matching_document_ids"] = document_matches
            evidence["matching_sourcefiles"] = sorted(value for value in matching_sourcefiles if value)
            same_source_matches = bool(sourcefile) and sourcefile in matching_sourcefiles
            if len(document_matches) == 1 and raw_occurrences > 1 and not source_has_repeated_occurrences:
                occurrences = 1
                match_method = f"{match_method}_document_scoped"
            elif occurrences > 1 and matching_sourcefiles and not same_source_matches:
                evidence["match_method"] = "cross_document_overlap"
                evidence["match_count"] = occurrences
                cross_document_overlaps.append(evidence)
                continue
        if occurrences == 1:
            evidence["match_method"] = match_method
            matched.append(evidence)
        elif occurrences > 1:
            evidence["match_method"] = "ambiguous"
            evidence["match_count"] = occurrences
            ambiguous.append(evidence)
        else:
            evidence["match_method"] = "unmatched"
            unmatched.append(evidence)
    return {
        "source_block_count": len(blocks),
        "matched_block_count": len(matched),
        "unmatched_block_count": len(unmatched),
        "ambiguous_block_count": len(ambiguous),
        "substantive_block_coverage": len(matched) / len(blocks) if blocks else 0.0,
        "matched_blocks": matched,
        "unmatched_blocks": unmatched,
        "ambiguous_blocks": ambiguous,
        "cross_document_overlap_count": len(cross_document_overlaps),
        "cross_document_overlaps": cross_document_overlaps,
        "occurrence_ledger": sorted(
            [
                {
                    **block,
                    "status": (
                        "MATCHED"
                        if block in matched
                        else "AMBIGUOUS"
                        if block in ambiguous
                        else "CROSS_DOCUMENT_OVERLAP"
                        if block in cross_document_overlaps
                        else "UNMATCHED"
                    ),
                }
                for block in [*matched, *ambiguous, *cross_document_overlaps, *unmatched]
            ],
            key=lambda block: (int(block["locator"].get("line", 0)), block["block_id"]),
        ),
    }


def normalize_url(value: str) -> str:
    if not value or value == "DISCOVER_FROM_PROTOCOL_PAGE":
        return ""
    parsed = urlsplit(unquote(value.strip()))
    host = parsed.netloc.casefold().removeprefix("www.")
    path = re.sub(r"/+", "/", parsed.path).rstrip("/") or "/"
    return urlunsplit((parsed.scheme.casefold() or "https", host, path, "", ""))


KNOWN_URL_ALIASES = {
    "Practice Direction 40F": [
        "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/practice-direction-40f-proceedings-involving-declarations-of-incompatibility",
    ],
}


class HtmlAuditCache:
    """Small JSON checkpoint store so interrupted crawls resume per URL."""

    def __init__(self, path: Path | None):
        self.path = path
        self.entries: dict[str, dict[str, Any]] = {}
        if path and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.entries = dict(payload.get("entries", {}))
            except (OSError, ValueError):
                self.entries = {}

    def get(self, url: str) -> dict[str, Any] | None:
        return self.entries.get(normalize_url(url))

    def put(self, url: str, value: dict[str, Any]) -> None:
        if not self.path:
            return
        self.entries[normalize_url(url)] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps({"schema_version": 1, "entries": self.entries}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.path)


def scrape_with_cache(
    session: Any,
    source: CanonicalSource,
    scraper: Any,
    cache: HtmlAuditCache,
) -> tuple[dict[str, Any] | None, str]:
    urls = [source.url, *KNOWN_URL_ALIASES.get(source.sourcefile, [])]
    for url in urls:
        if not normalize_url(url):
            continue
        cached = cache.get(url)
        if cached is not None:
            if cached.get("ok"):
                return dict(cached["result"]), url
            continue
        action = {"sourcefile": source.sourcefile, "url": url}
        scraped = scraper.scrape_page(session, action)
        cache.put(url, {"ok": scraped is not None, "result": scraped or {}, "requested_url": url})
        if scraped is not None:
            return scraped, url
    return None, source.url


def odata_escape(value: str) -> str:
    return value.replace("'", "''")


def endpoint_url(service: str) -> str:
    return service if service.startswith("https://") else f"https://{service}.search.windows.net"


def fetch_index_documents(client: Any) -> list[dict[str, Any]]:
    results = list(
        client.search(
            search_text="*",
            select=INDEX_SELECT_FIELDS,
            top=1000,
            include_total_count=True,
        )
    )
    if len(results) >= 1000:
        results = []
        facet_result = client.search(search_text="*", facets=["category,count:100"], top=0)
        facets = facet_result.get_facets() or {}
        categories = sorted(str(item["value"]) for item in facets.get("category", []))
        for category in categories:
            results.extend(
                client.search(
                    search_text="*",
                    filter=f"category eq '{odata_escape(category)}'",
                    select=INDEX_SELECT_FIELDS,
                    top=1000,
                )
            )
    return sorted((dict(result) for result in results), key=lambda document: str(document.get("id") or ""))


def fetch_live_index_documents(service: str, index_name: str) -> list[dict[str, Any]]:
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient

    client = SearchClient(
        endpoint=endpoint_url(service),
        index_name=index_name,
        credential=DefaultAzureCredential(),
    )
    return fetch_index_documents(client)


def _snapshot_documents_hash(documents: Iterable[dict[str, Any]]) -> str:
    payload = json.dumps(list(documents), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def serialize_index_snapshot(
    documents: Iterable[dict[str, Any]], service: str, index_name: str, captured_at_utc: str | None = None
) -> dict[str, Any]:
    ordered_documents = sorted((dict(document) for document in documents), key=lambda document: str(document.get("id") or ""))
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "service": service,
        "index": index_name,
        "captured_at_utc": captured_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selected_fields": list(INDEX_SELECT_FIELDS),
        "document_count": len(ordered_documents),
        "documents_sha256": _snapshot_documents_hash(ordered_documents),
        "documents": ordered_documents,
    }


def load_index_snapshot(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        documents = [dict(document) for document in payload]
        return documents, {"verified": False, "format": "legacy_array", "document_count": len(documents)}
    if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
        raise ValueError(f"Index snapshot must be a document array or provenance envelope: {path}")
    required = (
        "schema_version",
        "service",
        "index",
        "captured_at_utc",
        "selected_fields",
        "document_count",
        "documents_sha256",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Index snapshot envelope is missing fields: {', '.join(missing)}")
    if payload["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported index snapshot schema version: {payload['schema_version']}")
    for field in ("service", "index", "captured_at_utc"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"Index snapshot provenance field is empty: {field}")
    if payload["selected_fields"] != INDEX_SELECT_FIELDS:
        raise ValueError("Index snapshot selected fields do not match the audit field selection")
    documents = [dict(document) for document in payload["documents"]]
    if payload["document_count"] != len(documents):
        raise ValueError(f"Index snapshot document count mismatch: expected {payload['document_count']}, found {len(documents)}")
    actual_hash = _snapshot_documents_hash(documents)
    if payload["documents_sha256"] != actual_hash:
        raise ValueError("Index snapshot document hash mismatch")
    provenance = {
        key: payload[key]
        for key in ("schema_version", "service", "index", "captured_at_utc", "selected_fields", "document_count", "documents_sha256")
    }
    provenance.update({"verified": True, "format": "envelope"})
    return documents, provenance


def write_index_snapshot(path: Path, documents: Iterable[dict[str, Any]], service: str, index_name: str) -> dict[str, Any]:
    snapshot = serialize_index_snapshot(documents, service, index_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {key: snapshot[key] for key in snapshot if key != "documents"} | {"verified": True, "format": "envelope"}


def load_literal_assignment(path: Path, assignment_name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == assignment_name for target in targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Could not find literal assignment {assignment_name!r} in {path}")


def load_pdf_sources(path: Path = PDF_MANIFEST) -> list[CanonicalSource]:
    metadata = load_literal_assignment(path, "GUIDE_METADATA")
    return [
        CanonicalSource(
            source_type="pdf",
            sourcefile=item["sourcefile"],
            category=item["category"],
            url=item.get("storageUrl", ""),
            local_path=str(PDF_SOURCE_DIR / filename),
            updated=item.get("updated", ""),
            manifest_key=filename,
        )
        for filename, item in sorted(metadata.items())
    ]


def load_web_sources(path: Path = WEB_MANIFEST, corpus_dir: Path = WEB_CORPUS_DIR) -> list[CanonicalSource]:
    actions = load_literal_assignment(path, "ACTION_LIST")
    def source_type_for(item: dict[str, Any]) -> str:
        url = str(item.get("url") or "").split("?", 1)[0].casefold()
        return "pdf" if url.endswith(".pdf") else "html"

    sources_by_identity: dict[str, CanonicalSource] = {}
    for item in actions:
        source = CanonicalSource(
            source_type=source_type_for(item),
            sourcefile=item["sourcefile"],
            category="Civil Procedure Rules and Practice Directions",
            url=item.get("url", ""),
            manifest_key=item.get("azure_id") or item["sourcefile"],
        )
        sources_by_identity.setdefault(source.identity, source)

    # A URL shared by aliases in the same category is one canonical source;
    # different categories and unresolved URL conflicts remain separate.
    by_url_and_category: dict[tuple[str, str], CanonicalSource] = {}
    for source in sources_by_identity.values():
        normalized_url = normalize_url(source.url)
        key = (normalized_url, normalize_label(source.category)) if normalized_url else (source.identity, "")
        by_url_and_category.setdefault(key, source)
    for document_path in sorted(corpus_dir.glob("*.json")):
        document = json.loads(document_path.read_text(encoding="utf-8"))
        source = CanonicalSource(
            source_type="html",
            sourcefile=str(document.get("sourcefile") or ""),
            category=str(document.get("category") or "Civil Procedure Rules and Practice Directions"),
            url=str(document.get("storageUrl") or ""),
            updated=str(document.get("updated") or ""),
            manifest_key=str(document.get("parent_id") or document.get("id") or document_path.stem),
        )
        if source.sourcefile:
            if source.identity not in {item.identity for item in by_url_and_category.values()}:
                by_url_and_category.setdefault((normalize_url(source.url), source.identity), source)
    return sorted(by_url_and_category.values(), key=lambda source: source.identity)


def group_index_documents(documents: Iterable[dict[str, Any]]) -> list[IndexSource]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    display: dict[tuple[str, str], tuple[str, str, str]] = {}
    for document in documents:
        category = str(document.get("category") or "")
        sourcefile = str(document.get("sourcefile") or "")
        url = str(document.get("storageUrl") or "")
        location = normalize_url(url) or f"sourcefile:{normalize_label(sourcefile)}"
        key = (normalize_label(category), location)
        display.setdefault(key, (category, sourcefile, url))
        groups[key].append(str(document.get("id") or ""))

    return [
        IndexSource(
            category=display[key][0],
            sourcefile=display[key][1],
            url=display[key][2],
            document_ids=tuple(sorted(identifier for identifier in identifiers if identifier)),
        )
        for key, identifiers in sorted(groups.items())
    ]


def reconcile_sources(
    canonical_sources: Iterable[CanonicalSource],
    index_documents: Iterable[dict[str, Any]],
    include_index_only: bool = True,
) -> list[SourceAuditResult]:
    canonical = sorted(canonical_sources, key=lambda source: (source.source_type, source.identity, source.manifest_key))
    indexed = group_index_documents(index_documents)
    by_url: dict[str, list[int]] = defaultdict(list)
    by_identity: dict[str, list[int]] = defaultdict(list)
    by_sourcefile: dict[str, list[int]] = defaultdict(list)
    by_category: dict[str, list[int]] = defaultdict(list)
    for position, source in enumerate(indexed):
        if normalized_url := normalize_url(source.url):
            by_url[normalized_url].append(position)
        by_identity[source.identity].append(position)
        by_sourcefile[normalize_label(source.sourcefile)].append(position)
        by_category[normalize_label(source.category)].append(position)

    canonical_url_counts: dict[str, int] = defaultdict(int)
    canonical_identity_counts: dict[str, int] = defaultdict(int)
    for source in canonical:
        if normalized_url := normalize_url(source.url):
            canonical_url_counts[normalized_url] += 1
        canonical_identity_counts[source.identity] += 1

    matched_index_positions: set[int] = set()
    results: list[SourceAuditResult] = []
    for source in canonical:
        normalized_url = normalize_url(source.url)
        candidates = by_url.get(normalized_url, []) if normalized_url else []
        if not candidates:
            candidates = by_identity.get(source.identity, [])
        if not candidates:
            sourcefile_candidates = by_sourcefile.get(normalize_label(source.sourcefile), [])
            if len(sourcefile_candidates) == 1:
                candidates = sourcefile_candidates
        if not candidates and source.source_type == "pdf":
            category_candidates = by_category.get(normalize_label(source.category), [])
            if len(category_candidates) == 1:
                candidates = category_candidates

        issues: list[str] = []
        if canonical_identity_counts[source.identity] > 1:
            issues.append("duplicate canonical identity")
        if normalized_url and canonical_url_counts[normalized_url] > 1:
            issues.append("duplicate canonical URL")

        if len(candidates) == 1:
            position = candidates[0]
            matched_index_positions.add(position)
            index_source = indexed[position]
            if source.identity != index_source.identity:
                issues.append("matched by URL or unique sourcefile; category/sourcefile differs")
            status = "WARN" if issues else "PASS"
            results.append(
                SourceAuditResult(
                    source_type=source.source_type,
                    sourcefile=source.sourcefile,
                    category=source.category,
                    canonical_url=source.url,
                    index_url=index_source.url,
                    index_document_ids=list(index_source.document_ids),
                    status=status,
                    issues=sorted(issues),
                )
            )
        elif len(candidates) > 1:
            issues.append("canonical source maps to multiple index source groups")
            results.append(
                SourceAuditResult(
                    source_type=source.source_type,
                    sourcefile=source.sourcefile,
                    category=source.category,
                    canonical_url=source.url,
                    index_url="",
                    index_document_ids=[],
                    status="UNMAPPED",
                    issues=sorted(issues),
                )
            )
        else:
            if not normalized_url:
                issues.append("canonical URL is unavailable or requires discovery")
            results.append(
                SourceAuditResult(
                    source_type=source.source_type,
                    sourcefile=source.sourcefile,
                    category=source.category,
                    canonical_url=source.url,
                    index_url="",
                    index_document_ids=[],
                    status="MISSING_FROM_INDEX",
                    issues=sorted(issues),
                )
            )

    if include_index_only:
        for position, source in enumerate(indexed):
            if position in matched_index_positions:
                continue
            results.append(
                SourceAuditResult(
                    source_type="index",
                    sourcefile=source.sourcefile,
                    category=source.category,
                    canonical_url="",
                    index_url=source.url,
                    index_document_ids=list(source.document_ids),
                    status="INDEX_ONLY",
                    issues=["index source does not match a canonical manifest entry"],
                )
            )

    return sorted(results, key=lambda result: (result.status, normalize_label(result.category), normalize_label(result.sourcefile)))


def build_report(
    results: Iterable[SourceAuditResult],
    snapshot_provenance: dict[str, Any] | None = None,
    *,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered = set_remediation_status(results)
    totals: dict[str, int] = defaultdict(int)
    dispositions: dict[str, int] = defaultdict(int)
    for result in ordered:
        totals[result.status] += 1
        if result.remediation_status:
            dispositions[result.remediation_status] += 1
    summary: dict[str, Any] = {"source_count": len(ordered), "statuses": dict(sorted(totals.items()))}
    if dispositions:
        summary["dispositions"] = dict(sorted(dispositions.items()))
    report = {
        "schema_version": 2,
        "summary": summary,
        "remediation": {
            status: sum(result.remediation_status == status for result in ordered)
            for status in sorted(REMEDIATION_STATUSES)
            if any(result.remediation_status == status for result in ordered)
        },
        "sources": [result.to_dict() for result in ordered],
    }
    if snapshot_provenance is not None:
        report["snapshot_provenance"] = snapshot_provenance
    if run_metadata:
        report.update(run_metadata)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Source Document Accuracy Audit",
        "",
        "This inventory audit is read-only. It does not upload, delete, re-embed, or modify index documents.",
        "",
        "## Summary",
        "",
    ]
    provenance = report.get("snapshot_provenance")
    if provenance:
        verification = "verified" if provenance.get("verified") else "unverified"
        lines.extend(
            [
                f"- **Snapshot**: {verification} ({provenance.get('format', 'unknown')})",
                f"- **Search target**: {provenance.get('service', 'unknown')} / {provenance.get('index', 'unknown')}",
                f"- **Snapshot documents**: {provenance.get('document_count', 'unknown')}",
                "",
            ]
        )
    for status, count in report["summary"]["statuses"].items():
        lines.append(f"- **{status}**: {count}")
    lines.extend(
        [
            "",
            "## Sources",
            "",
                "| Status | Remediation | Type | Category | Source | Index documents | Source coverage | Issues |",
                "|---|---|---|---|---:|---:|---|---|",
        ]
    )
    for source in report["sources"]:
        coverage = source["metrics"].get("source_to_index_coverage")
        coverage_text = f"{coverage:.1%}" if coverage is not None else "-"
        issues = "; ".join(source["issues"]) or "-"
        lines.append(
            f"| {source['status']} | {source['remediation_status']} | {source['source_type']} | {source['category']} | "
            f"{source['sourcefile']} | {len(source['index_document_ids'])} | {coverage_text} | {issues} |"
        )
    notable = [source for source in report["sources"] if source["issues"] or source["evidence"]]
    if notable:
        lines.extend(["", "## Findings", ""])
        for source in notable:
            lines.append(f"### {source['sourcefile']}")
            lines.append("")
            for issue in source["issues"]:
                lines.append(f"- {issue}")
            if source["evidence"]:
                classifications: dict[str, int] = defaultdict(int)
                for evidence in source["evidence"]:
                    classifications[str(evidence.get("classification") or evidence.get("kind") or "other")] += 1
                summary = ", ".join(f"{key}: {value}" for key, value in sorted(classifications.items()))
                lines.append(f"- Low-coverage evidence: {len(source['evidence'])} pages ({summary})")
            lines.append("")
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = (
        (json_path, json.dumps(report, indent=2, sort_keys=True) + "\n"),
        (markdown_path, render_markdown(report)),
    )
    temporary_paths: list[Path] = []
    try:
        for output_path, content in outputs:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
            ) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            temporary_paths.append(temporary_path)
        for temporary_path, (output_path, _) in zip(temporary_paths, outputs):
            temporary_path.replace(output_path)
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-snapshot", type=Path, help="Use a cached JSON snapshot instead of querying Azure AI Search")
    parser.add_argument(
        "--allow-unverified-snapshot",
        action="store_true",
        help="Allow legacy/unverified snapshots for diagnostic-only runs",
    )
    parser.add_argument("--write-snapshot", type=Path, help="Save a provenance-bearing read-only index snapshot")
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="Azure AI Search service name or endpoint")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="Azure AI Search index name")
    parser.add_argument("--family", choices=("all", "pdf", "html"), default="all")
    parser.add_argument("--source", help="Audit one canonical sourcefile (case-insensitive)")
    parser.add_argument("--html-fidelity", action="store_true", help="Fetch and compare official web sources")
    parser.add_argument("--html-cache", type=Path, default=DEFAULT_HTML_CACHE, help="Per-URL HTML response checkpoint JSON")
    parser.add_argument("--checkpoint", type=Path, help="Atomic audit progress checkpoint")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    run_id = hashlib.sha256(f"{started_at_utc}:{os.getpid()}".encode("utf-8")).hexdigest()[:16]
    sources = load_pdf_sources() + load_web_sources()
    source_identity_digest = hashlib.sha256(
        json.dumps(sorted(source.identity for source in sources), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if args.family != "all":
        sources = [source for source in sources if source.source_type == args.family]
    if args.source:
        requested_source = normalize_label(args.source)
        sources = [source for source in sources if normalize_label(source.sourcefile) == requested_source]
        if not sources:
            raise SystemExit(f"No canonical sourcefile matched {args.source!r}")

    if args.index_snapshot:
        documents, snapshot_provenance = load_index_snapshot(args.index_snapshot)
        if not snapshot_provenance.get("verified", False) and not args.allow_unverified_snapshot:
            raise SystemExit(
                "Refusing unverified index snapshot; use a provenance envelope or "
                "--allow-unverified-snapshot for diagnostic-only analysis"
            )
    else:
        documents = fetch_live_index_documents(args.service, args.index)
        snapshot = serialize_index_snapshot(documents, args.service, args.index)
        snapshot_provenance = {key: snapshot[key] for key in snapshot if key != "documents"} | {
            "verified": True,
            "format": "live_query",
        }
    if args.write_snapshot:
        snapshot_provenance = write_index_snapshot(args.write_snapshot, documents, args.service, args.index)

    results = reconcile_sources(sources, documents, include_index_only=args.family == "all" and not args.source)
    if args.checkpoint:
        write_checkpoint(
            args.checkpoint,
            {
                "schema_version": 1,
                "run_id": run_id,
                "phase": "reconciled",
                "source_identity_digest": source_identity_digest,
                "snapshot_provenance": snapshot_provenance,
                "processed_source_count": len(results),
            },
        )
    results = apply_pdf_fidelity(results, sources, documents)
    if args.checkpoint:
        write_checkpoint(
            args.checkpoint,
            {
                "schema_version": 1,
                "run_id": run_id,
                "phase": "pdf_fidelity",
                "source_identity_digest": source_identity_digest,
                "snapshot_provenance": snapshot_provenance,
                "processed_source_count": len(results),
            },
        )
    if args.html_fidelity:
        results = apply_html_fidelity(results, sources, documents, cache_path=args.html_cache)
        if args.checkpoint:
            write_checkpoint(
                args.checkpoint,
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "phase": "html_fidelity",
                    "source_identity_digest": source_identity_digest,
                    "snapshot_provenance": snapshot_provenance,
                    "processed_source_count": len(results),
                },
            )
    report = build_report(
        results,
        snapshot_provenance,
        run_metadata={
            "run_id": run_id,
            "started_at_utc": started_at_utc,
            "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "complete": True,
            "expected_source_count": len(sources),
            "processed_source_count": len(results),
            "source_identity_digest": source_identity_digest,
            "json_output": str(args.json_output),
            "markdown_output": str(args.markdown_output),
        },
    )
    write_report(report, args.json_output, args.markdown_output)
    return 1 if any(source["status"] in TERMINAL_FAILURE_STATUSES for source in report["sources"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())