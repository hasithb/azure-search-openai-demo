"""Audit raw HTML oracle blocks through the production scraper and chunker.

This is deliberately offline: it consumes saved oracle snapshots and never
fetches or writes Azure data. It proves that the current transformation keeps
raw legal blocks and emits valid section-level chunk metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "reports" / "html_oracle_snapshots"
REPORT_PATH = ROOT / "reports" / "html_transition_audit.json"
sys.path.insert(0, str(ROOT / "scripts"))

from audit_source_documents import load_web_sources, normalize_legal_block  # noqa: E402
import update_cpr_index_v3 as updater  # noqa: E402
from html_schema_oracle import extract_legal_blocks  # noqa: E402


EMBEDDING_TOKEN_LIMIT = 8191


def _occurrences(needle: str, haystack: str) -> int:
    if not needle:
        return 0
    return haystack.count(needle)


def _token_ngram_coverage(needle: str, haystack: str, ngram_size: int = 5) -> float:
    needle_tokens = needle.split()
    haystack_tokens = set(haystack.split())
    if len(needle_tokens) < ngram_size:
        return 0.0
    ngrams = [tuple(needle_tokens[index : index + ngram_size]) for index in range(len(needle_tokens) - ngram_size + 1)]
    matched = sum(1 for ngram in ngrams if all(token in haystack_tokens for token in ngram))
    return matched / len(ngrams)


def _block_match(normalized: str, scraped_text: str, kind: str = "") -> tuple[bool, str, float, int]:
    exact_occurrences = _occurrences(normalized, scraped_text)
    if exact_occurrences:
        return True, "exact_substring", 1.0, exact_occurrences
    words = normalized.split()
    if 2 <= len(words) <= 10 and not re.search(r"[.!?;:]", normalized):
        scraped_words = Counter(scraped_text.split())
        matched_words = sum(min(count, scraped_words[word]) for word, count in Counter(words).items())
        table_score = matched_words / len(words)
        if table_score >= 0.8:
            return True, "table_token_coverage", table_score, 1
    if kind == "li" and words:
        scraped_words = set(scraped_text.split())
        list_score = sum(word in scraped_words for word in words) / len(words)
        if list_score == 1.0:
            return True, "list_token_coverage", list_score, 1
    if len(words) < 12:
        return False, "unmatched", 0.0, 0
    coverage = _token_ngram_coverage(normalized, scraped_text)
    return coverage >= 0.8, "token_ngram_coverage", coverage, 1


def _comparison_text(value: str) -> str:
    value = normalize_legal_block(value).casefold()
    value = re.sub(r"(?m)^#{1,6}\s*", "", value)
    value = re.sub(r"(?m)^\[[^\]]+\]\s+(?=\S)", "", value)
    value = re.sub(r"\s+\d+[A-Za-z]?\.\d+[A-Za-z]?$", "", value)
    value = re.sub(r"(?<=\w)\s+\d{1,2}(?=\s*[;,.])", "", value)
    value = re.sub(r"(?<=[;,.])\s+\d{1,2}$", "", value)
    return value.strip()


def _snapshot_action(snapshot: dict[str, Any], source: Any, actions: dict[str, dict[str, Any]], actions_by_url: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    requested_url = str(snapshot.get("requested_url") or "").rstrip("/")
    final_url = str(snapshot.get("final_url") or "").rstrip("/")
    action_entry = actions.get(snapshot.get("sourcefile")) or actions_by_url.get(requested_url) or actions_by_url.get(final_url)
    if action_entry is None and requested_url:
        action_entry = {
            "sourcefile": source.sourcefile,
            "azure_id": None,
            "url": requested_url,
            "section": "ORACLE",
        }
    if action_entry is None:
        return None
    if requested_url:
        action_entry = {
            **action_entry,
            "url": requested_url,
            "verified_snapshot_url": final_url or requested_url,
        }
    return {**action_entry, "sourcefile": source.sourcefile}


def audit_snapshot(snapshot: dict[str, Any], source: Any, action_entry: dict[str, Any]) -> dict[str, Any]:
    html = str(snapshot.get("html") or "")
    raw_blocks = extract_legal_blocks(html)
    soup = BeautifulSoup(html, "html.parser")
    scraped = updater.scrape_page(
        updater.requests.Session(),
        action_entry,
        prefetched_result=(soup, snapshot.get("final_url", action_entry["url"]), snapshot.get("redirect_count", 0)),
    )
    result: dict[str, Any] = {
        "identity": source.identity,
        "sourcefile": source.sourcefile,
        "status": "PASS",
        "raw_block_count": len(raw_blocks),
        "matched_block_count": 0,
        "missing_blocks": [],
        "chunk_count": 0,
        "oversized_chunks": [],
        "metadata_failures": [],
    }
    if scraped is None:
        result["status"] = "FAIL"
        result["metadata_failures"].append("production scraper returned no content")
        return result

    scraped_text = _comparison_text(str(scraped.get("content") or ""))
    observed_counts: Counter[tuple[str, str]] = Counter()
    for block in raw_blocks:
        normalized = _comparison_text(block.text)
        heading_occurrences = _occurrences(normalized, scraped_text)
        if block.schema == "heading" and heading_occurrences:
            matched, match_method, match_score, available_count = True, "heading_context", 1.0, heading_occurrences
        else:
            matched, match_method, match_score, available_count = _block_match(normalized, scraped_text, block.kind)
        occurrence_key = (block.kind, normalized)
        if matched and observed_counts[occurrence_key] < available_count:
            observed_counts[occurrence_key] += 1
            result["matched_block_count"] += 1
        else:
            result["missing_blocks"].append({
                "kind": block.kind,
                "schema": block.schema,
                "locator": block.locator,
                "text": block.text[:240],
                "match_method": match_method,
                "match_score": match_score,
            })

    docs = updater.build_index_docs(action_entry, scraped)
    result["chunk_count"] = len(docs)
    for document in docs:
        token_count = updater.LegalDocumentChunker(max_tokens=8000).count_tokens(document["content"])
        if token_count > EMBEDDING_TOKEN_LIMIT:
            result["oversized_chunks"].append({"id": document.get("id", ""), "token_count": token_count})
        for field in ("id", "content", "sourcefile", "sourcepage", "parent_id", "subsection_id", "subsections"):
            if field not in document:
                result["metadata_failures"].append(f"{document.get('id', '<unknown>')}: missing {field}")
        if document.get("sourcefile") != source.sourcefile:
            result["metadata_failures"].append(f"{document.get('id', '<unknown>')}: sourcefile mismatch")

    if result["missing_blocks"] or result["oversized_chunks"] or result["metadata_failures"]:
        result["status"] = "FAIL"
    result["raw_block_coverage"] = (
        result["matched_block_count"] / result["raw_block_count"] if result["raw_block_count"] else 1.0
    )
    return result


def run(snapshot_dir: Path = SNAPSHOT_DIR) -> dict[str, Any]:
    sources = {source.identity: source for source in load_web_sources()}
    actions = {entry["sourcefile"]: entry for entry in updater.ACTION_LIST}
    actions_by_url = {entry["url"].rstrip("/"): entry for entry in updater.ACTION_LIST}
    results: list[dict[str, Any]] = []
    for path in sorted(snapshot_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if snapshot.get("status") != "ok":
            continue
        source = sources.get(snapshot.get("identity"))
        action_entry = _snapshot_action(snapshot, source, actions, actions_by_url) if source is not None else None
        if source is None or action_entry is None:
            results.append({
                "identity": snapshot.get("identity", ""),
                "sourcefile": snapshot.get("sourcefile", ""),
                "status": "BLOCKED",
                "reason": "no matching production ACTION_LIST entry",
            })
            continue
        results.append(audit_snapshot(snapshot, source, action_entry))
    return {
        "snapshot_count": len(results),
        "passed_count": sum(result["status"] == "PASS" for result in results),
        "failed_count": sum(result["status"] == "FAIL" for result in results),
        "blocked_count": sum(result["status"] == "BLOCKED" for result in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = run(args.snapshot_dir)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, sort_keys=True))
    return 0 if report["failed_count"] == 0 and report["blocked_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())