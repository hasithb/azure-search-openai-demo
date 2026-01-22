#!/usr/bin/env python3
"""
Comprehensive coverage test for court guide subsection identification.

Goal: Achieve 100% accuracy on identifying ALL subsection tokens that
appear in a court guide chunk/section. This script compares:
- SubsectionExtractor.extract_all_subsections(content)
vs
- A broad regex sweep of all subsection-like tokens in the content.

Outputs a summary and a JSON report with missing/extra tokens per guide.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

# Add app/backend to path for customizations
ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "app" / "backend"
import sys
sys.path.insert(0, str(BACKEND_DIR))

from customizations.subsection_extractor import SubsectionExtractor


TOKEN_PATTERN = re.compile(
    r"\b("
    r"[A-Z]\.\d+(?:\.\d+)?[A-Z]?"  # A.1, B.7, C.2.3, 5.1A
    r"|[A-Z]\d+\.\d+(?:\.\d+)?[A-Z]?"  # A4.1, B2.3, A1.1A
    r"|\d+\.\d+(?:\.\d+)?[A-Z]?"  # 1.1, 2.3, 1.2.3, 7.3A
    r"|Rule\s+\d+(?:\.\d+)?"  # Rule 3.1
    r"|Para\s+\d+(?:\.\d+)?"  # Para 5.2
    r")\b",
    re.IGNORECASE,
)


@dataclass
class GuideStats:
    total_docs: int = 0
    total_tokens: int = 0
    total_extracted: int = 0
    missing_tokens: int = 0
    extra_tokens: int = 0
    docs_with_missing: int = 0
    docs_with_extra: int = 0


@dataclass
class DocResult:
    doc_id: str
    sourcepage: str
    sourcefile: str
    expected_tokens: list[str]
    extracted_tokens: list[str]
    missing_tokens: list[str]
    extra_tokens: list[str]
    content_preview: str


def normalize_token(token: str) -> str:
    normalized = token.replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized.strip())
    return normalized.rstrip(".: ").upper()


def extract_expected_tokens(content: str) -> list[str]:
    if not content:
        return []
    tokens = [normalize_token(m.group(0)) for m in TOKEN_PATTERN.finditer(content)]
    # Preserve order but unique
    seen = set()
    ordered = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def build_equivalent_set(tokens: list[str]) -> set[str]:
    result = set(tokens)
    for token in list(result):
        match = re.match(r"^(RULE|PARA)\s+(\d+(?:\.\d+)?[A-Z]?)$", token, re.IGNORECASE)
        if match:
            result.add(match.group(2))
    return result


def load_documents(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def analyze_guide(path: Path, max_docs: int | None) -> tuple[GuideStats, list[DocResult]]:
    stats = GuideStats()
    doc_results: list[DocResult] = []

    for idx, doc in enumerate(load_documents(path)):
        if max_docs is not None and idx >= max_docs:
            break
        content = (doc.get("content") or "").strip()
        extracted = SubsectionExtractor.extract_all_subsections(content)
        expected = extract_expected_tokens(content)

        extracted_norm = [normalize_token(t) for t in extracted]
        expected_norm = [normalize_token(t) for t in expected]

        expected_set = build_equivalent_set(expected_norm)
        extracted_set = build_equivalent_set(extracted_norm)

        missing = [t for t in expected_set if t not in extracted_set]
        extra = [t for t in extracted_set if t not in expected_set]

        stats.total_docs += 1
        stats.total_tokens += len(expected_norm)
        stats.total_extracted += len(extracted_norm)
        stats.missing_tokens += len(missing)
        stats.extra_tokens += len(extra)
        stats.docs_with_missing += 1 if missing else 0
        stats.docs_with_extra += 1 if extra else 0

        if missing or extra:
            preview = content[:240].replace("\n", " ")
            doc_results.append(
                DocResult(
                    doc_id=str(doc.get("id", "")),
                    sourcepage=str(doc.get("sourcepage", "")),
                    sourcefile=str(doc.get("sourcefile", "")),
                    expected_tokens=expected_norm,
                    extracted_tokens=extracted_norm,
                    missing_tokens=missing,
                    extra_tokens=extra,
                    content_preview=preview,
                )
            )

    return stats, doc_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Court guide subsection coverage test")
    parser.add_argument(
        "--input-dir",
        default=str(ROOT / "court_guides_processing_pipeline" / "outputs"),
        help="Directory containing *_processed.json files",
    )
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap per guide")
    parser.add_argument(
        "--output",
        default=str(ROOT / "evals" / "results" / "court_guide_subsection_coverage.json"),
        help="Output JSON report path",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "summary": {},
        "guides": {},
    }

    total = GuideStats()

    files = sorted(input_dir.glob("*_processed.json"))
    if not files:
        print(f"No processed JSONs found in {input_dir}")
        return 1

    for path in files:
        stats, doc_results = analyze_guide(path, args.max_docs)
        name = path.name

        report["guides"][name] = {
            "stats": asdict(stats),
            "examples": [asdict(r) for r in doc_results[:50]],
        }

        total.total_docs += stats.total_docs
        total.total_tokens += stats.total_tokens
        total.total_extracted += stats.total_extracted
        total.missing_tokens += stats.missing_tokens
        total.extra_tokens += stats.extra_tokens
        total.docs_with_missing += stats.docs_with_missing
        total.docs_with_extra += stats.docs_with_extra

        missing_rate = (stats.missing_tokens / stats.total_tokens * 100) if stats.total_tokens else 0.0
        print(
            f"{name}: docs={stats.total_docs} tokens={stats.total_tokens} "
            f"missing={stats.missing_tokens} ({missing_rate:.1f}%) extras={stats.extra_tokens}"
        )

    report["summary"] = asdict(total)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
