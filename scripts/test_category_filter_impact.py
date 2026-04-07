#!/usr/bin/env python3
"""A/B comparison: test whether category-aware rewrite improves court-specific queries.

This script focuses specifically on queries that target a single source family
(e.g., "Commercial Court", "Chancery Guide") and tests whether explicitly filtering
by category at the API level produces better results than trusting the search alone.

It compares:
1. Current: no category filter (search finds results across all sources)
2. Category-Filtered: pre-filtered to the expected source family

This helps answer: "Would giving the rewrite tool the ability to suggest a category
filter actually improve retrieval for court-specific questions?"
"""

from __future__ import annotations

import json
import re
import sys

import httpx


BASE_URL = "http://localhost:50505"


# Test cases with expected primary source category
TEST_CASES = [
    {
        "name": "Commercial Court CMC",
        "question": "How does the Commercial Court handle case management conferences?",
        "expected_category": "Commercial Court",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["Commercial Court Guide", "Commercial Court"],
                "description": "Cites Commercial Court Guide",
            },
            {
                "type": "source_category_match",
                "category": "Commercial Court",
                "min_count": 2,
                "description": "At least 2 sources from Commercial Court",
            },
        ],
    },
    {
        "name": "Chancery Trial Bundles",
        "question": "What is the Chancery Guide's approach to trial bundles?",
        "expected_category": "Chancery Division",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["Chancery Guide", "Chancery"],
                "description": "Cites Chancery Guide",
            },
            {
                "type": "source_category_match",
                "category": "Chancery Division",
                "min_count": 2,
                "description": "At least 2 sources from Chancery Guide",
            },
        ],
    },
    {
        "name": "Patents Court Disclosure",
        "question": "What do the Patents Court rules say about disclosure?",
        "expected_category": "Patents Court",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["Patents Court"],
                "description": "Cites Patents Court Guide",
            },
            {
                "type": "source_category_match",
                "category": "Patents Court",
                "min_count": 2,
                "description": "At least 2 sources from Patents Court",
            },
        ],
    },
    {
        "name": "TCC Experts",
        "question": "How does the Technology and Construction Court handle expert evidence?",
        "expected_category": "Technology and Construction Court",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["Technology and Construction Court", "TCC"],
                "description": "Cites TCC Guide",
            },
            {
                "type": "source_category_match",
                "category": "Technology and Construction Court",
                "min_count": 1,
                "description": "At least 1 source from TCC",
            },
        ],
    },
    {
        "name": "KB Division Listing",
        "question": "What are the King's Bench Division Guide's provisions on listing?",
        "expected_category": "King's Bench Division",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["King's Bench", "KBD", "KB Division"],
                "description": "Cites KB Division Guide",
            },
            {
                "type": "source_category_match",
                "category": "King's Bench Division",
                "min_count": 2,
                "description": "At least 2 sources from KB Division",
            },
        ],
    },
    {
        "name": "CPR Appeal Time Limits (should NOT filter)",
        "question": "What are the time limits for filing an appeal under the CPR?",
        "expected_category": "Civil Procedure Rules and Practice Directions",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["Part 52", "CPR 52", "21 days", "appeal"],
                "description": "Identifies CPR Part 52 appeal rules",
            },
        ],
    },
    {
        "name": "Cross-Court Disclosure (should NOT filter)",
        "question": "How do the different court guides approach disclosure differently?",
        "expected_category": None,  # No filter — should be cross-source
        "checks": [
            {
                "type": "contains_any",
                "terms": ["disclosure", "Part 31", "court guide"],
                "description": "References disclosure across sources",
            },
        ],
    },
]


def merge_overrides(extra: dict[str, object] | None = None) -> dict[str, object]:
    overrides: dict[str, object] = {
        "retrieval_mode": "hybrid",
        "semantic_ranker": True,
        "semantic_captions": False,
        "top": 5,
        "suggest_followup_questions": False,
        "seed": 42,
    }
    if extra:
        overrides.update(extra)
    return overrides


def send_chat(question: str, overrides: dict[str, object]) -> dict:
    payload = {
        "messages": [{"content": question, "role": "user"}],
        "context": {"overrides": overrides},
    }
    with httpx.Client(timeout=90.0) as client:
        response = client.post(f"{BASE_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json()


def extract_source_details(response: dict) -> list[dict[str, str]]:
    """Extract source details from response for category analysis."""
    data_points = response.get("context", {}).get("data_points", {})
    text_sources = data_points.get("text", [])
    sources = []
    for source in text_sources:
        if isinstance(source, dict):
            sources.append({
                "category": source.get("category", ""),
                "sourcepage": source.get("sourcepage", ""),
                "sourcefile": source.get("sourcefile", ""),
            })
    return sources


def count_category_matches(sources: list[dict[str, str]], category: str) -> int:
    """Count how many sources match a given category."""
    return sum(1 for s in sources if s.get("category", "") == category)


def run_check(check: dict, answer: str, sources: list[dict[str, str]]) -> tuple[bool, str]:
    lowered = answer.lower()
    check_type = str(check["type"])
    description = str(check["description"])
    if check_type == "contains_any":
        passed = any(str(term).lower() in lowered for term in check["terms"])
    elif check_type == "not_contains_any":
        passed = all(str(term).lower() not in lowered for term in check["terms"])
    elif check_type == "has_citation":
        passed = bool(re.search(r"\[\d+\]", answer))
    elif check_type == "source_category_match":
        category = str(check.get("category", ""))
        min_count = int(check.get("min_count", 1))
        count = count_category_matches(sources, category)
        passed = count >= min_count
        description = f"{description} (found {count})"
    else:
        passed = False
    return passed, description


def main() -> int:
    unfiltered_total = 0
    unfiltered_passed = 0
    filtered_total = 0
    filtered_passed = 0
    results_log: list[dict[str, object]] = []

    print("=" * 88)
    print("CATEGORY-AWARE RETRIEVAL A/B COMPARISON")
    print("=" * 88)
    print(f"\nRunning {len(TEST_CASES)} test cases against {BASE_URL}")
    print("Comparing: No category filter vs. Expected category filter\n")

    for index, test_case in enumerate(TEST_CASES, 1):
        unfiltered_overrides = merge_overrides(test_case.get("overrides"))
        filtered_overrides = dict(unfiltered_overrides)
        if test_case.get("expected_category"):
            filtered_overrides["include_category"] = test_case["expected_category"]

        print(f"{'=' * 88}")
        print(f"[{index}/{len(TEST_CASES)}] {test_case['name']}")
        print(f"Q: {test_case['question']}")
        if test_case.get("expected_category"):
            print(f"  Filter: category eq '{test_case['expected_category']}'")
        else:
            print(f"  Filter: None (cross-source)")

        try:
            unfiltered_response = send_chat(str(test_case["question"]), unfiltered_overrides)
            if test_case.get("expected_category"):
                filtered_response = send_chat(str(test_case["question"]), filtered_overrides)
            else:
                filtered_response = unfiltered_response  # Same — no filter to apply
        except httpx.HTTPError as exc:
            print(f"  ERROR: {exc}")
            results_log.append({"name": test_case["name"], "error": str(exc)})
            continue

        unfiltered_answer = unfiltered_response.get("message", {}).get("content", "")
        filtered_answer = filtered_response.get("message", {}).get("content", "")

        unfiltered_sources = extract_source_details(unfiltered_response)
        filtered_sources = extract_source_details(filtered_response)

        unfiltered_categories = [s["category"] for s in unfiltered_sources]
        filtered_categories = [s["category"] for s in filtered_sources]

        unfiltered_results = []
        filtered_results = []
        for check in test_case["checks"]:
            unfiltered_ok, description = run_check(check, unfiltered_answer, unfiltered_sources)
            filtered_ok, filtered_desc = run_check(check, filtered_answer, filtered_sources)
            unfiltered_total += 1
            filtered_total += 1
            unfiltered_passed += int(unfiltered_ok)
            filtered_passed += int(filtered_ok)
            unfiltered_results.append({"description": description, "passed": unfiltered_ok})
            filtered_results.append({"description": filtered_desc, "passed": filtered_ok})
            print(f"  {description}: unfiltered={'PASS' if unfiltered_ok else 'FAIL'}")
            print(f"  {filtered_desc}: filtered={'PASS' if filtered_ok else 'FAIL'}")

        unfiltered_cats_summary = {cat: unfiltered_categories.count(cat) for cat in set(unfiltered_categories)}
        filtered_cats_summary = {cat: filtered_categories.count(cat) for cat in set(filtered_categories)}
        print(f"  Unfiltered categories: {unfiltered_cats_summary}")
        print(f"  Filtered categories:   {filtered_cats_summary}")
        print(f"  Unfiltered answer: {unfiltered_answer[:250].replace(chr(10), ' ')}")
        print(f"  Filtered answer:   {filtered_answer[:250].replace(chr(10), ' ')}")

        results_log.append({
            "name": test_case["name"],
            "question": test_case["question"],
            "expected_category": test_case.get("expected_category"),
            "unfiltered": {
                "answer": unfiltered_answer,
                "categories": unfiltered_cats_summary,
                "source_count": len(unfiltered_sources),
                "checks": unfiltered_results,
            },
            "filtered": {
                "answer": filtered_answer,
                "categories": filtered_cats_summary,
                "source_count": len(filtered_sources),
                "checks": filtered_results,
            },
        })

    summary = {
        "unfiltered": {
            "passed_checks": unfiltered_passed,
            "total_checks": unfiltered_total,
            "pass_rate": round(unfiltered_passed / unfiltered_total, 3) if unfiltered_total else 0,
        },
        "filtered": {
            "passed_checks": filtered_passed,
            "total_checks": filtered_total,
            "pass_rate": round(filtered_passed / filtered_total, 3) if filtered_total else 0,
        },
        "results": results_log,
    }

    output_path = "scripts/category_filter_results.json"
    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 88}")
    print("SUMMARY")
    print(f"{'=' * 88}")
    print(f"Unfiltered: {unfiltered_passed}/{unfiltered_total} ({summary['unfiltered']['pass_rate']:.1%})")
    print(f"Filtered:   {filtered_passed}/{filtered_total} ({summary['filtered']['pass_rate']:.1%})")
    diff = filtered_passed - unfiltered_passed
    if diff > 0:
        print(f"Category filtering wins by +{diff} checks")
    elif diff < 0:
        print(f"Unfiltered wins by +{abs(diff)} checks")
    else:
        print("Tied — no difference in check pass rates")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
