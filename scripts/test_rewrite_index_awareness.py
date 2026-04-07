#!/usr/bin/env python3
"""A/B comparison: current rewrite prompt vs. index-aware rewrite prompt.

This script tests whether giving the rewrite tool more knowledge about the index
structure and the ability to suggest a category filter improves retrieval quality.

It sends the same questions through two paths:
1. Current production rewrite (no changes)
2. Enhanced rewrite with index-awareness overrides injected via prompt_template

The enhanced path adds:
- Index field awareness (category, sourcepage, sourcefile, subsection_id)
- Category key → display name mapping so the tool can suggest filters
- Sourcepage structure patterns
- A "recommended_category" tool parameter hint in the prompt

Both paths hit the live app at localhost:50505.
"""

from __future__ import annotations

import json
import re
import sys

import httpx


BASE_URL = "http://localhost:50505"

# Index-awareness prompt injection: adds structural knowledge about the search index
# to the rewrite prompt so the LLM can write more targeted queries.
INDEX_AWARENESS_INJECTION = """>>>Enhanced index-awareness instructions:

IMPORTANT - Index structure knowledge:
The Azure AI Search index has these key searchable/filterable fields:
- content: The main text of each chunk (searchable, used for full-text and vector search)
- category: The document collection/source family (filterable, facetable). Examples of category values:
  "Civil Procedure Rules and Practice Directions", "Commercial Court", "Chancery Division",
  "King's Bench Division", "Technology and Construction Court", "Patents Court",
  "Pre-Action Protocols", "Court of Appeal Civil Division", "Circuit Commercial Court",
  "Senior Courts Costs Office"
- sourcepage: Section or page label (filterable). Examples: "Case management (p. 141)",
  "Part 31 - Disclosure and inspection of documents", "PD44 - General rules about costs",
  "Annex 1 - Guidance on witness statements (p. 73)", "Pre-Action Protocol for Personal Injury Claims"
- sourcefile: Source document filename (filterable). Usually matches the category or specific Part name.
- subsection_id: Precise subsection within a chunk (e.g., "1.1", "A4.1", "Para 1.1")

The semantic ranker uses sourcepage as the title field and content as the content field.
This means sourcepage has high influence on semantic ranking — queries that match
sourcepage patterns are boosted.

IMPORTANT - Targeted query strategies:
1. For EXACT RULE lookups (e.g., "What does CPR 3.9 say?"):
   - Use the specific Part number AND rule number in the search query
   - Include the rule's topic keywords for semantic ranking
   - Example: "CPR 3.9 relief from sanctions power of court"

2. For BROAD CONCEPT queries (e.g., "How do costs work?"):
   - Include the most likely CPR Part or Practice Direction number
   - Add distinctive terminology from that source
   - Example: "costs assessment Part 44 basis of assessment standard indemnity"

3. For COURT-SPECIFIC queries (e.g., "Commercial Court case management"):
   - The category field can filter to specific court guides
   - Target the court guide's section structure in your query terms
   - Example: "case management conference Commercial Court" (this matches sourcepage patterns)

4. For CROSS-SOURCE queries (e.g., "How do different courts handle disclosure?"):
   - Write a query targeting the general CPR rule first
   - Do NOT narrow to a single category — let the search retrieve from multiple sources

When you identify that a question clearly targets a specific source family, note this
in your legal_concept_analysis. The search system may use this to pre-filter results.
"""


TEST_CASES = [
    {
        "name": "Exact CPR Rule Lookup",
        "question": "What does CPR 3.9 say about relief from sanctions?",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["3.9", "relief from sanctions"],
                "description": "Identifies CPR 3.9 as the authority",
            },
            {
                "type": "has_citation",
                "description": "Includes numbered citations",
            },
        ],
    },
    {
        "name": "Court-Specific Query",
        "question": "How does the Commercial Court handle case management conferences?",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["Commercial Court Guide", "Commercial Court"],
                "description": "Cites the Commercial Court Guide",
            },
            {
                "type": "has_citation",
                "description": "Includes numbered citations",
            },
        ],
    },
    {
        "name": "Sourcepage-Pattern Query",
        "question": "What are the rules on disclosure and inspection of documents?",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["Part 31", "CPR 31", "disclosure"],
                "description": "Targets Part 31 disclosure rules",
            },
        ],
    },
    {
        "name": "PAD Disambiguation",
        "question": "What is PAD?",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["CPR 31.16", "pre-action disclosure", "disclosure before proceedings"],
                "description": "Correctly identifies PAD as CPR 31.16",
            },
            {
                "type": "not_contains_any",
                "terms": [
                    "Practice Direction – Pre-Action Conduct",
                    "PAD appears to mean the **Practice Direction",
                    "PAD most likely means the Practice Direction",
                ],
                "description": "Does not confuse PAD with Pre-Action Protocols",
            },
        ],
    },
    {
        "name": "Practice Direction Lookup",
        "question": "What does Practice Direction 44 say about costs?",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["PD44", "PD 44", "Practice Direction 44", "Part 44"],
                "description": "Identifies PD44 or Part 44 costs rules",
            },
            {
                "type": "has_citation",
                "description": "Includes numbered citations",
            },
        ],
    },
    {
        "name": "Cross-Source Conceptual",
        "question": "How do different courts approach witness statements?",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["Part 32", "witness statement", "CPR 32"],
                "description": "References CPR Part 32 or witness statement rules",
            },
        ],
    },
    {
        "name": "Chancery Guide Specific",
        "question": "What is the Chancery Guide's approach to trial bundles?",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["Chancery Guide", "trial bundle", "bundle"],
                "description": "Cites the Chancery Guide on bundles",
            },
            {
                "type": "has_citation",
                "description": "Includes numbered citations",
            },
        ],
    },
    {
        "name": "Summary Judgment CPR Part 24",
        "question": "What is the test for summary judgment?",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["Part 24", "CPR 24", "no real prospect", "summary judgment"],
                "description": "Identifies CPR Part 24 as the authority",
            },
        ],
    },
    {
        "name": "Subsection Precision",
        "question": "What does rule 1.1 of the CPR say?",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["1.1", "overriding objective", "just", "proportionate"],
                "description": "Targets CPR 1.1 overriding objective",
            },
        ],
    },
    {
        "name": "Category-Filtered Cross-Reference",
        "question": "What do the Patents Court rules say about disclosure?",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["Patents Court", "disclosure"],
                "description": "Cites Patents Court Guide on disclosure",
            },
        ],
    },
    {
        "name": "Broad Query Needing Narrowing",
        "question": "What are the time limits?",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["specific", "which", "context", "time limit", "days", "Part"],
                "description": "Narrows or clarifies the broad scope",
            },
        ],
    },
    {
        "name": "Out of Scope",
        "question": "What is the weather forecast for London?",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["do not cover", "not available", "do not contain", "outside", "weather", "cannot"],
                "description": "Refuses unsupported out-of-scope requests",
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


def extract_sources(response: dict) -> list[str]:
    """Extract source citations from a chat response for deeper comparison."""
    data_points = response.get("context", {}).get("data_points", {})
    text_sources = data_points.get("text", [])
    sources = []
    for source in text_sources:
        if isinstance(source, dict):
            label = source.get("sourcepage", source.get("category", "unknown"))
            sources.append(label)
        elif isinstance(source, str):
            sources.append(source[:80])
    return sources


def extract_thoughts(response: dict) -> dict[str, str]:
    """Extract key thought steps from the response for comparison."""
    thoughts = response.get("context", {}).get("thoughts", [])
    result = {}
    for thought in thoughts:
        if isinstance(thought, dict):
            title = thought.get("title", "")
            description = thought.get("description", "")
            if "search query" in title.lower() or "rewrite" in title.lower():
                result["rewrite"] = description[:500]
            elif "search results" in title.lower():
                result["results"] = description[:500]
    return result


def run_check(check: dict[str, object], answer: str) -> tuple[bool, str]:
    lowered = answer.lower()
    check_type = str(check["type"])
    description = str(check["description"])
    if check_type == "contains_any":
        passed = any(str(term).lower() in lowered for term in check["terms"])
    elif check_type == "not_contains_any":
        passed = all(str(term).lower() not in lowered for term in check["terms"])
    elif check_type == "has_citation":
        passed = bool(re.search(r"\[\d+\]", answer))
    else:
        passed = False
    return passed, description


def main() -> int:
    current_total = 0
    current_passed = 0
    enhanced_total = 0
    enhanced_passed = 0
    results_log: list[dict[str, object]] = []

    print("=" * 88)
    print("REWRITE INDEX-AWARENESS A/B COMPARISON")
    print("=" * 88)
    print(f"\nRunning {len(TEST_CASES)} test cases against {BASE_URL}")
    print("Comparing: Current rewrite prompt vs. Index-aware enhanced rewrite prompt\n")

    for index, test_case in enumerate(TEST_CASES, 1):
        current_overrides = merge_overrides(test_case.get("overrides"))
        enhanced_overrides = dict(current_overrides)
        enhanced_overrides["prompt_template"] = INDEX_AWARENESS_INJECTION

        print(f"{'=' * 88}")
        print(f"[{index}/{len(TEST_CASES)}] {test_case['name']}")
        print(f"Q: {test_case['question']}")

        try:
            current_response = send_chat(str(test_case["question"]), current_overrides)
            enhanced_response = send_chat(str(test_case["question"]), enhanced_overrides)
        except httpx.HTTPError as exc:
            print(f"  ERROR: {exc}")
            results_log.append({
                "name": test_case["name"],
                "question": test_case["question"],
                "error": str(exc),
            })
            continue

        current_answer = current_response.get("message", {}).get("content", "")
        enhanced_answer = enhanced_response.get("message", {}).get("content", "")

        current_sources = extract_sources(current_response)
        enhanced_sources = extract_sources(enhanced_response)

        current_thoughts = extract_thoughts(current_response)
        enhanced_thoughts = extract_thoughts(enhanced_response)

        current_results = []
        enhanced_results = []
        for check in test_case["checks"]:
            current_ok, description = run_check(check, current_answer)
            enhanced_ok, _ = run_check(check, enhanced_answer)
            current_total += 1
            enhanced_total += 1
            current_passed += int(current_ok)
            enhanced_passed += int(enhanced_ok)
            current_results.append({"description": description, "passed": current_ok})
            enhanced_results.append({"description": description, "passed": enhanced_ok})
            marker_current = "PASS" if current_ok else "FAIL"
            marker_enhanced = "PASS" if enhanced_ok else "FAIL"
            print(f"  {description}: current={marker_current} | enhanced={marker_enhanced}")

        print(f"  Current sources:  {current_sources[:3]}")
        print(f"  Enhanced sources: {enhanced_sources[:3]}")
        if current_thoughts.get("rewrite"):
            print(f"  Current rewrite:  {current_thoughts['rewrite'][:200]}")
        if enhanced_thoughts.get("rewrite"):
            print(f"  Enhanced rewrite: {enhanced_thoughts['rewrite'][:200]}")
        print(f"  Current answer:   {current_answer[:300].replace(chr(10), ' ')}")
        print(f"  Enhanced answer:  {enhanced_answer[:300].replace(chr(10), ' ')}")

        results_log.append({
            "name": test_case["name"],
            "question": test_case["question"],
            "overrides": test_case.get("overrides", {}),
            "current": {
                "answer": current_answer,
                "sources": current_sources,
                "thoughts": current_thoughts,
                "checks": current_results,
            },
            "enhanced": {
                "answer": enhanced_answer,
                "sources": enhanced_sources,
                "thoughts": enhanced_thoughts,
                "checks": enhanced_results,
            },
        })

    summary = {
        "current": {
            "passed_checks": current_passed,
            "total_checks": current_total,
            "pass_rate": round(current_passed / current_total, 3) if current_total else 0,
        },
        "enhanced": {
            "passed_checks": enhanced_passed,
            "total_checks": enhanced_total,
            "pass_rate": round(enhanced_passed / enhanced_total, 3) if enhanced_total else 0,
        },
        "results": results_log,
    }

    output_path = "scripts/rewrite_index_awareness_results.json"
    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 88}")
    print("SUMMARY")
    print(f"{'=' * 88}")
    print(f"Current:  {current_passed}/{current_total} ({summary['current']['pass_rate']:.1%})")
    print(f"Enhanced: {enhanced_passed}/{enhanced_total} ({summary['enhanced']['pass_rate']:.1%})")
    diff = enhanced_passed - current_passed
    if diff > 0:
        print(f"Enhanced wins by +{diff} checks")
    elif diff < 0:
        print(f"Current wins by +{abs(diff)} checks")
    else:
        print("Tied — no difference in check pass rates")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
