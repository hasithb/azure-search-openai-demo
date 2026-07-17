#!/usr/bin/env python3
"""Compare the current prompt against a low-risk injected answer-prompt guardrail.

This script does not modify app behavior. It sends the same request twice:
1. Current production prompt
2. Current production prompt plus a small injected instruction block

The goal is to test option 3 from the investigation: extract low-risk prompt ideas
from the experimental planner work without changing retrieval architecture.
"""

from __future__ import annotations

import json
import re
import sys

import httpx


BASE_URL = "http://localhost:50505"

PROMPT_INJECTION = ">>>Prompt-only experimental instructions:\n" \
    "- If the retrieved sources do not include the likely primary authority for the user's question, say so explicitly instead of answering from a tangentially related source.\n" \
    "- For abbreviations, shorthand, or acronym-like terms, do not expand them from similarly named documents unless a retrieved source clearly supports that interpretation.\n" \
    "- When the retrieved material is court-specific or secondary only, say that clearly and recommend the more authoritative CPR or Practice Direction source or a refined query.\n" \
    "- Do not answer acronym collisions, especially between pre-action disclosure and pre-action protocols, unless the retrieved sources clearly support the expansion."


TEST_CASES = [
    {
        "name": "PAD",
        "question": "What is PAD?",
        "checks": [
            {
                "type": "contains_any",
                "terms": ["CPR 31.16", "pre-action disclosure", "do not directly define", "primary authority"],
                "description": "Names PAD authority or explicitly states the source gap",
            },
            {
                "type": "not_contains_any",
                "terms": [
                    "Practice Direction – Pre-Action Conduct and Protocols",
                    "PAD appears to mean the **Practice Direction",
                    "PAD most likely means the Practice Direction",
                ],
                "description": "Does not answer PAD as pre-action protocols",
            },
        ],
    },
    {
        "name": "Standard Disclosure",
        "question": "What is standard disclosure?",
        "checks": [
            {"type": "contains_any", "terms": ["Part 31", "CPR 31", "31.6", "court-specific"], "description": "Identifies the general CPR position or states court-specific limits"},
        ],
    },
    {
        "name": "Broad Time Limits",
        "question": "What are the time limits?",
        "checks": [
            {"type": "contains_any", "terms": ["specific", "which", "context", "time limits", "days"], "description": "Clarifies the scope of the broad query"},
        ],
    },
    {
        "name": "Commercial Court with CPR Filter",
        "question": "How does the Commercial Court handle case management conferences?",
        "overrides": {"include_category": "Civil Procedure Rules and Practice Directions"},
        "checks": [
            {"type": "contains_any", "terms": ["Commercial Court Guide", "do not directly", "court-specific", "Part 58"], "description": "Notes the CPR-only mismatch and recommends a better source"},
        ],
    },
    {
        "name": "Weather Out of Scope",
        "question": "What is the weather forecast for London?",
        "checks": [
            {"type": "contains_any", "terms": ["do not cover", "not available", "do not contain", "weather"], "description": "Refuses unsupported out-of-scope requests"},
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
    baseline_total = 0
    baseline_passed = 0
    injected_total = 0
    injected_passed = 0
    results_log: list[dict[str, object]] = []

    print("=" * 88)
    print("PROMPT INJECTION COMPARISON")
    print("=" * 88)

    for index, test_case in enumerate(TEST_CASES, 1):
        base_overrides = merge_overrides(test_case.get("overrides"))
        injected_overrides = dict(base_overrides)
        injected_overrides["prompt_template"] = PROMPT_INJECTION

        print(f"\n{'=' * 88}")
        print(f"[{index}/{len(TEST_CASES)}] {test_case['name']}")
        print(f"Q: {test_case['question']}")

        baseline = send_chat(str(test_case["question"]), base_overrides)
        injected = send_chat(str(test_case["question"]), injected_overrides)
        baseline_answer = baseline.get("message", {}).get("content", "")
        injected_answer = injected.get("message", {}).get("content", "")

        baseline_results = []
        injected_results = []
        for check in test_case["checks"]:
            baseline_ok, description = run_check(check, baseline_answer)
            injected_ok, _ = run_check(check, injected_answer)
            baseline_total += 1
            injected_total += 1
            baseline_passed += int(baseline_ok)
            injected_passed += int(injected_ok)
            baseline_results.append({"description": description, "passed": baseline_ok})
            injected_results.append({"description": description, "passed": injected_ok})
            print(f"  {description}: current={'PASS' if baseline_ok else 'FAIL'} | injected={'PASS' if injected_ok else 'FAIL'}")

        print(f"Current answer:  {baseline_answer[:320].replace(chr(10), ' ')}")
        print(f"Injected answer: {injected_answer[:320].replace(chr(10), ' ')}")

        results_log.append(
            {
                "name": test_case["name"],
                "question": test_case["question"],
                "overrides": test_case.get("overrides", {}),
                "current": {"answer": baseline_answer, "checks": baseline_results},
                "injected": {"answer": injected_answer, "checks": injected_results},
            }
        )

    summary = {
        "current": {
            "passed_checks": baseline_passed,
            "total_checks": baseline_total,
            "pass_rate": round(baseline_passed / baseline_total, 3) if baseline_total else 0,
        },
        "injected": {
            "passed_checks": injected_passed,
            "total_checks": injected_total,
            "pass_rate": round(injected_passed / injected_total, 3) if injected_total else 0,
        },
        "results": results_log,
    }

    output_path = "scripts/prompt_injection_results.json"
    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 88}")
    print("SUMMARY")
    print(f"Current:  {baseline_passed}/{baseline_total} ({summary['current']['pass_rate']:.1%})")
    print(f"Injected: {injected_passed}/{injected_total} ({summary['injected']['pass_rate']:.1%})")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())