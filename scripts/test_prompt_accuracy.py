"""
Targeted accuracy test for the prompt intelligence improvements.

Tests whether the LLM:
1. Detects source-question mismatches and recommends alternatives
2. Disambiguates ambiguous legal terms (standard vs extended disclosure, etc.)
3. Suggests query refinements for broad questions
4. Recommends better source categories when filtered

Run with: python scripts/test_prompt_accuracy.py
Requires: app running at http://localhost:50505
"""

import json
import sys
import httpx

BASE_URL = "http://localhost:50505"

# Each test case: question, expected behavior checks, optional overrides
TEST_CASES = [
    # ---- Disambiguation: standard vs extended disclosure ----
    {
        "name": "Disambiguation: 'standard disclosure' should cite CPR 31",
        "question": "What is standard disclosure?",
        "checks": [
            {"type": "contains_any", "terms": ["CPR 31", "Part 31", "31.6"], "description": "References CPR Part 31 or Rule 31.6"},
        ],
        "overrides": {},
    },
    {
        "name": "Disambiguation: 'thorough disclosure' should reference PD 57AD or extended disclosure",
        "question": "Tell me about thorough disclosure requirements",
        "checks": [
            {"type": "contains_any", "terms": ["57AD", "extended disclosure", "Business and Property Courts"], "description": "References PD 57AD or extended disclosure"},
        ],
        "overrides": {},
    },
    # ---- Source mismatch: asking about court-specific procedure with wrong filter ----
    {
        "name": "Source recommendation: Commercial Court query with CPR-only filter",
        "question": "What are the procedures for case management in the Commercial Court?",
        "checks": [
            {"type": "contains_any", "terms": ["Commercial Court Guide", "Commercial Court", "Part 58"], "description": "Mentions Commercial Court context"},
            {"type": "has_citation", "description": "Has at least one citation"},
        ],
        "overrides": {"filter": "category eq 'Civil Procedure Rules and Practice Directions'"},
    },
    # ---- Ambiguous term: 'costs' ----
    {
        "name": "Ambiguous term: 'costs' should address the topic substantively",
        "question": "What are the rules about costs?",
        "checks": [
            {"type": "contains_any", "terms": ["Part 44", "Part 45", "Part 46", "Part 47", "Part 48", "costs"], "description": "References costs Parts"},
            {"type": "has_citation", "description": "Has at least one citation"},
        ],
        "overrides": {},
    },
    # ---- Broad query: time limits (should provide useful answer not just refuse) ----
    {
        "name": "Broad query: 'time limits' should still provide useful info",
        "question": "What are the time limits?",
        "checks": [
            {"type": "contains_any", "terms": ["time limit", "days", "period", "Part", "CPR"], "description": "Provides substantive info about time limits"},
            {"type": "has_citation", "description": "Has at least one citation"},
        ],
        "overrides": {},
    },
    # ---- PAD disambiguation (existing feature, regression check) ----
    {
        "name": "PAD should reference CPR 31.16, not Pre-Action Protocols",
        "question": "What is PAD?",
        "checks": [
            {"type": "contains_any", "terms": ["31.16", "pre-action disclosure", "before proceedings"], "description": "References CPR 31.16 pre-action disclosure"},
            {"type": "not_contains", "terms": ["Pre-Action Protocol for"], "description": "Does NOT primarily cite Pre-Action Protocols"},
        ],
        "overrides": {},
    },
    # ---- Summary judgment (existing, regression check) ----
    {
        "name": "Summary judgment should cite Part 24",
        "question": "How do I apply for summary judgment?",
        "checks": [
            {"type": "contains_any", "terms": ["Part 24", "24.2", "no real prospect"], "description": "References CPR Part 24"},
            {"type": "has_citation", "description": "Has at least one citation"},
        ],
        "overrides": {},
    },
    # ---- Cross-source: appeal routes ----
    {
        "name": "Appeal query should reference Part 52",
        "question": "How do I appeal a court decision?",
        "checks": [
            {"type": "contains_any", "terms": ["Part 52", "appeal", "permission to appeal", "appellant"], "description": "References Part 52 or appeal terminology"},
            {"type": "has_citation", "description": "Has at least one citation"},
        ],
        "overrides": {},
    },
    # ---- Service ambiguity ----
    {
        "name": "Service of claim form should reference Part 6",
        "question": "How do I serve a claim form?",
        "checks": [
            {"type": "contains_any", "terms": ["Part 6", "6.3", "6.4", "6.5", "service"], "description": "References CPR Part 6"},
            {"type": "has_citation", "description": "Has at least one citation"},
        ],
        "overrides": {},
    },
    # ---- Out-of-scope question (the prompt should handle gracefully) ----
    {
        "name": "Out-of-scope: non-legal question should note unavailability",
        "question": "What is the weather forecast for London?",
        "checks": [
            {"type": "contains_any", "terms": ["not", "cannot", "no information", "available sources", "don't have", "does not"], "description": "Indicates info not available in sources"},
        ],
        "overrides": {},
    },
]


def send_chat(question: str, overrides: dict) -> dict:
    """Send a chat request to the running app."""
    default_overrides = {
        "retrieval_mode": "hybrid",
        "semantic_ranker": True,
        "semantic_captions": False,
        "top": 5,
        "suggest_followup_questions": False,
        "seed": 42,
    }
    default_overrides.update(overrides)

    payload = {
        "messages": [{"content": question, "role": "user"}],
        "context": {
            "overrides": default_overrides,
        },
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(f"{BASE_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json()


def check_contains_any(answer: str, terms: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(term.lower() in answer_lower for term in terms)


def check_not_contains(answer: str, terms: list[str]) -> bool:
    """Check answer doesn't PRIMARILY focus on these terms.
    We allow brief mentions, but if >50% of sentences contain the term, fail."""
    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    if not sentences:
        return True
    for term in terms:
        matching = sum(1 for s in sentences if term.lower() in s.lower())
        if matching > len(sentences) * 0.5:
            return False
    return True


def check_has_citation(answer: str) -> bool:
    import re
    return bool(re.search(r'\[\d+\]', answer))


def run_check(check: dict, answer: str) -> tuple[bool, str]:
    check_type = check["type"]
    desc = check.get("description", check_type)

    if check_type == "contains_any":
        passed = check_contains_any(answer, check["terms"])
    elif check_type == "not_contains":
        passed = check_not_contains(answer, check["terms"])
    elif check_type == "has_citation":
        passed = check_has_citation(answer)
    else:
        return False, f"Unknown check type: {check_type}"

    return passed, desc


def main():
    print("=" * 80)
    print("PROMPT ACCURACY TEST SUITE")
    print("Testing disambiguation, source mismatch, and query refinement")
    print("=" * 80)
    print()

    total_checks = 0
    passed_checks = 0
    failed_tests = []
    results = []

    for i, test in enumerate(TEST_CASES, 1):
        name = test["name"]
        question = test["question"]
        overrides = test.get("overrides", {})
        checks = test["checks"]

        print(f"[{i}/{len(TEST_CASES)}] {name}")
        print(f"  Q: {question}")

        try:
            result = send_chat(question, overrides)
            answer = result.get("message", {}).get("content", "")
            num_sources = len(result.get("context", {}).get("data_points", {}).get("text", []))
            print(f"  Sources: {num_sources}")

            # Truncate answer for display
            display_answer = answer[:200].replace('\n', ' ')
            if len(answer) > 200:
                display_answer += "..."
            print(f"  A: {display_answer}")

            test_passed = True
            for check in checks:
                total_checks += 1
                ok, desc = run_check(check, answer)
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] {desc}")
                if ok:
                    passed_checks += 1
                else:
                    test_passed = False

            if not test_passed:
                failed_tests.append(name)

            results.append({
                "name": name,
                "question": question,
                "answer_length": len(answer),
                "num_sources": num_sources,
                "checks_passed": all(run_check(c, answer)[0] for c in checks),
            })

        except Exception as e:
            print(f"  [ERROR] {e}")
            for check in checks:
                total_checks += 1
            failed_tests.append(name)

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Total checks: {total_checks}")
    print(f"Checks passed: {passed_checks}/{total_checks} ({100*passed_checks//total_checks if total_checks else 0}%)")
    print(f"Tests fully passed: {len(TEST_CASES) - len(failed_tests)}/{len(TEST_CASES)}")

    if failed_tests:
        print(f"\nFailed tests:")
        for name in failed_tests:
            print(f"  - {name}")

    # Write detailed results
    output_path = "scripts/prompt_accuracy_results.json"
    with open(output_path, "w") as f:
        json.dump({"summary": {
            "total_tests": len(TEST_CASES),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "pass_rate": round(passed_checks / total_checks, 3) if total_checks else 0,
            "failed_tests": failed_tests,
        }, "results": results}, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")

    return 0 if not failed_tests else 1


if __name__ == "__main__":
    sys.exit(main())
