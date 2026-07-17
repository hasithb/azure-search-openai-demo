"""
A/B test for knowledge-grounded query rewrite vs baseline.

Tests the query rewrite step over an extensive dataset of legal queries
to measure whether the knowledge-grounded reasoning (legal_concept_analysis)
improves source targeting accuracy.

We evaluate TWO dimensions:
1. Query quality: Does the generated search query contain the expected CPR rule/Part/PD?
2. Source accuracy: Do the retrieved sources come from the expected source documents?

Run with: python scripts/test_query_rewrite_ab.py
Requires: app running at http://localhost:50505
"""

import json
import re
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:50505"

# ==================================================================================
# EXTENSIVE TEST DATASET - 40 queries spanning all source types
# Each test specifies:
#   - question: The user query
#   - expected_query_terms: Terms that SHOULD appear in the rewritten search query
#   - expected_source_patterns: Patterns that should match in retrieved source metadata
#   - avoid_source_patterns: Source patterns that indicate wrong retrieval
#   - difficulty: "easy" (exact CPR ref given), "medium" (common term), "hard" (ambiguous/confusable)
# ==================================================================================
TEST_DATASET = [
    # ---- HARD: Confusable concepts (the main motivation for this change) ----
    {
        "id": "H01",
        "question": "What is pre-action disclosure?",
        "expected_query_terms": ["31.16", "disclosure", "before proceedings"],
        "expected_source_patterns": ["Part 31"],
        "avoid_source_patterns": ["Pre-Action Protocol"],
        "difficulty": "hard",
        "category": "confusable",
    },
    {
        "id": "H02",
        "question": "Tell me about PAD requirements",
        "expected_query_terms": ["31.16"],
        "expected_source_patterns": ["Part 31"],
        "avoid_source_patterns": ["Pre-Action Protocol"],
        "difficulty": "hard",
        "category": "confusable",
    },
    {
        "id": "H03",
        "question": "When can I apply for pre-action disclosure?",
        "expected_query_terms": ["31.16", "disclosure", "before proceedings"],
        "expected_source_patterns": ["Part 31"],
        "avoid_source_patterns": ["Pre-Action Protocol"],
        "difficulty": "hard",
        "category": "confusable",
    },
    {
        "id": "H04",
        "question": "What are the conditions for getting documents before a case starts?",
        "expected_query_terms": ["31.16"],
        "expected_source_patterns": ["Part 31"],
        "avoid_source_patterns": ["Pre-Action Protocol"],
        "difficulty": "hard",
        "category": "confusable",
    },
    {
        "id": "H05",
        "question": "What is standard disclosure?",
        "expected_query_terms": ["31.6", "standard disclosure"],
        "expected_source_patterns": ["Part 31"],
        "avoid_source_patterns": ["57AD"],
        "difficulty": "hard",
        "category": "confusable",
    },
    {
        "id": "H06",
        "question": "Tell me about thorough disclosure requirements",
        "expected_query_terms": ["57AD", "extended disclosure"],
        "expected_source_patterns": ["57AD"],
        "avoid_source_patterns": [],
        "difficulty": "hard",
        "category": "confusable",
    },
    {
        "id": "H07",
        "question": "What are the pre-action steps I need to take?",
        "expected_query_terms": ["pre-action", "protocol"],
        "expected_source_patterns": ["Pre-Action"],
        "avoid_source_patterns": [],
        "difficulty": "hard",
        "category": "confusable",
    },
    {
        "id": "H08",
        "question": "How do I get a Norwich Pharmacal order?",
        "expected_query_terms": ["31.18"],
        "expected_source_patterns": ["Part 31"],
        "avoid_source_patterns": [],
        "difficulty": "hard",
        "category": "confusable",
    },

    # ---- MEDIUM: Common legal concepts (should target correct CPR Part) ----
    {
        "id": "M01",
        "question": "How do I apply for summary judgment?",
        "expected_query_terms": ["Part 24", "24"],
        "expected_source_patterns": ["Part 24"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M02",
        "question": "What is the test for summary judgment?",
        "expected_query_terms": ["24", "no real prospect"],
        "expected_source_patterns": ["Part 24"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M03",
        "question": "How do I appeal a court decision?",
        "expected_query_terms": ["Part 52", "52", "appeal"],
        "expected_source_patterns": ["Part 52"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M04",
        "question": "What is the time limit for filing an appeal?",
        "expected_query_terms": ["52", "appeal", "time"],
        "expected_source_patterns": ["Part 52"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M05",
        "question": "How do I get default judgment?",
        "expected_query_terms": ["Part 12", "12", "default"],
        "expected_source_patterns": ["Part 12"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M06",
        "question": "How do I apply for relief from sanctions?",
        "expected_query_terms": ["3.9", "relief", "sanctions"],
        "expected_source_patterns": ["Part 3"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M07",
        "question": "What are the rules about expert evidence?",
        "expected_query_terms": ["Part 35", "35", "expert"],
        "expected_source_patterns": ["Part 35"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M08",
        "question": "How do I serve a claim form?",
        "expected_query_terms": ["Part 6", "6", "service", "claim form"],
        "expected_source_patterns": ["Part 6"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M09",
        "question": "What are the costs budgeting rules?",
        "expected_query_terms": ["costs", "budget"],
        "expected_source_patterns": ["Part 3", "PD 3E", "3E"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M10",
        "question": "How do I get a freezing injunction?",
        "expected_query_terms": ["Part 25", "25", "freezing"],
        "expected_source_patterns": ["Part 25"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M11",
        "question": "What is the overriding objective?",
        "expected_query_terms": ["Part 1", "1.1", "overriding objective"],
        "expected_source_patterns": ["Part 1"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M12",
        "question": "How do I strike out a statement of case?",
        "expected_query_terms": ["3.4", "strike out"],
        "expected_source_patterns": ["Part 3"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M13",
        "question": "What are the rules on witness statements?",
        "expected_query_terms": ["Part 32", "32", "witness"],
        "expected_source_patterns": ["Part 32"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M14",
        "question": "How do I make a Part 36 offer?",
        "expected_query_terms": ["Part 36", "36", "offer"],
        "expected_source_patterns": ["Part 36"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M15",
        "question": "What is specific disclosure?",
        "expected_query_terms": ["31.12", "specific disclosure"],
        "expected_source_patterns": ["Part 31"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },
    {
        "id": "M16",
        "question": "When can I get an unless order?",
        "expected_query_terms": ["3.1", "unless order"],
        "expected_source_patterns": ["Part 3"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "cpr_part",
    },

    # ---- COURT GUIDE QUERIES ----
    {
        "id": "CG01",
        "question": "What is the procedure for urgent applications in the Commercial Court?",
        "expected_query_terms": ["Commercial Court", "urgent"],
        "expected_source_patterns": ["Commercial Court"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "court_guide",
    },
    {
        "id": "CG02",
        "question": "How does case management work in the Technology and Construction Court?",
        "expected_query_terms": ["Technology and Construction", "TCC", "case management"],
        "expected_source_patterns": ["Technology and Construction"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "court_guide",
    },
    {
        "id": "CG03",
        "question": "What are the filing requirements in the Chancery Division?",
        "expected_query_terms": ["Chancery"],
        "expected_source_patterns": ["Chancery"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "court_guide",
    },
    {
        "id": "CG04",
        "question": "What does the Patents Court Guide say about expert evidence?",
        "expected_query_terms": ["Patents Court", "expert"],
        "expected_source_patterns": ["Patents Court"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "court_guide",
    },
    {
        "id": "CG05",
        "question": "How do I file an appeal in the Court of Appeal Civil Division?",
        "expected_query_terms": ["Court of Appeal"],
        "expected_source_patterns": ["Court of Appeal"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "court_guide",
    },
    {
        "id": "CG06",
        "question": "What are the costs rules in the King's Bench Division?",
        "expected_query_terms": ["King's Bench"],
        "expected_source_patterns": ["King"],
        "avoid_source_patterns": [],
        "difficulty": "medium",
        "category": "court_guide",
    },

    # ---- ACRONYM / SHORTHAND QUERIES ----
    {
        "id": "A01",
        "question": "What is SJ?",
        "expected_query_terms": ["Part 24", "summary judgment"],
        "expected_source_patterns": ["Part 24"],
        "avoid_source_patterns": [],
        "difficulty": "hard",
        "category": "acronym",
    },
    {
        "id": "A02",
        "question": "When can I get RFS?",
        "expected_query_terms": ["3.9", "relief", "sanctions"],
        "expected_source_patterns": ["Part 3"],
        "avoid_source_patterns": [],
        "difficulty": "hard",
        "category": "acronym",
    },
    {
        "id": "A03",
        "question": "What happens at a CMC?",
        "expected_query_terms": ["case management conference"],
        "expected_source_patterns": ["Part 29"],
        "avoid_source_patterns": [],
        "difficulty": "hard",
        "category": "acronym",
    },
    {
        "id": "A04",
        "question": "What is ADR in civil proceedings?",
        "expected_query_terms": ["alternative dispute resolution"],
        "expected_source_patterns": [],
        "avoid_source_patterns": [],
        "difficulty": "hard",
        "category": "acronym",
    },

    # ---- BROAD / AMBIGUOUS QUERIES ----
    {
        "id": "B01",
        "question": "What are the rules about costs?",
        "expected_query_terms": ["Part 44", "costs"],
        "expected_source_patterns": ["Part 44", "Part 45", "Part 46"],
        "avoid_source_patterns": [],
        "difficulty": "hard",
        "category": "ambiguous",
    },
    {
        "id": "B02",
        "question": "Tell me about disclosure",
        "expected_query_terms": ["Part 31", "disclosure"],
        "expected_source_patterns": ["Part 31"],
        "avoid_source_patterns": [],
        "difficulty": "hard",
        "category": "ambiguous",
    },
    {
        "id": "B03",
        "question": "What are the time limits?",
        "expected_query_terms": ["time", "limit"],
        "expected_source_patterns": [],
        "avoid_source_patterns": [],
        "difficulty": "hard",
        "category": "ambiguous",
    },
    {
        "id": "B04",
        "question": "How does service work?",
        "expected_query_terms": ["Part 6", "service"],
        "expected_source_patterns": ["Part 6"],
        "avoid_source_patterns": [],
        "difficulty": "hard",
        "category": "ambiguous",
    },

    # ---- PRACTICE DIRECTION QUERIES ----
    {
        "id": "PD01",
        "question": "What does Practice Direction 57AD say about extended disclosure?",
        "expected_query_terms": ["57AD", "extended disclosure"],
        "expected_source_patterns": ["57AD"],
        "avoid_source_patterns": [],
        "difficulty": "easy",
        "category": "practice_direction",
    },
    {
        "id": "PD02",
        "question": "What is PD 3E about?",
        "expected_query_terms": ["3E", "costs management"],
        "expected_source_patterns": ["3E"],
        "avoid_source_patterns": [],
        "difficulty": "easy",
        "category": "practice_direction",
    },
]


def send_chat(question: str, overrides: dict | None = None) -> dict:
    """Send a chat request to the running app."""
    default_overrides = {
        "retrieval_mode": "hybrid",
        "semantic_ranker": True,
        "semantic_captions": False,
        "top": 5,
        "suggest_followup_questions": False,
        "seed": 42,
    }
    if overrides:
        default_overrides.update(overrides)

    payload = {
        "messages": [{"content": question, "role": "user"}],
        "context": {"overrides": default_overrides},
    }

    with httpx.Client(timeout=90.0) as client:
        response = client.post(f"{BASE_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json()


def extract_search_query(result: dict) -> str:
    """Extract the generated search query from thought steps."""
    thoughts = result.get("context", {}).get("thoughts", [])
    for t in thoughts:
        title = t.get("title", "")
        if "search" in title.lower() and "generated" in title.lower():
            desc = t.get("description", "")
            if isinstance(desc, str):
                return desc
    # Fallback: look for query_text in props
    for t in thoughts:
        props = t.get("props", {})
        if isinstance(props, dict):
            qt = props.get("query_text")
            if qt:
                return qt
    return ""


def extract_sources(result: dict) -> list[dict]:
    """Extract source metadata from response."""
    text_sources = result.get("context", {}).get("data_points", {}).get("text", [])
    sources = []
    for s in text_sources:
        if isinstance(s, dict):
            sources.append({
                "category": s.get("category", ""),
                "sourcepage": s.get("sourcepage", ""),
                "sourcefile": s.get("sourcefile", ""),
            })
    return sources


def check_query_terms(query: str, expected_terms: list[str]) -> tuple[int, int, list[str]]:
    """Check how many expected terms appear in the query. Returns (matched, total, missing)."""
    query_lower = query.lower()
    matched = 0
    missing = []
    for term in expected_terms:
        if term.lower() in query_lower:
            matched += 1
        else:
            missing.append(term)
    return matched, len(expected_terms), missing


def check_source_patterns(sources: list[dict], patterns: list[str]) -> tuple[bool, str]:
    """Check if any source matches any of the expected patterns."""
    if not patterns:
        return True, "no patterns to check"

    all_source_text = " ".join(
        f"{s.get('category', '')} {s.get('sourcepage', '')} {s.get('sourcefile', '')}"
        for s in sources
    ).lower()

    for pattern in patterns:
        if pattern.lower() in all_source_text:
            return True, f"matched: {pattern}"
    return False, f"none of {patterns} found in sources"


def check_avoid_patterns(sources: list[dict], patterns: list[str]) -> tuple[bool, str]:
    """Check that sources do NOT predominantly match avoided patterns."""
    if not patterns:
        return True, "no patterns to avoid"

    total = len(sources)
    if total == 0:
        return True, "no sources"

    for pattern in patterns:
        matching = 0
        for s in sources:
            source_text = f"{s.get('category', '')} {s.get('sourcepage', '')} {s.get('sourcefile', '')}".lower()
            if pattern.lower() in source_text:
                matching += 1
        # Fail if >50% of sources match avoided pattern
        if matching > total * 0.5:
            return False, f"WRONG: {matching}/{total} sources match avoided pattern '{pattern}'"

    return True, "avoided patterns clear"


def run_test(test: dict) -> dict:
    """Run a single test case and return detailed results."""
    test_id = test["id"]
    question = test["question"]

    start = time.time()
    try:
        result = send_chat(question)
    except Exception as e:
        return {
            "id": test_id,
            "question": question,
            "error": str(e),
            "query_score": 0,
            "source_score": 0,
            "avoid_score": 0,
            "elapsed": 0,
        }
    elapsed = time.time() - start

    search_query = extract_search_query(result)
    sources = extract_sources(result)
    answer = result.get("message", {}).get("content", "")

    # Score query quality
    q_matched, q_total, q_missing = check_query_terms(search_query, test["expected_query_terms"])
    query_score = q_matched / q_total if q_total > 0 else 1.0

    # Score source accuracy
    src_ok, src_detail = check_source_patterns(sources, test["expected_source_patterns"])
    source_score = 1.0 if src_ok else 0.0

    # Score avoidance
    avoid_ok, avoid_detail = check_avoid_patterns(sources, test["avoid_source_patterns"])
    avoid_score = 1.0 if avoid_ok else 0.0

    return {
        "id": test_id,
        "question": question,
        "difficulty": test["difficulty"],
        "category": test["category"],
        "search_query": search_query,
        "query_score": query_score,
        "query_missing_terms": q_missing,
        "source_score": source_score,
        "source_detail": src_detail,
        "avoid_score": avoid_score,
        "avoid_detail": avoid_detail,
        "num_sources": len(sources),
        "sources": [f"{s['sourcefile']}" for s in sources[:3]],
        "answer_length": len(answer),
        "elapsed": round(elapsed, 1),
    }


def main():
    print("=" * 90)
    print("QUERY REWRITE A/B TEST — Knowledge-Grounded Reasoning Evaluation")
    print(f"Dataset: {len(TEST_DATASET)} queries | Server: {BASE_URL}")
    print("=" * 90)
    print()

    # Verify server is reachable
    try:
        httpx.get(f"{BASE_URL}/config", timeout=5)
    except Exception:
        print("ERROR: Cannot reach server at", BASE_URL)
        print("Start the app first: ./app/start.sh")
        return 1

    results = []
    for i, test in enumerate(TEST_DATASET, 1):
        tid = test["id"]
        print(f"[{i:2d}/{len(TEST_DATASET)}] {tid}: {test['question'][:70]}", end=" ", flush=True)

        r = run_test(test)
        results.append(r)

        # Compact status
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            q_icon = "✓" if r["query_score"] >= 0.5 else "✗"
            s_icon = "✓" if r["source_score"] >= 0.5 else "✗"
            a_icon = "✓" if r["avoid_score"] >= 0.5 else "✗"
            print(f" Q:{q_icon} S:{s_icon} A:{a_icon}  ({r['elapsed']}s)  query=\"{r.get('search_query', '')[:60]}\"")
            if r["query_missing_terms"]:
                print(f"       missing query terms: {r['query_missing_terms']}")
            if r["avoid_score"] < 1.0:
                print(f"       ⚠ {r['avoid_detail']}")

    # ---- AGGREGATE SCORES ----
    print()
    print("=" * 90)
    print("AGGREGATE RESULTS")
    print("=" * 90)

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No successful tests!")
        return 1

    avg_query = sum(r["query_score"] for r in valid) / len(valid)
    avg_source = sum(r["source_score"] for r in valid) / len(valid)
    avg_avoid = sum(r["avoid_score"] for r in valid) / len(valid)
    avg_elapsed = sum(r["elapsed"] for r in valid) / len(valid)

    print(f"\nOverall ({len(valid)} tests):")
    print(f"  Query term accuracy:     {avg_query:.1%}")
    print(f"  Source retrieval accuracy:{avg_source:.1%}")
    print(f"  Avoidance accuracy:      {avg_avoid:.1%}")
    print(f"  Avg latency:             {avg_elapsed:.1f}s")

    # By difficulty
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in valid if r["difficulty"] == diff]
        if subset:
            q = sum(r["query_score"] for r in subset) / len(subset)
            s = sum(r["source_score"] for r in subset) / len(subset)
            a = sum(r["avoid_score"] for r in subset) / len(subset)
            print(f"\n  {diff.upper()} ({len(subset)} tests):")
            print(f"    Query: {q:.1%}  Source: {s:.1%}  Avoid: {a:.1%}")

    # By category
    categories = sorted(set(r["category"] for r in valid))
    for cat in categories:
        subset = [r for r in valid if r["category"] == cat]
        if subset:
            q = sum(r["query_score"] for r in subset) / len(subset)
            s = sum(r["source_score"] for r in subset) / len(subset)
            a = sum(r["avoid_score"] for r in subset) / len(subset)
            print(f"\n  {cat} ({len(subset)}):")
            print(f"    Query: {q:.1%}  Source: {s:.1%}  Avoid: {a:.1%}")

    # Failed tests
    failures = [r for r in valid if r["query_score"] < 0.5 or r["source_score"] < 0.5 or r["avoid_score"] < 0.5]
    if failures:
        print(f"\n--- FAILURES ({len(failures)}) ---")
        for r in failures:
            issues = []
            if r["query_score"] < 0.5:
                issues.append(f"query({r['query_score']:.0%})")
            if r["source_score"] < 0.5:
                issues.append(f"source({r['source_score']:.0%})")
            if r["avoid_score"] < 0.5:
                issues.append(f"avoid({r['avoid_score']:.0%})")
            print(f"  {r['id']}: {r['question'][:50]}  [{', '.join(issues)}]")
            print(f"    query: \"{r.get('search_query', '')[:80]}\"")
            if r["query_missing_terms"]:
                print(f"    missing: {r['query_missing_terms']}")
            if r["avoid_score"] < 0.5:
                print(f"    avoid: {r['avoid_detail']}")

    # Save detailed results
    output_path = Path("scripts/query_rewrite_ab_results.json")
    output_path.write_text(json.dumps({
        "summary": {
            "total_tests": len(TEST_DATASET),
            "successful_tests": len(valid),
            "avg_query_accuracy": round(avg_query, 3),
            "avg_source_accuracy": round(avg_source, 3),
            "avg_avoid_accuracy": round(avg_avoid, 3),
            "avg_latency_s": round(avg_elapsed, 1),
            "failures": len(failures),
        },
        "results": results,
    }, indent=2))
    print(f"\nDetailed results saved to {output_path}")

    # Overall pass/fail
    overall_pass = avg_query >= 0.7 and avg_source >= 0.7 and avg_avoid >= 0.9
    print(f"\n{'PASS' if overall_pass else 'NEEDS REVIEW'}: query={avg_query:.1%} source={avg_source:.1%} avoid={avg_avoid:.1%}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
