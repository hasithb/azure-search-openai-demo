#!/usr/bin/env python3
"""Regression test for source hierarchy behaviour in answer generation.

Sends test questions to the live app with different source filters and checks
that the answer follows the intended hierarchy rules:
- All Sources + general question -> CPR first, court guides supplementary
- Court-specific question -> court guide first, CPR supplementary
- Specific court guide filter -> stay inside that guide
- Cross-guide contamination -> should not happen
- Weak retrieval / wrong filter -> answer should flag the mismatch or suggest a better source
"""

import asyncio
import json
import os
import httpx

BASE = os.environ.get("APP_URL", "http://localhost:50505")


async def ask(question: str, category: str = "", top: int = 7) -> dict:
    """Send a chat question and return the full response."""
    payload = {
        "messages": [{"role": "user", "content": question}],
        "context": {
            "overrides": {
                "retrieval_mode": "hybrid",
                "semantic_ranker": True,
                "top": top,
                "include_category": category,
            }
        },
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{BASE}/chat", json=payload)
        resp.raise_for_status()
        return resp.json()


def extract_sources(result: dict) -> list[dict]:
    """Extract source metadata from data_points."""
    texts = result.get("context", {}).get("data_points", {}).get("text", [])
    sources = []
    for t in texts:
        if isinstance(t, dict):
            sources.append(
                {
                    "category": t.get("category", ""),
                    "sourcepage": t.get("sourcepage", ""),
                    "content_preview": t.get("content", "")[:100],
                }
            )
    return sources


def check_answer_hierarchy(answer: str, sources: list[dict], scenario: dict) -> list[str]:
    """Apply explicit scenario expectations plus generic hierarchy heuristics."""
    issues = []
    answer_lower = answer.lower()
    source_categories = {source["category"] for source in sources if source.get("category")}

    cpr_sources = [
        source
        for source in sources
        if "civil procedure" in source["category"].lower() or "practice direction" in source["category"].lower()
    ]
    guide_sources = [
        source
        for source in sources
        if "court" in source["category"].lower()
        or "guide" in source["category"].lower()
        or "chancery" in source["category"].lower()
        or "king" in source["category"].lower()
        or "patent" in source["category"].lower()
    ]

    allowed_categories = set(scenario.get("allowed_categories", []))
    forbidden_categories = set(scenario.get("forbidden_categories", []))
    required_terms = [term.lower() for term in scenario.get("required_answer_terms", [])]
    required_any_terms = [term.lower() for term in scenario.get("required_any_answer_terms", [])]
    forbidden_terms = [term.lower() for term in scenario.get("forbidden_answer_terms", [])]

    if allowed_categories:
        unexpected_categories = sorted(category for category in source_categories if category not in allowed_categories)
        if unexpected_categories:
            issues.append(f"Unexpected source categories: {', '.join(unexpected_categories)}")

    if forbidden_categories:
        present_forbidden_categories = sorted(category for category in source_categories if category in forbidden_categories)
        if present_forbidden_categories:
            issues.append(f"Forbidden source categories present: {', '.join(present_forbidden_categories)}")

    if required_terms and not any(term in answer_lower for term in required_terms):
        issues.append(
            "Answer missing expected guidance terms: " + ", ".join(scenario.get("required_answer_terms", []))
        )

    if required_any_terms and not any(term in answer_lower for term in required_any_terms):
        issues.append(
            "Answer missing one-of guidance terms: " + ", ".join(scenario.get("required_any_answer_terms", []))
        )

    present_forbidden_terms = [term for term in forbidden_terms if term in answer_lower]
    if present_forbidden_terms:
        issues.append("Answer contains forbidden terms: " + ", ".join(present_forbidden_terms))

    if cpr_sources and guide_sources:
        if "guide" in answer_lower and "cpr" not in answer_lower and "part" not in answer_lower and "rule" not in answer_lower:
            issues.append("Answer mentions guides but doesn't reference CPR rules")

    return issues


SCENARIOS = [
    {
        "name": "All Sources - general CMC question",
        "question": "What are the rules on case management conferences?",
        "category": "",
        "check_description": "Should lead with CPR Part 29, supplement with court guides",
        "required_any_answer_terms": ["CPR Part 29", "Part 29", "court-specific"],
    },
    {
        "name": "All Sources - Commercial Court CMC",
        "question": "How does the Commercial Court handle case management conferences?",
        "category": "",
        "check_description": "Should lead with Commercial Court Guide, reference underlying CPR",
        "required_answer_terms": ["Commercial Court"],
    },
    {
        "name": "Patents Court filter - CMC",
        "question": "What are the CMC deadlines and procedures?",
        "category": "Patents Court",
        "check_description": "Should use only Patents Court Guide sources",
        "allowed_categories": ["Patents Court"],
        "forbidden_categories": ["Commercial Court", "Chancery Division", "King's Bench Division"],
        "required_answer_terms": ["Patents Court"],
    },
    {
        "name": "All Sources - standard disclosure",
        "question": "What is the standard disclosure process?",
        "category": "",
        "check_description": "Should lead with CPR Part 31, supplement with court guides",
        "required_answer_terms": ["CPR Part 31", "Part 31", "standard disclosure"],
    },
    {
        "name": "CPR filter - disclosure",
        "question": "What is the standard disclosure process?",
        "category": "Civil Procedure Rules and Practice Directions",
        "check_description": "Should use only CPR/PD sources, suggest checking court guides",
        "allowed_categories": ["Civil Procedure Rules and Practice Directions"],
        "required_answer_terms": ["CPR Part 31", "Part 31", "standard disclosure"],
    },
    {
        "name": "Chancery filter - should not use Commercial Court content",
        "question": "How are case management conferences handled?",
        "category": "Chancery Division",
        "check_description": "Should use only Chancery Guide, NOT Commercial Court Guide",
        "allowed_categories": ["Chancery Division"],
        "forbidden_categories": ["Commercial Court"],
        "required_answer_terms": ["Chancery Division", "Chancery"],
    },
    {
        "name": "All Sources - pre-action disclosure",
        "question": "Tell me about pre-action disclosure under CPR 31.16",
        "category": "",
        "check_description": "Should lead with CPR 31.16, supplement with court guide references if any",
        "required_answer_terms": ["CPR 31.16", "31.16", "pre-action disclosure"],
    },
    {
        "name": "CPR filter - Commercial Court procedure mismatch",
        "question": "How does the Commercial Court handle case management conferences?",
        "category": "Civil Procedure Rules and Practice Directions",
        "check_description": "Should stay in CPR sources and recommend the Commercial Court Guide",
        "allowed_categories": ["Civil Procedure Rules and Practice Directions"],
        "required_answer_terms": ["Commercial Court", "Guide"],
    },
    {
        "name": "Patents filter - Commercial Court mismatch",
        "question": "How does the Commercial Court handle case management conferences?",
        "category": "Patents Court",
        "check_description": "Should not import Commercial Court guidance while filtered to Patents Court",
        "allowed_categories": ["Patents Court"],
        "forbidden_categories": ["Commercial Court"],
        "required_answer_terms": ["Commercial Court"],
    },
]


async def main():
    print("=" * 80)
    print("SOURCE HIERARCHY TEST")
    print(f"Target: {BASE}")
    print("=" * 80)

    results = []
    failure_count = 0

    for i, scenario in enumerate(SCENARIOS):
        print(f"\n[{i+1}/{len(SCENARIOS)}] {scenario['name']}")
        print(f"  Q: {scenario['question']}")
        print(f"  Filter: {scenario['category'] or 'All Sources'}")
        print(f"  Check: {scenario['check_description']}")

        try:
            result = await ask(scenario["question"], scenario["category"])
            answer = result.get("message", {}).get("content", "")
            sources = extract_sources(result)

            # Categorize sources
            source_cats = {}
            for s in sources:
                cat = s["category"]
                source_cats[cat] = source_cats.get(cat, 0) + 1

            print(f"  Sources: {json.dumps(source_cats, indent=None)}")
            print(f"  Answer preview: {answer[:200]}...")

            issues = check_answer_hierarchy(answer, sources, scenario)
            if issues:
                failure_count += 1
                print(f"  ⚠ Issues: {'; '.join(issues)}")
            else:
                print(f"  ✓ No obvious hierarchy issues")

            results.append({
                "scenario": scenario["name"],
                "question": scenario["question"],
                "filter": scenario["category"],
                "source_categories": source_cats,
                "answer": answer,
                "answer_length": len(answer),
                "issues": issues,
            })

        except Exception as e:
            failure_count += 1
            print(f"  ERROR: {e}")
            results.append(
                {
                    "scenario": scenario["name"],
                    "error": str(e),
                }
            )

    outfile = os.path.join(os.path.dirname(__file__), "source_hierarchy_results.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")

    print(f"\nSummary: {len(SCENARIOS) - failure_count}/{len(SCENARIOS)} scenarios passed")
    if failure_count:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
