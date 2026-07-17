"""Test canonical concept retrieval accuracy across multiple queries.

Run with: python scripts/test_canonical_retrieval.py
Requires the app to be running at localhost:50505.
"""

import json
import sys
import urllib.request

APP_URL = "http://localhost:50505/chat"

# Each test case: (query, expected_sourcefile_substring, description)
TEST_CASES = [
    (
        "what are the requirements to successfully apply for pre action disclosure",
        "Part 31",
        "Pre-action disclosure should retrieve CPR Part 31 (rule 31.16)",
    ),
    (
        "Tell me about pre-action disclosure",
        "Part 31",
        "Direct PAD query should retrieve CPR Part 31",
    ),
    (
        "How do I get summary judgment",
        "Part 24",
        "Summary judgment should retrieve CPR Part 24",
    ),
    (
        "what is the test for summary judgment under the CPR",
        "Part 24",
        "Explicit SJ query should retrieve Part 24",
    ),
    (
        "What are the rules on standard disclosure",
        "Part 31",
        "Standard disclosure should retrieve Part 31",
    ),
    (
        "How do I apply for relief from sanctions",
        "Part 3",
        "Relief from sanctions should retrieve Part 3 (rule 3.9)",
    ),
    (
        "What is the test for striking out a statement of case",
        "Part 3",
        "Strike out should retrieve Part 3 (rule 3.4)",
    ),
    (
        "What are the time limits for filing an appeal",
        "Part 52",
        "Appeals should retrieve Part 52",
    ),
]


def run_query(query: str) -> dict:
    payload = json.dumps({
        "messages": [{"role": "user", "content": query}],
        "context": {
            "overrides": {
                "retrieval_mode": "hybrid",
                "semantic_ranker": True,
                "top": 7,
            }
        },
        "stream": False,
    }).encode()
    req = urllib.request.Request(APP_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def analyze_result(data: dict, expected_source: str) -> dict:
    ctx = data.get("context", {})
    thoughts = ctx.get("thoughts", [])
    dp = ctx.get("data_points", {})
    texts = dp.get("text", [])

    # Get final source list
    sources = []
    for t in texts:
        if isinstance(t, dict):
            sources.append({
                "sourcefile": t.get("sourcefile", "?"),
                "sourcepage": t.get("sourcepage", "?"),
            })

    # Get search results from thoughts
    search_results = []
    rewritten_query = ""
    adaptive_info = {}
    for th in thoughts:
        title = th.get("title", "")
        if "Search using" in title:
            rewritten_query = str(th.get("description", ""))
        if title == "Search results":
            search_results = th.get("description", [])
        if "Adaptive" in title:
            adaptive_info = th.get("description", {})

    # Check if expected source is in final results
    found_in_final = any(
        expected_source.lower() in s.get("sourcefile", "").lower()
        for s in sources
    )
    # Check position if found
    position = None
    for i, s in enumerate(sources):
        if expected_source.lower() in s.get("sourcefile", "").lower():
            position = i + 1
            break

    # Check if expected source is in raw search results
    found_in_search = any(
        expected_source.lower() in (r.get("sourcefile", "") if isinstance(r, dict) else "").lower()
        for r in search_results
    )

    return {
        "found_in_final": found_in_final,
        "position": position,
        "total_sources": len(sources),
        "found_in_search": found_in_search,
        "rewritten_query": rewritten_query[:150],
        "adaptive_used": adaptive_info.get("used", False),
        "targeted_queries": adaptive_info.get("targeted_reference_queries", []),
        "source_files": [s["sourcefile"] for s in sources[:7]],
    }


def main():
    print("=" * 80)
    print("CANONICAL CONCEPT RETRIEVAL TEST")
    print("=" * 80)

    results = []
    for i, (query, expected, description) in enumerate(TEST_CASES, 1):
        print(f"\n--- Test {i}/{len(TEST_CASES)}: {description} ---")
        print(f"  Query: {query}")
        print(f"  Expected source: {expected}")

        try:
            data = run_query(query)
            analysis = analyze_result(data, expected)
            analysis["query"] = query
            analysis["expected"] = expected
            analysis["description"] = description
            results.append(analysis)

            status = "PASS" if analysis["found_in_final"] else "FAIL"
            pos_str = f" (position {analysis['position']})" if analysis["position"] else ""
            print(f"  Result: {status}{pos_str}")
            print(f"  Rewritten query: {analysis['rewritten_query']}")
            print(f"  Adaptive retry: {analysis['adaptive_used']}, targeted: {analysis['targeted_queries']}")
            print(f"  Sources: {analysis['source_files'][:5]}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "query": query, "expected": expected, "description": description,
                "found_in_final": False, "error": str(e),
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(1 for r in results if r.get("found_in_final"))
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    for r in results:
        status = "PASS" if r.get("found_in_final") else "FAIL"
        pos = f" @{r.get('position')}" if r.get("position") else ""
        print(f"  [{status}{pos}] {r['description']}")

    return passed, total


if __name__ == "__main__":
    passed, total = main()
    sys.exit(0 if passed == total else 1)
