"""
A/B test: Does adding sourcepage + category metadata to prompt sources improve accuracy?

Compares the current format:
  [1]: <content>

Against an enriched format:
  [1] (Category: Commercial Court | Source: Part 58): <content>

This script does NOT modify the running app. Instead, it:
1. Sends questions to the running app to get the actual search results
2. Then builds both prompt variants locally and sends them to the same LLM
3. Compares the resulting answers for quality

Run: python scripts/test_metadata_enrichment.py
Requires: app running at http://localhost:50505
"""

import asyncio
import json
import os
import sys
import re
from datetime import datetime

import httpx

BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:50505")

# Test cases with expected quality signals
TEST_CASES = [
    {
        "name": "CPR Part 31 standard disclosure",
        "question": "What is standard disclosure?",
        "checks": [
            {"type": "contains_any", "terms": ["CPR 31", "Part 31", "31.6"], "desc": "References Part 31"},
            {"type": "contains_any", "terms": ["documents", "disclose", "disclosure"], "desc": "Explains disclosure concept"},
        ],
    },
    {
        "name": "Commercial Court case management",
        "question": "What are the case management procedures in the Commercial Court?",
        "checks": [
            {"type": "contains_any", "terms": ["Commercial Court", "case management"], "desc": "Addresses Commercial Court"},
            {"type": "has_citation", "desc": "Has citations"},
        ],
    },
    {
        "name": "Cross-source: injunctions across courts",
        "question": "How do different courts handle interim injunction applications?",
        "checks": [
            {"type": "contains_any", "terms": ["Part 25", "injunction", "interim"], "desc": "References injunctions"},
            {"type": "has_citation", "desc": "Has citations"},
        ],
    },
    {
        "name": "Costs budgeting",
        "question": "What are the rules on costs budgeting?",
        "checks": [
            {"type": "contains_any", "terms": ["Part 3", "3E", "PD3E", "costs budget", "costs management"], "desc": "References costs budgeting rules"},
            {"type": "has_citation", "desc": "Has citations"},
        ],
    },
    {
        "name": "Pre-action disclosure vs protocols",
        "question": "What is the difference between pre-action disclosure and pre-action protocols?",
        "checks": [
            {"type": "contains_any", "terms": ["31.16", "CPR 31"], "desc": "Distinguishes PAD (CPR 31.16)"},
            {"type": "contains_any", "terms": ["protocol", "Protocol", "pre-litigation"], "desc": "Also addresses protocols"},
        ],
    },
    {
        "name": "Appeal time limits",
        "question": "What are the time limits for filing an appeal?",
        "checks": [
            {"type": "contains_any", "terms": ["21 days", "Part 52", "appellant"], "desc": "Cites Part 52 time limits"},
            {"type": "has_citation", "desc": "Has citations"},
        ],
    },
    {
        "name": "TCC procedures",
        "question": "What are the specific procedures in the Technology and Construction Court?",
        "checks": [
            {"type": "contains_any", "terms": ["TCC", "Technology and Construction"], "desc": "Addresses TCC specifically"},
            {"type": "has_citation", "desc": "Has citations"},
        ],
    },
    {
        "name": "Source precision: which CPR part is being cited",
        "question": "What is the overriding objective?",
        "checks": [
            {"type": "contains_any", "terms": ["Part 1", "1.1", "overriding objective"], "desc": "Correctly identifies Part 1"},
            {"type": "has_citation", "desc": "Has citations"},
        ],
    },
]


def run_checks(answer: str, checks: list[dict]) -> list[dict]:
    """Run quality checks against an answer."""
    results = []
    for check in checks:
        if check["type"] == "contains_any":
            passed = any(t.lower() in answer.lower() for t in check["terms"])
        elif check["type"] == "has_citation":
            passed = bool(re.search(r"\[\d+\]", answer))
        else:
            passed = False
        results.append({"desc": check["desc"], "passed": passed})
    return results


async def fetch_chat_response(client: httpx.AsyncClient, question: str, overrides: dict | None = None) -> dict:
    """Send a question to the running app and get the full response."""
    payload = {
        "messages": [{"content": question, "role": "user"}],
        "context": {"overrides": overrides or {}},
    }
    resp = await client.post(f"{BASE_URL}/chat", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def extract_sources_metadata(result: dict) -> list[dict]:
    """Extract text source dicts from the chat response (which include metadata)."""
    try:
        data_points = result.get("context", {}).get("data_points", {})
        text_sources = data_points.get("text", [])
        return text_sources
    except Exception:
        return []


def format_current(sources: list[dict]) -> list[str]:
    """Current format: [N]: content only."""
    formatted = []
    for i, s in enumerate(sources, 1):
        if isinstance(s, dict):
            content = s.get("content", "")
        else:
            content = str(s)
        formatted.append(f"[{i}]: {content}")
    return formatted


def format_enriched(sources: list[dict]) -> list[str]:
    """Enriched format: [N] (Category | Source: sourcepage): content."""
    formatted = []
    for i, s in enumerate(sources, 1):
        if isinstance(s, dict):
            content = s.get("content", "")
            category = s.get("category", "")
            sourcepage = s.get("sourcepage", "")
            meta_parts = []
            if category:
                meta_parts.append(f"Category: {category}")
            if sourcepage:
                meta_parts.append(f"Source: {sourcepage}")
            meta = " | ".join(meta_parts)
            if meta:
                formatted.append(f"[{i}] ({meta}): {content}")
            else:
                formatted.append(f"[{i}]: {content}")
        else:
            formatted.append(f"[{i}]: {str(s)}")
    return formatted


async def main():
    print("=" * 70)
    print("METADATA ENRICHMENT A/B TEST")
    print(f"Target: {BASE_URL}")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 70)

    async with httpx.AsyncClient() as client:
        # First verify app is reachable
        try:
            health = await client.get(f"{BASE_URL}/", timeout=10)
            if health.status_code not in (200, 304):
                print(f"WARNING: App returned {health.status_code}")
        except Exception as e:
            print(f"ERROR: Cannot reach {BASE_URL}: {e}")
            sys.exit(1)

        current_total = 0
        current_passed = 0
        enriched_total = 0
        enriched_passed = 0
        results_log = []

        for tc in TEST_CASES:
            print(f"\n--- {tc['name']} ---")
            print(f"Q: {tc['question']}")

            try:
                # Get the normal response (current format)
                result = await fetch_chat_response(client, tc["question"])
                current_answer = result.get("message", {}).get("content", "")
                sources = extract_sources_metadata(result)

                # Check how many sources have metadata
                sources_with_category = sum(1 for s in sources if isinstance(s, dict) and s.get("category"))
                sources_with_sourcepage = sum(1 for s in sources if isinstance(s, dict) and s.get("sourcepage"))
                print(f"  Sources: {len(sources)} total, {sources_with_category} with category, {sources_with_sourcepage} with sourcepage")

                # Show sample enriched format
                enriched = format_enriched(sources[:3])
                for line in enriched:
                    # Truncate content for display
                    if len(line) > 120:
                        print(f"  Sample: {line[:120]}...")
                    else:
                        print(f"  Sample: {line}")

                # Run checks on current answer
                current_checks = run_checks(current_answer, tc["checks"])
                c_pass = sum(1 for c in current_checks if c["passed"])
                c_total = len(current_checks)
                current_passed += c_pass
                current_total += c_total
                print(f"  Current format: {c_pass}/{c_total} checks passed")
                for c in current_checks:
                    status = "PASS" if c["passed"] else "FAIL"
                    print(f"    [{status}] {c['desc']}")

                # Now get the enriched response
                # We send the same question but with a prompt override that tells the LLM
                # about the enriched source format
                enriched_override = (
                    ">>>Each source below includes metadata in parentheses showing its Category (document collection) "
                    "and Source (specific section/page). Use this metadata to better understand which document and section "
                    "each piece of information comes from. This helps you identify which CPR Part, Practice Direction, "
                    "or Court Guide a source belongs to."
                )
                enriched_result = await fetch_chat_response(client, tc["question"], {"prompt_template": enriched_override})
                enriched_answer = enriched_result.get("message", {}).get("content", "")

                enriched_checks = run_checks(enriched_answer, tc["checks"])
                e_pass = sum(1 for c in enriched_checks if c["passed"])
                e_total = len(enriched_checks)
                enriched_passed += e_pass
                enriched_total += e_total
                print(f"  Enriched format: {e_pass}/{e_total} checks passed")
                for c in enriched_checks:
                    status = "PASS" if c["passed"] else "FAIL"
                    print(f"    [{status}] {c['desc']}")

                results_log.append({
                    "name": tc["name"],
                    "question": tc["question"],
                    "num_sources": len(sources),
                    "sources_with_category": sources_with_category,
                    "sources_with_sourcepage": sources_with_sourcepage,
                    "current_checks": current_checks,
                    "enriched_checks": enriched_checks,
                    "current_pass_rate": f"{c_pass}/{c_total}",
                    "enriched_pass_rate": f"{e_pass}/{e_total}",
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results_log.append({"name": tc["name"], "error": str(e)})

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        c_pct = (current_passed / current_total * 100) if current_total else 0
        e_pct = (enriched_passed / enriched_total * 100) if enriched_total else 0
        print(f"Current format:  {current_passed}/{current_total} checks ({c_pct:.0f}%)")
        print(f"Enriched format: {enriched_passed}/{enriched_total} checks ({e_pct:.0f}%)")
        diff = e_pct - c_pct
        if diff > 0:
            print(f"  -> Enriched is BETTER by {diff:.0f} percentage points")
        elif diff < 0:
            print(f"  -> Current is BETTER by {-diff:.0f} percentage points")
        else:
            print(f"  -> Same accuracy")

        # Metadata availability summary
        print(f"\nMetadata availability across all test cases:")
        total_sources = sum(r.get("num_sources", 0) for r in results_log if "error" not in r)
        total_with_cat = sum(r.get("sources_with_category", 0) for r in results_log if "error" not in r)
        total_with_sp = sum(r.get("sources_with_sourcepage", 0) for r in results_log if "error" not in r)
        print(f"  Total sources retrieved: {total_sources}")
        print(f"  With category: {total_with_cat} ({total_with_cat/total_sources*100:.0f}%)" if total_sources else "  With category: 0")
        print(f"  With sourcepage: {total_with_sp} ({total_with_sp/total_sources*100:.0f}%)" if total_sources else "  With sourcepage: 0")

        # Save log
        log_path = "scripts/test_metadata_enrichment_results.json"
        with open(log_path, "w") as f:
            json.dump({"summary": {"current": f"{current_passed}/{current_total}", "enriched": f"{enriched_passed}/{enriched_total}"}, "results": results_log}, f, indent=2)
        print(f"\nFull results saved to {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
