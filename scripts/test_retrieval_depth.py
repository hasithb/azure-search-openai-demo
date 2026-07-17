"""
Test script to evaluate retrieval quality with realistic practitioner questions.

Tests combinations of:
- Search depth: minimal (Quick), low (Standard), medium (Thorough)
- Top-K: 3, 5, 7, 10
- Practical question types: pre-action correspondence, pre-action disclosure,
  case management documents, key deadlines, disclosure during proceedings,
  ADR/settlement, scenario-based, court-specific, and complex timeline questions.

Evaluates:
- Whether an answer was provided (vs. "I don't know")
- Number of citations in the answer (numeric [1][2][3] format)
- Number of unique source documents referenced via citation_map
- Answer length (proxy for detail)
- Number of data_points returned (retrieval depth)
- Response time
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:50505"

# Citation regex: this app uses numeric citations like [1], [2], [3]
NUMERIC_CITATION_REGEX = re.compile(r"\[(\d+)\]")

# Test questions: realistic practitioner questions that a lawyer would ask
# Organised by practical topic areas
TEST_QUESTIONS = [
    # ── Pre-Action Correspondence ────────────────────────────────────────────
    {
        "id": "preaction_letter_1",
        "question": "What must I include in a letter before claim under the Practice Direction on Pre-Action Conduct and Protocols?",
        "type": "pre_action_correspondence",
        "expected_min_citations": 1,
    },
    {
        "id": "preaction_response_2",
        "question": "How long does the defendant have to respond to a pre-action letter of claim, and what should the response contain?",
        "type": "pre_action_correspondence",
        "expected_min_citations": 1,
    },
    {
        "id": "preaction_sanctions_3",
        "question": "What costs sanctions can the court impose if a party fails to comply with a pre-action protocol or the Practice Direction on Pre-Action Conduct?",
        "type": "pre_action_correspondence",
        "expected_min_citations": 1,
    },
    # ── Pre-Action Disclosure ────────────────────────────────────────────────
    {
        "id": "preaction_disclosure_4",
        "question": "Can I obtain disclosure of documents before issuing proceedings, and what conditions must be met under CPR 31.16?",
        "type": "pre_action_disclosure",
        "expected_min_citations": 1,
    },
    {
        "id": "preaction_keydocs_5",
        "question": "What key documents should the parties disclose during the pre-action stage before issuing a claim at court?",
        "type": "pre_action_disclosure",
        "expected_min_citations": 1,
    },
    # ── Case Management Documents ────────────────────────────────────────────
    {
        "id": "casemanage_dq_6",
        "question": "What is a directions questionnaire, when must it be filed after a defence is served, and what happens if a party fails to file it?",
        "type": "case_management",
        "expected_min_citations": 1,
    },
    {
        "id": "casemanage_disclosure_report_7",
        "question": "What must a disclosure report under CPR 31.5 contain and when should it be filed before a case management conference?",
        "type": "case_management",
        "expected_min_citations": 1,
    },
    {
        "id": "casemanage_cmc_8",
        "question": "What documents and information do I need to prepare for a case management conference in the Commercial Court?",
        "type": "case_management",
        "expected_min_citations": 1,
    },
    # ── Key Deadlines ────────────────────────────────────────────────────────
    {
        "id": "deadline_service_9",
        "question": "What are the time limits for serving a claim form after it has been issued, and what methods of service are permitted under CPR Part 6?",
        "type": "key_deadlines",
        "expected_min_citations": 1,
    },
    {
        "id": "deadline_ack_defence_10",
        "question": "How many days does a defendant have to file an acknowledgment of service and then a defence after being served with a claim form?",
        "type": "key_deadlines",
        "expected_min_citations": 2,
    },
    {
        "id": "deadline_allocation_11",
        "question": "After defences are filed by all parties, what is the timeline for allocation to a track and what deadlines apply for the directions questionnaire?",
        "type": "key_deadlines",
        "expected_min_citations": 1,
    },
    # ── Disclosure During Proceedings ─────────────────────────────────────────
    {
        "id": "disclosure_standard_12",
        "question": "What documents must a party disclose under standard disclosure, and what is the duty of search under CPR Part 31?",
        "type": "disclosure",
        "expected_min_citations": 1,
    },
    {
        "id": "disclosure_specific_13",
        "question": "How do I apply for specific disclosure of documents from the other party under CPR 31.12, and what must the court order specify?",
        "type": "disclosure",
        "expected_min_citations": 1,
    },
    {
        "id": "disclosure_pd57ad_14",
        "question": "What disclosure models are available under Practice Direction 57AD in the Business and Property Courts, and how does Extended Disclosure differ from standard disclosure?",
        "type": "disclosure",
        "expected_min_citations": 1,
    },
    # ── ADR and Settlement ──────────────────────────────────────────────────
    {
        "id": "adr_mediation_15",
        "question": "What ADR options should I consider before issuing proceedings, and what could happen if my client unreasonably refuses to mediate?",
        "type": "adr_settlement",
        "expected_min_citations": 1,
    },
    # ── Practical Scenario: Track Allocation ─────────────────────────────────
    {
        "id": "scenario_allocation_16",
        "question": "My client has a commercial contract dispute worth £50,000. Which track is it likely to be allocated to, and what are the key procedural steps from allocation to trial?",
        "type": "scenario",
        "expected_min_citations": 2,
    },
    # ── Urgent Applications ──────────────────────────────────────────────────
    {
        "id": "urgent_injunction_17",
        "question": "I need to apply for an interim injunction urgently. Can I skip the pre-action protocol requirements, and what does CPR Part 25 say about interim remedies?",
        "type": "scenario",
        "expected_min_citations": 2,
    },
    # ── Court-Specific Practical Questions ────────────────────────────────────
    {
        "id": "court_tcc_18",
        "question": "What pre-trial steps and case management procedures does the Technology and Construction Court Guide require for construction disputes?",
        "type": "court_guide",
        "expected_min_citations": 1,
    },
    {
        "id": "court_chancery_19",
        "question": "What are the costs budgeting and costs management requirements in the Chancery Division, and how does Practice Direction 3D apply?",
        "type": "court_guide",
        "expected_min_citations": 1,
    },
    # ── Complex Multi-Step Timeline ──────────────────────────────────────────
    {
        "id": "timeline_full_20",
        "question": "Walk me through the complete timeline and key deadlines from sending a letter before claim, through issuing and serving proceedings under Part 7, to the first case management conference, including what documents are needed at each stage.",
        "type": "complex_timeline",
        "expected_min_citations": 3,
    },
]

# Configurations to test
CONFIGS = []
for depth in ["minimal", "low", "medium"]:
    for top_k in [3, 5, 7, 10]:
        CONFIGS.append({
            "label": f"depth={depth}, top={top_k}",
            "depth": depth,
            "top": top_k,
        })


def build_request(question: str, top: int, depth: str, category: str = "All") -> dict:
    """Build a chat API request body."""
    overrides = {
        "retrieval_mode": "hybrid",
        "semantic_ranker": True,
        "semantic_captions": False,
        "top": top,
        "query_rewriting": True,
        "retrieval_reasoning_effort": depth,
        "send_text_sources": True,
    }
    if category != "All":
        overrides["category_filter"] = category

    return {
        "messages": [{"role": "user", "content": question}],
        "context": {"overrides": overrides},
        "session_state": None,
        "stream": False,
    }


def extract_citations(answer_text: str) -> list[int]:
    """Extract numeric citation references from an answer like [1], [2], [3]."""
    return [int(m) for m in NUMERIC_CITATION_REGEX.findall(answer_text)]


def extract_unique_sources_from_citation_map(citation_map: dict, cited_numbers: list[int]) -> set[str]:
    """Get unique source descriptions from citation_map for cited numbers."""
    sources = set()
    for num in cited_numbers:
        key = str(num)
        if key in citation_map:
            sources.add(citation_map[key])
    return sources


def extract_source_categories(data_points_text: list[dict]) -> set[str]:
    """Get unique categories from data points."""
    return {dp.get("category", "unknown") for dp in data_points_text if dp.get("category")}


def extract_source_documents(data_points_text: list[dict]) -> set[str]:
    """Get unique source document identifiers from data points."""
    docs = set()
    for dp in data_points_text:
        citation = dp.get("citation", "")
        # Extract the document-level identifier (last part of citation)
        parts = citation.rsplit(", ", 1)
        if len(parts) > 1:
            docs.add(parts[-1])
        elif citation:
            docs.add(citation)
    return docs


def is_refusal(answer: str) -> bool:
    """Check if the answer is essentially 'I don't know'."""
    refusal_phrases = [
        "i'm not sure",
        "i don't know",
        "i cannot find",
        "the retrieved documents do not",
        "no relevant information",
        "not mentioned in the sources",
        "information is not available",
        "cannot be determined from",
        "does not appear in",
    ]
    lower = answer.lower()
    return any(phrase in lower for phrase in refusal_phrases)


async def run_single_test(
    client: httpx.AsyncClient,
    question_data: dict,
    config: dict,
) -> dict:
    """Run a single question with a specific config and return metrics."""
    req = build_request(
        question=question_data["question"],
        top=config["top"],
        depth=config["depth"],
    )

    start = time.monotonic()
    try:
        resp = await client.post(f"{BASE_URL}/chat", json=req, timeout=120.0)
        elapsed = time.monotonic() - start

        if resp.status_code != 200:
            return {
                "question_id": question_data["id"],
                "config": config["label"],
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "elapsed_s": elapsed,
            }

        data = resp.json()
        answer = data.get("message", {}).get("content", "")
        cited_numbers = extract_citations(answer)
        unique_cited = set(cited_numbers)

        # Extract from context
        context = data.get("context", {})
        citation_map = context.get("citation_map", {})
        unique_sources = extract_unique_sources_from_citation_map(citation_map, cited_numbers)
        data_points_text = context.get("data_points", {}).get("text", [])
        source_docs = extract_source_documents(data_points_text)
        source_categories = extract_source_categories(data_points_text)

        # Count thoughts/search results
        thoughts = context.get("thoughts", [])
        search_results_count = 0
        for thought in thoughts:
            if isinstance(thought, dict) and thought.get("title") == "Search results":
                props = thought.get("props", [])
                if isinstance(props, list):
                    search_results_count = len(props)

        return {
            "question_id": question_data["id"],
            "question_type": question_data["type"],
            "config": config["label"],
            "depth": config["depth"],
            "top_k": config["top"],
            "answer_length": len(answer),
            "citation_count": len(unique_cited),
            "cited_numbers": sorted(unique_cited),
            "unique_source_count": len(unique_sources),
            "unique_sources": sorted(unique_sources),
            "source_docs_in_retrieval": sorted(source_docs),
            "source_categories": sorted(source_categories),
            "is_refusal": is_refusal(answer),
            "data_points_count": len(data_points_text),
            "total_citation_map_entries": len(citation_map),
            "search_results_count": search_results_count,
            "elapsed_s": round(elapsed, 2),
            "expected_min_citations": question_data["expected_min_citations"],
            "meets_citation_threshold": len(unique_cited) >= question_data["expected_min_citations"],
            "answer_preview": answer[:200],
        }
    except Exception as e:
        elapsed = time.monotonic() - start
        return {
            "question_id": question_data["id"],
            "config": config["label"],
            "error": str(e),
            "elapsed_s": round(elapsed, 2),
        }


async def main():
    """Run all test combinations and output results."""
    print(f"Testing {len(TEST_QUESTIONS)} questions × {len(CONFIGS)} configs = {len(TEST_QUESTIONS) * len(CONFIGS)} total requests")
    print("=" * 80)

    all_results = []

    async with httpx.AsyncClient() as client:
        # Verify app is running
        try:
            resp = await client.get(f"{BASE_URL}/config", timeout=5.0)
            if resp.status_code != 200:
                print(f"ERROR: App not responding correctly at {BASE_URL}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Cannot reach app at {BASE_URL}: {e}")
            sys.exit(1)

        total = len(TEST_QUESTIONS) * len(CONFIGS)
        done = 0

        for question_data in TEST_QUESTIONS:
            print(f"\n--- Question: {question_data['id']} ({question_data['type']}) ---")
            print(f"    \"{question_data['question'][:80]}...\"" if len(question_data['question']) > 80 else f"    \"{question_data['question']}\"")

            for config in CONFIGS:
                done += 1
                print(f"  [{done}/{total}] {config['label']}...", end=" ", flush=True)

                result = await run_single_test(client, question_data, config)
                all_results.append(result)

                if "error" in result:
                    print(f"ERROR: {result['error'][:60]}")
                else:
                    status = "REFUSAL" if result["is_refusal"] else "OK"
                    print(
                        f"{status} | "
                        f"cites={result['citation_count']}, "
                        f"sources={result['unique_source_count']}, "
                        f"dp={result['data_points_count']}, "
                        f"cm={result['total_citation_map_entries']}, "
                        f"len={result['answer_length']}, "
                        f"time={result['elapsed_s']}s"
                    )

    # Save raw results
    output_dir = Path(__file__).parent / "retrieval_depth_test_results"
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "raw_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {results_file}")

    # Generate summary analysis
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    # Group by config
    config_stats = {}
    for r in all_results:
        if "error" in r:
            continue
        key = r["config"]
        if key not in config_stats:
            config_stats[key] = {
                "total": 0,
                "refusals": 0,
                "meets_threshold": 0,
                "total_citations": 0,
                "total_sources": 0,
                "total_data_points": 0,
                "total_cm_entries": 0,
                "total_answer_length": 0,
                "total_time": 0.0,
                "depth": r["depth"],
                "top_k": r["top_k"],
            }
        s = config_stats[key]
        s["total"] += 1
        s["refusals"] += 1 if r["is_refusal"] else 0
        s["meets_threshold"] += 1 if r["meets_citation_threshold"] else 0
        s["total_citations"] += r["citation_count"]
        s["total_sources"] += r["unique_source_count"]
        s["total_data_points"] += r["data_points_count"]
        s["total_cm_entries"] += r["total_citation_map_entries"]
        s["total_answer_length"] += r["answer_length"]
        s["total_time"] += r["elapsed_s"]

    print(f"\n{'Config':<30} {'Answd':>6} {'MetCt':>6} {'AvCite':>7} {'AvSrc':>7} {'AvDP':>6} {'AvCM':>6} {'AvLen':>7} {'AvTime':>7}")
    print("-" * 100)

    for key in sorted(config_stats.keys()):
        s = config_stats[key]
        n = s["total"]
        answered = n - s["refusals"]
        print(
            f"{key:<30} "
            f"{answered}/{n:>3} "
            f"{s['meets_threshold']}/{n:>3} "
            f"{s['total_citations']/n:>7.1f} "
            f"{s['total_sources']/n:>7.1f} "
            f"{s['total_data_points']/n:>6.1f} "
            f"{s['total_cm_entries']/n:>6.1f} "
            f"{s['total_answer_length']/n:>7.0f} "
            f"{s['total_time']/n:>6.1f}s"
        )

    # Group by depth (averaging across top-K)
    print(f"\n\n--- By Search Depth (averaged across top-K) ---")
    depth_stats = {}
    for key, s in config_stats.items():
        d = s["depth"]
        if d not in depth_stats:
            depth_stats[d] = {"total": 0, "refusals": 0, "meets_threshold": 0, "total_citations": 0, "total_sources": 0, "total_data_points": 0, "total_cm_entries": 0, "total_answer_length": 0, "total_time": 0.0}
        for k in ["total", "refusals", "meets_threshold", "total_citations", "total_sources", "total_data_points", "total_cm_entries", "total_answer_length", "total_time"]:
            depth_stats[d][k] += s[k]

    print(f"{'Depth':<15} {'Answd':>6} {'MetCt':>6} {'AvCite':>7} {'AvSrc':>7} {'AvDP':>6} {'AvCM':>6} {'AvLen':>7} {'AvTime':>7}")
    print("-" * 80)
    for d in ["minimal", "low", "medium"]:
        if d in depth_stats:
            s = depth_stats[d]
            n = s["total"]
            answered = n - s["refusals"]
            print(
                f"{d:<15} "
                f"{answered}/{n:>3} "
                f"{s['meets_threshold']}/{n:>3} "
                f"{s['total_citations']/n:>7.1f} "
                f"{s['total_sources']/n:>7.1f} "
                f"{s['total_data_points']/n:>6.1f} "
                f"{s['total_cm_entries']/n:>6.1f} "
                f"{s['total_answer_length']/n:>7.0f} "
                f"{s['total_time']/n:>6.1f}s"
            )

    # Group by top-K (averaging across depth)
    print(f"\n\n--- By Top-K (averaged across search depth) ---")
    topk_stats = {}
    for r in all_results:
        if "error" in r:
            continue
        tk = r["top_k"]
        if tk not in topk_stats:
            topk_stats[tk] = {"total": 0, "refusals": 0, "meets_threshold": 0, "total_citations": 0, "total_sources": 0, "total_data_points": 0, "total_cm_entries": 0, "total_answer_length": 0, "total_time": 0.0}
        s = topk_stats[tk]
        s["total"] += 1
        s["refusals"] += 1 if r["is_refusal"] else 0
        s["meets_threshold"] += 1 if r["meets_citation_threshold"] else 0
        s["total_citations"] += r["citation_count"]
        s["total_sources"] += r["unique_source_count"]
        s["total_data_points"] += r["data_points_count"]
        s["total_cm_entries"] += r["total_citation_map_entries"]
        s["total_answer_length"] += r["answer_length"]
        s["total_time"] += r["elapsed_s"]

    print(f"{'Top-K':<10} {'Answd':>6} {'MetCt':>6} {'AvCite':>7} {'AvSrc':>7} {'AvDP':>6} {'AvCM':>6} {'AvLen':>7} {'AvTime':>7}")
    print("-" * 75)
    for tk in sorted(topk_stats.keys()):
        s = topk_stats[tk]
        n = s["total"]
        answered = n - s["refusals"]
        print(
            f"{tk:<10} "
            f"{answered}/{n:>3} "
            f"{s['meets_threshold']}/{n:>3} "
            f"{s['total_citations']/n:>7.1f} "
            f"{s['total_sources']/n:>7.1f} "
            f"{s['total_data_points']/n:>6.1f} "
            f"{s['total_cm_entries']/n:>6.1f} "
            f"{s['total_answer_length']/n:>7.0f} "
            f"{s['total_time']/n:>6.1f}s"
        )

    # Per-question analysis - which questions benefit most from more retrieval?
    print(f"\n\n--- Per-Question: Impact of Increasing Top-K ---")
    print(f"{'Question':<15} {'Type':<16} {'t=3 cite':>9} {'t=5 cite':>9} {'t=7 cite':>9} {'t=10 cite':>10} {'t=3 dp':>7} {'t=10 dp':>8} {'Δcite':>7}")
    print("-" * 100)

    for q in TEST_QUESTIONS:
        row = {"id": q["id"], "type": q["type"]}
        for tk in [3, 5, 7, 10]:
            matching = [r for r in all_results
                        if r.get("question_id") == q["id"]
                        and r.get("top_k") == tk
                        and "error" not in r]
            if matching:
                avg_cite = sum(r["citation_count"] for r in matching) / len(matching)
                avg_dp = sum(r["data_points_count"] for r in matching) / len(matching)
                row[f"cite_{tk}"] = avg_cite
                row[f"dp_{tk}"] = avg_dp
            else:
                row[f"cite_{tk}"] = -1
                row[f"dp_{tk}"] = -1

        delta = row.get("cite_10", 0) - row.get("cite_3", 0)
        print(
            f"{row['id']:<15} "
            f"{row['type']:<16} "
            f"{row.get('cite_3', -1):>9.1f} "
            f"{row.get('cite_5', -1):>9.1f} "
            f"{row.get('cite_7', -1):>9.1f} "
            f"{row.get('cite_10', -1):>10.1f} "
            f"{row.get('dp_3', -1):>7.0f} "
            f"{row.get('dp_10', -1):>8.0f} "
            f"{delta:>+7.1f}"
        )

    # Final recommendation
    print("\n\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Calculate net benefit of higher top-K
    topk_3_data = topk_stats.get(3, {})
    topk_10_data = topk_stats.get(10, {})
    if topk_3_data.get("total", 0) > 0 and topk_10_data.get("total", 0) > 0:
        n3 = topk_3_data["total"]
        n10 = topk_10_data["total"]
        src_3 = topk_3_data["total_sources"] / n3
        src_10 = topk_10_data["total_sources"] / n10
        cite_3 = topk_3_data["total_citations"] / n3
        cite_10 = topk_10_data["total_citations"] / n10
        time_3 = topk_3_data["total_time"] / n3
        time_10 = topk_10_data["total_time"] / n10
        refusal_3 = topk_3_data["refusals"] / n3 * 100
        refusal_10 = topk_10_data["refusals"] / n10 * 100

        print(f"\nTop-3 vs Top-10 comparison:")
        print(f"  Avg unique sources:  {src_3:.1f} → {src_10:.1f} (Δ {src_10 - src_3:+.1f})")
        print(f"  Avg citations:       {cite_3:.1f} → {cite_10:.1f} (Δ {cite_10 - cite_3:+.1f})")
        print(f"  Refusal rate:        {refusal_3:.0f}% → {refusal_10:.0f}%")
        print(f"  Avg response time:   {time_3:.1f}s → {time_10:.1f}s (Δ {time_10 - time_3:+.1f}s)")

        # Determine if increase is worth it
        source_gain = src_10 - src_3
        time_cost = time_10 - time_3
        refusal_improvement = refusal_3 - refusal_10

        if source_gain > 0.5 and refusal_improvement > 5:
            print("\n→ SIGNIFICANT BENEFIT from increasing top-K. Consider raising default from 3 to 5 or 7.")
        elif source_gain > 0.2:
            print("\n→ MODERATE BENEFIT from increasing top-K. May help for complex cross-reference questions.")
        else:
            print("\n→ MINIMAL BENEFIT from increasing top-K. Current default of 3 appears sufficient.")

        if time_cost > 5.0:
            print(f"  ⚠ However, response time increases by {time_cost:.1f}s which may impact UX.")

    # Save summary
    summary_file = output_dir / "summary.json"
    summary = {
        "by_config": config_stats,
        "by_depth": depth_stats,
        "by_topk": {str(k): v for k, v in topk_stats.items()},
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
