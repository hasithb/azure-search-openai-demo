"""
Prompt variant A/B test – round 2.

Builds on the B_completeness amendment that is now permanently applied.
Tests four NEW amendments layered on top of the current prompt.

The script:
  1. Backs up the current prompt
  2. For "current" baseline, runs without modification
  3. For each variant, inserts amendment text after the completeness block
  4. Restores original prompt when done

Run:
    source .venv/bin/activate && python scripts/test_prompt_variants_v2.py
"""

import json
import os
import re
import shutil
import time

import httpx

BASE_URL = "http://localhost:50505"
PROMPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "app",
    "backend",
    "approaches",
    "prompts",
    "chat_answer.system.jinja2",
)
PROMPT_PATH = os.path.normpath(PROMPT_PATH)
BACKUP_PATH = PROMPT_PATH + ".bak"

CITATION_REGEX = re.compile(r"\[[\w\s.#=()_:-]+\]")

# ---------- Expanded test questions: 25 total ----------
# Original 15 (persistent misses) + 10 new questions
TEST_QUESTIONS = [
    # ---- Original 15 from round 1 ----
    {
        "id": "NL-CPR-03",
        "question": "What documents do I have to share with the other side in a lawsuit?",
        "expected_topics": ["disclosure", "inspection", "documents"],
    },
    {
        "id": "NL-CPR-06",
        "question": "Can I use an expert witness in my case and what rules apply to them?",
        "expected_topics": ["expert", "duty to the court", "report", "single joint expert"],
    },
    {
        "id": "NL-CPR-07",
        "question": "What steps do I need to take before starting a court claim?",
        "expected_topics": ["pre-action", "protocol", "letter of claim", "response"],
    },
    {
        "id": "NL-CPR-10",
        "question": "How do I apply for an injunction to stop someone doing something urgently?",
        "expected_topics": ["injunction", "without notice", "interim", "undertaking"],
    },
    {
        "id": "NL-CPR-11",
        "question": "What happens if someone doesn't follow a court order?",
        "expected_topics": ["contempt", "committal", "unless order"],
    },
    {
        "id": "NL-CPR-14",
        "question": "How do I get the court to order the other side to pay my legal costs?",
        "expected_topics": ["costs", "assessment", "indemnity"],
    },
    {
        "id": "NL-CCG-01",
        "question": "How do I challenge an arbitration award in the Commercial Court?",
        "expected_topics": ["arbitration", "challenge", "award", "28 days", "section 67"],
    },
    {
        "id": "NL-CCG-03",
        "question": "What are the rules about sharing documents in commercial litigation?",
        "expected_topics": ["disclosure", "documents", "proportionality", "overriding objective"],
    },
    {
        "id": "NL-TCC-01",
        "question": "Do I have to try mediation before going to trial in a construction dispute?",
        "expected_topics": ["ADR", "mediation", "encouragement", "costs"],
    },
    {
        "id": "NL-TCC-02",
        "question": "What are the rules about using expert witnesses in a building dispute?",
        "expected_topics": ["expert", "independent", "duty", "court", "report"],
    },
    {
        "id": "NL-KBD-01",
        "question": "How do I get someone released from unlawful detention?",
        "expected_topics": ["habeas corpus", "writ", "detention", "Administrative Court"],
    },
    {
        "id": "NL-CROSS-01",
        "question": "How do I make a witness give evidence if they don't want to?",
        "expected_topics": ["witness", "summons", "subpoena", "compel"],
    },
    {
        "id": "NL-CPR-01",
        "question": "How do I get a court to decide my case quickly without a full trial?",
        "expected_topics": ["summary judgment", "no real prospect", "Part 24"],
    },
    {
        "id": "NL-CROSS-04",
        "question": "What are the time limits for starting different types of court claims?",
        "expected_topics": ["limitation", "time", "months"],
    },
    {
        "id": "NL-CROSS-05",
        "question": "How do courts decide who pays the legal costs at the end of a case?",
        "expected_topics": ["costs", "loser pays", "discretion"],
    },
    # ---- 10 new questions for round 2 ----
    {
        "id": "NL-CPR-15",
        "question": "How do I appeal a court decision I disagree with?",
        "expected_topics": ["appeal", "permission", "Part 52", "grounds"],
    },
    {
        "id": "NL-CPR-16",
        "question": "How is a civil case allocated to the right court track?",
        "expected_topics": ["small claims", "fast track", "multi-track", "allocation"],
    },
    {
        "id": "NL-CPR-17",
        "question": "What are my options if the other side is destroying or hiding evidence?",
        "expected_topics": ["search order", "preservation", "injunction"],
    },
    {
        "id": "NL-CPR-18",
        "question": "How do I serve court documents on the other party?",
        "expected_topics": ["service", "claim form", "methods"],
    },
    {
        "id": "NL-CCG-04",
        "question": "How is a case managed in the Commercial Court from start to trial?",
        "expected_topics": ["case management", "conference", "list", "timetable"],
    },
    {
        "id": "NL-CHAN-01",
        "question": "How do I bring a trust dispute to the Chancery Division?",
        "expected_topics": ["Chancery", "trust", "claim form"],
    },
    {
        "id": "NL-TCC-03",
        "question": "How are costs budgets managed in large construction cases?",
        "expected_topics": ["costs", "budget", "management"],
    },
    {
        "id": "NL-PAT-01",
        "question": "How do I start a patent infringement case in court?",
        "expected_topics": ["patent", "infringement", "Patents Court"],
    },
    {
        "id": "NL-CROSS-06",
        "question": "What is the overriding objective and how does it affect court cases?",
        "expected_topics": ["overriding objective", "proportionate", "fair", "CPR"],
    },
    {
        "id": "NL-CROSS-07",
        "question": "When can a freezing order be obtained to prevent someone moving assets?",
        "expected_topics": ["freezing", "order", "assets", "risk"],
    },
]


# ---------- New variants: all build on the current prompt (which already has B) ----------

# Anchor: we insert AFTER the B completeness block, before "Answer ONLY..."
ANCHOR_LINE = "Aim to cover 3-4 key aspects of the topic rather than just 1-2."

VARIANTS = {
    "current": "",
    "D_terminology": """
When the user asks about a legal concept in plain language but the sources use a specific legal term, include that term explicitly. For example: if asked about "not following a court order" and the sources discuss "contempt of court" and "committal", use those terms. If asked about "making a witness attend" and the sources refer to a "witness summons", use that phrase. Always pair the legal term with any plain-English explanation.""",
    "E_consequences": """
When explaining a legal procedure or remedy, also briefly address: (a) key duties or obligations that apply to the parties (e.g. an expert's duty to the court, the applicant's duty of full and frank disclosure), (b) what happens if a party fails to comply (e.g. striking out, unless orders, contempt), and (c) any specific time limits or deadlines mentioned in the sources.""",
    "F_principles": """
When the sources reference well-known legal principles, name them explicitly (e.g. "the loser pays principle", "the overriding objective of dealing with cases justly and at proportionate cost", "the duty of full and frank disclosure"). Include specific numerical thresholds and time limits from the sources (e.g. "28 days", "21 days for acknowledgment of service", "14 days to file a defence").""",
    "G_structured": """
Structure your answer to cover these aspects when relevant: (1) what the procedure or remedy is and when it applies, (2) key requirements and conditions for obtaining it, (3) duties and obligations on the parties involved, and (4) consequences of non-compliance and relevant time limits. Keep each point concise — one to two sentences.""",
}


def send_question(question: str) -> dict:
    payload = {
        "messages": [{"content": question, "role": "user"}],
        "context": {
            "overrides": {
                "top": 5,
                "retrieval_mode": "hybrid",
                "semantic_ranker": True,
                "semantic_captions": False,
                "query_rewriting": True,
                "suggest_followup_questions": False,
                "use_oid_security_filter": False,
                "use_groups_security_filter": False,
                "search_text_embeddings": True,
                "send_text_sources": True,
                "language": "en",
                "use_agentic_knowledgebase": False,
            }
        },
    }
    with httpx.Client(timeout=120.0) as client:
        for attempt in range(3):
            try:
                resp = client.post(f"{BASE_URL}/chat", json=payload)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))
                else:
                    raise


def evaluate(test_case: dict, response: dict) -> dict:
    answer = response.get("message", {}).get("content", "")
    answer_lower = answer.lower()
    topics_found = [t for t in test_case["expected_topics"] if t.lower() in answer_lower]
    topics_missing = [t for t in test_case["expected_topics"] if t.lower() not in answer_lower]
    coverage = len(topics_found) / len(test_case["expected_topics"])
    citations = CITATION_REGEX.findall(answer)
    return {
        "test_id": test_case["id"],
        "topic_coverage": round(coverage, 2),
        "topics_found": topics_found,
        "topics_missing": topics_missing,
        "citation_count": len(citations),
        "word_count": len(answer.split()),
    }


def apply_variant(amendment_text: str) -> None:
    """Patch the prompt file: insert amendment after the anchor line."""
    with open(BACKUP_PATH) as f:
        content = f.read()

    if amendment_text:
        patched = content.replace(
            ANCHOR_LINE,
            ANCHOR_LINE + "\n" + amendment_text.strip(),
            1,
        )
    else:
        patched = content

    with open(PROMPT_PATH, "w") as f:
        f.write(patched)


def wait_for_reload() -> None:
    time.sleep(6)
    for _ in range(5):
        try:
            with httpx.Client(timeout=10.0) as c:
                r = c.get(f"{BASE_URL}/config")
                if r.status_code == 200:
                    return
        except Exception:
            pass
        time.sleep(2)


def run_variant(variant_name: str) -> list[dict]:
    results = []
    for i, test in enumerate(TEST_QUESTIONS):
        print(f"  [{i+1}/{len(TEST_QUESTIONS)}] {test['id']}...", end=" ", flush=True)
        start = time.time()
        resp = send_question(test["question"])
        elapsed = time.time() - start
        result = evaluate(test, resp)
        result["latency"] = round(elapsed, 1)
        results.append(result)
        status = "✓" if result["topic_coverage"] >= 0.75 else "△" if result["topic_coverage"] >= 0.5 else "✗"
        print(f"{status} {result['topic_coverage']:.0%} ({elapsed:.1f}s)")
        time.sleep(1)
    return results


def main() -> None:
    print("=" * 70)
    print("PROMPT VARIANT A/B TEST — ROUND 2")
    print(f"Testing {len(TEST_QUESTIONS)} questions × {len(VARIANTS)} variants")
    print("=" * 70)

    # Backup original
    shutil.copy2(PROMPT_PATH, BACKUP_PATH)
    print(f"Backed up prompt to {BACKUP_PATH}")

    all_results: dict[str, list[dict]] = {}

    try:
        for variant_name, amendment in VARIANTS.items():
            print(f"\n{'─'*70}")
            print(f"VARIANT: {variant_name}")
            print(f"{'─'*70}")

            apply_variant(amendment)

            print("Waiting for hot-reload...")
            wait_for_reload()

            results = run_variant(variant_name)
            all_results[variant_name] = results
    finally:
        # Restore original
        shutil.copy2(BACKUP_PATH, PROMPT_PATH)
        os.remove(BACKUP_PATH)
        print(f"\nRestored original prompt.")

    # ---------- Comparison ----------
    print("\n" + "=" * 110)
    print("COMPARISON TABLE")
    print("=" * 110)

    variant_names = list(all_results.keys())
    header = f"{'Test ID':<16}"
    for v in variant_names:
        header += f" {v:>18}"
    print(header)
    print("-" * 110)

    for i, test in enumerate(TEST_QUESTIONS):
        row = f"{test['id']:<16}"
        for v in variant_names:
            r = all_results[v][i]
            coverage = r["topic_coverage"]
            row += f" {coverage:>17.0%}"
        print(row)

        # Show topic changes vs current baseline
        base_missing = set(all_results["current"][i]["topics_missing"])
        for v in variant_names[1:]:
            v_missing = set(all_results[v][i]["topics_missing"])
            newly_found = base_missing - v_missing
            newly_lost = v_missing - base_missing
            if newly_found:
                print(f"  {v}: ✅ found {sorted(newly_found)}")
            if newly_lost:
                print(f"  {v}: ❌ lost {sorted(newly_lost)}")

    # Summary stats
    print("\n" + "-" * 110)
    print(f"{'SUMMARY':<16}", end="")
    for v in variant_names:
        results = all_results[v]
        avg_cov = sum(r["topic_coverage"] for r in results) / len(results)
        print(f" {avg_cov:>17.0%}", end="")
    print()

    print(f"{'Avg latency':<16}", end="")
    for v in variant_names:
        results = all_results[v]
        avg_lat = sum(r["latency"] for r in results) / len(results)
        print(f" {avg_lat:>16.1f}s", end="")
    print()

    print(f"{'Avg words':<16}", end="")
    for v in variant_names:
        results = all_results[v]
        avg_wc = sum(r["word_count"] for r in results) / len(results)
        print(f" {avg_wc:>16.0f}", end="")
    print()

    print(f"{'>=75% topics':<16}", end="")
    for v in variant_names:
        results = all_results[v]
        above = sum(1 for r in results if r["topic_coverage"] >= 0.75)
        print(f" {above:>13}/{len(results)}", end="")
    print()

    print(f"{'100% topics':<16}", end="")
    for v in variant_names:
        results = all_results[v]
        perfect = sum(1 for r in results if r["topic_coverage"] >= 1.0)
        print(f" {perfect:>13}/{len(results)}", end="")
    print()

    # Pairwise improvements vs current
    print("\n--- Improvements vs Current ---")
    for v in variant_names[1:]:
        better = worse = same = 0
        for i in range(len(TEST_QUESTIONS)):
            bc = all_results["current"][i]["topic_coverage"]
            vc = all_results[v][i]["topic_coverage"]
            if vc > bc + 0.001:
                better += 1
            elif vc < bc - 0.001:
                worse += 1
            else:
                same += 1
        print(f"  {v}: {better} improved, {worse} regressed, {same} same")

    # Best variant per question
    print("\n--- Best Variant Per Question ---")
    for i, test in enumerate(TEST_QUESTIONS):
        best_v = max(variant_names, key=lambda v: all_results[v][i]["topic_coverage"])
        best_cov = all_results[best_v][i]["topic_coverage"]
        current_cov = all_results["current"][i]["topic_coverage"]
        if best_cov > current_cov + 0.001:
            print(f"  {test['id']}: {best_v} ({best_cov:.0%} vs current {current_cov:.0%})")

    # Save
    output = {
        "test_count": len(TEST_QUESTIONS),
        "variants": {v: all_results[v] for v in variant_names},
        "amendments": {v: VARIANTS[v] for v in variant_names},
    }
    out_path = "scripts/prompt_variant_results_v2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
