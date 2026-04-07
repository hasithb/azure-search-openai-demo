"""
A/B comparison test: old prompt vs. new prompt for disambiguation and mismatch detection.

Sends the same questions twice — once with the old system prompt (via override_prompt),
once with the new default prompt — and compares the quality of responses.

Run with: python scripts/test_prompt_ab_compare.py
Requires: app running at http://localhost:50505
"""

import json
import re
import sys
import httpx

BASE_URL = "http://localhost:50505"

OLD_PROMPT = """Assistant helps the company employees with their questions about internal documents. Be brief in your answers.
Answer ONLY with the facts listed in the list of sources below. Do not generate answers that don't use the sources below.
CRITICAL - Source relevance check: Before citing any source, verify it is genuinely about the topic the user is asking about — not merely a document that mentions the term in passing. For example, if asked about "pre-action disclosure" (a court application under CPR 31.16 for disclosure before proceedings start), do NOT cite Pre-Action Protocol documents that only mention "pre-action disclosure" incidentally as one of many remedies. A source that briefly references a concept is not a source that explains that concept. Only cite sources where the topic is substantively addressed.
If after filtering out irrelevant sources there is not enough information to fully answer the question, clearly state that the specific information requested could not be found in the available sources. Then present whatever related information IS available from the relevant sources, and explain what is missing. For example: "The sources available do not directly address [specific topic]. However, the following related information was found: ..."
If asking a clarifying question to the user would help, ask the question.
If the question is not in English, answer in the language used in the question.
Each source is numbered (e.g., [1], [2], [3]) followed by colon and the actual information. Always include the source number for each fact you use in the response. Use square brackets to reference the source, for example [1]. Don't combine sources, list each source separately, for example [1][2]."""


# Targeted questions that probe the new capabilities
QUESTIONS = [
    {
        "question": "What is standard disclosure?",
        "ideal_behavior": "Should reference CPR 31.6 and distinguish from extended disclosure under PD 57AD",
        "quality_checks": {
            "mentions_cpr31": lambda a: bool(re.search(r"(CPR\s*)?Part\s*31|CPR\s*31|31\.6", a, re.I)),
            "mentions_pd57ad_distinction": lambda a: "57ad" in a.lower() or "extended disclosure" in a.lower(),
            "has_citation": lambda a: bool(re.search(r'\[\d+\]', a)),
        },
    },
    {
        "question": "Tell me about thorough disclosure requirements",
        "ideal_behavior": "Should explain this maps to 'extended disclosure' under PD 57AD, not standard CPR 31",
        "quality_checks": {
            "maps_to_extended": lambda a: "extended" in a.lower() or "57ad" in a.lower(),
            "mentions_models": lambda a: any(w in a.lower() for w in ["model", "models"]),
            "has_citation": lambda a: bool(re.search(r'\[\d+\]', a)),
        },
    },
    {
        "question": "What are the time limits?",
        "ideal_behavior": "Should clarify the question is broad and mention specific time limit contexts",
        "quality_checks": {
            "provides_some_info": lambda a: len(a) > 50,
            "suggests_specificity": lambda a: any(w in a.lower() for w in ["specific", "specify", "depend", "which", "context", "various", "clarif"]),
            "has_citation": lambda a: bool(re.search(r'\[\d+\]', a)),
        },
    },
    {
        "question": "What is the weather forecast for London?",
        "ideal_behavior": "Should clearly state this info is not in the sources",
        "quality_checks": {
            "notes_unavailable": lambda a: any(w in a.lower() for w in ["not", "cannot", "no information", "can't", "don't have", "does not", "unavailable"]),
        },
    },
    {
        "question": "How do costs work?",
        "ideal_behavior": "Should reference costs rules and potentially clarify different aspects of costs",
        "quality_checks": {
            "mentions_costs_parts": lambda a: any(w in a for w in ["Part 44", "Part 45", "Part 46", "Part 47"]),
            "has_citation": lambda a: bool(re.search(r'\[\d+\]', a)),
        },
    },
]


def send_chat(question: str, prompt_override: str | None = None) -> dict:
    overrides = {
        "retrieval_mode": "hybrid",
        "semantic_ranker": True,
        "semantic_captions": False,
        "top": 5,
        "suggest_followup_questions": False,
        "seed": 42,
    }
    if prompt_override:
        overrides["prompt_template"] = prompt_override

    payload = {
        "messages": [{"content": question, "role": "user"}],
        "context": {"overrides": overrides},
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{BASE_URL}/chat", json=payload)
        resp.raise_for_status()
        return resp.json()


def main():
    print("=" * 80)
    print("A/B PROMPT COMPARISON: Old vs. New")
    print("=" * 80)

    old_total = 0
    old_pass = 0
    new_total = 0
    new_pass = 0
    comparison_results = []

    for i, tc in enumerate(QUESTIONS, 1):
        q = tc["question"]
        checks = tc["quality_checks"]
        ideal = tc["ideal_behavior"]

        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(QUESTIONS)}] Q: {q}")
        print(f"  Ideal: {ideal}")
        print()

        # --- OLD PROMPT ---
        try:
            old_result = send_chat(q, prompt_override=OLD_PROMPT)
            old_answer = old_result.get("message", {}).get("content", "")
        except Exception as e:
            old_answer = f"[ERROR: {e}]"

        old_scores = {}
        for name, check_fn in checks.items():
            ok = check_fn(old_answer)
            old_scores[name] = ok
            old_total += 1
            if ok:
                old_pass += 1

        # --- NEW PROMPT (default) ---
        try:
            new_result = send_chat(q)
            new_answer = new_result.get("message", {}).get("content", "")
        except Exception as e:
            new_answer = f"[ERROR: {e}]"

        new_scores = {}
        for name, check_fn in checks.items():
            ok = check_fn(new_answer)
            new_scores[name] = ok
            new_total += 1
            if ok:
                new_pass += 1

        # Print comparison
        print(f"  OLD PROMPT answer ({len(old_answer)} chars):")
        print(f"    {old_answer[:180].replace(chr(10), ' ')}{'...' if len(old_answer) > 180 else ''}")
        print(f"  NEW PROMPT answer ({len(new_answer)} chars):")
        print(f"    {new_answer[:180].replace(chr(10), ' ')}{'...' if len(new_answer) > 180 else ''}")
        print()

        print(f"  {'Check':<35} {'OLD':>6} {'NEW':>6} {'Delta':>6}")
        print(f"  {'─' * 55}")
        for name in checks:
            o = "PASS" if old_scores[name] else "FAIL"
            n = "PASS" if new_scores[name] else "FAIL"
            delta = ""
            if old_scores[name] != new_scores[name]:
                delta = "+NEW" if new_scores[name] else "+OLD"
            print(f"  {name:<35} {o:>6} {n:>6} {delta:>6}")

        comparison_results.append({
            "question": q,
            "old_scores": old_scores,
            "new_scores": new_scores,
            "old_answer_len": len(old_answer),
            "new_answer_len": len(new_answer),
        })

    # Final summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    old_rate = round(100 * old_pass / old_total) if old_total else 0
    new_rate = round(100 * new_pass / new_total) if new_total else 0
    print(f"  OLD prompt: {old_pass}/{old_total} checks passed ({old_rate}%)")
    print(f"  NEW prompt: {new_pass}/{new_total} checks passed ({new_rate}%)")
    delta = new_rate - old_rate
    direction = "IMPROVED" if delta > 0 else ("SAME" if delta == 0 else "REGRESSED")
    print(f"  Delta: {'+' if delta > 0 else ''}{delta}% ({direction})")

    # Save
    output_path = "scripts/prompt_ab_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "old_prompt_pass_rate": old_rate,
            "new_prompt_pass_rate": new_rate,
            "delta_pct": delta,
            "details": comparison_results,
        }, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
