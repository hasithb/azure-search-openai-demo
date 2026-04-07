"""
A/B comparison: gpt-5.4-mini vs gpt-5.4 (full) for answer generation.

Uses the live app to retrieve sources (via gpt-5.4-mini query rewrite + hybrid search),
then re-generates the answer with gpt-5.4 (full) using the exact same sources and prompt.
Compares topic coverage scores.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from azure.identity import AzureDeveloperCliCredential
from openai import AzureOpenAI

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("APP_URL", "http://localhost:50505")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://cog-gz2m4s637t5me-us2.openai.azure.com/")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
TENANT_ID = os.getenv("AZURE_AUTH_TENANT_ID", "3bfe16b2-5fcc-4565-b1f1-15271d20fecf")
FULL_MODEL_DEPLOYMENT = "gpt-5.4"

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "app" / "backend" / "approaches" / "prompts" / "chat_answer.system.jinja2"

TESTS = [
    {"id": "CPR-01",   "question": "How do I get a court to decide my case quickly without a full trial?",
     "expected_topics": ["summary judgment", "no real prospect", "Part 24"]},
    {"id": "CPR-03",   "question": "What documents do I have to share with the other side in a lawsuit?",
     "expected_topics": ["disclosure", "inspection", "documents"]},
    {"id": "CPR-06",   "question": "Can I use an expert witness in my case and what rules apply to them?",
     "expected_topics": ["expert", "duty to the court", "report", "single joint expert"]},
    {"id": "CPR-10",   "question": "How do I apply for an injunction to stop someone doing something urgently?",
     "expected_topics": ["injunction", "without notice", "interim", "undertaking"]},
    {"id": "CPR-15",   "question": "How do I appeal a court decision I disagree with?",
     "expected_topics": ["appeal", "permission", "Part 52", "grounds"]},
    {"id": "CPR-16",   "question": "How is a civil case allocated to the right court track?",
     "expected_topics": ["small claims", "fast track", "multi-track", "allocation"]},
    {"id": "CPR-18",   "question": "How do I serve court documents on the other party?",
     "expected_topics": ["service", "claim form", "methods"]},
    {"id": "CCG-01",   "question": "How do I challenge an arbitration award in the Commercial Court?",
     "expected_topics": ["arbitration", "challenge", "award", "28 days", "section 67"]},
    {"id": "CCG-04",   "question": "How is a case managed in the Commercial Court from start to trial?",
     "expected_topics": ["case management", "conference", "list", "timetable"]},
    {"id": "TCC-01",   "question": "Do I have to try mediation before going to trial in a construction dispute?",
     "expected_topics": ["ADR", "mediation", "encouragement", "costs"]},
    {"id": "KBD-01",   "question": "How do I get someone released from unlawful detention?",
     "expected_topics": ["habeas corpus", "writ", "detention", "Administrative Court"]},
    {"id": "CROSS-06", "question": "What is the overriding objective and how does it affect court cases?",
     "expected_topics": ["overriding objective", "proportionate", "fair", "CPR"]},
    {"id": "CROSS-07", "question": "When can a freezing order be obtained to prevent someone moving assets?",
     "expected_topics": ["freezing", "order", "assets", "risk"]},
    {"id": "PAT-01",   "question": "How do I start a patent infringement case in court?",
     "expected_topics": ["patent", "infringement", "Patents Court"]},
    {"id": "CHAN-01",   "question": "How do I bring a trust dispute to the Chancery Division?",
     "expected_topics": ["Chancery", "trust", "claim form"]},
]


def score_answer(answer: str, expected_topics: list[str]) -> tuple[int, list[str]]:
    """Return (percentage, list_of_found_topics) for topic coverage."""
    lower = answer.lower()
    found = [t for t in expected_topics if t.lower() in lower]
    pct = int(100 * len(found) / len(expected_topics)) if expected_topics else 0
    return pct, found


def build_system_prompt(sources: list[str]) -> str:
    """Build a minimal rendered version of the system prompt for direct LLM calls."""
    # We use a simplified but faithful version of the Jinja2 template
    citations = [str(i) for i in range(1, len(sources) + 1)]
    citation_str = " ".join(f"[{c}]" for c in citations)

    return (
        "Assistant helps the company employees with their questions about English civil court procedure documents. "
        "Be brief in your answers.\n"
        "When a topic has multiple related procedural aspects covered in the sources, address each one briefly "
        "rather than only the main point. Aim to cover 3-4 key aspects of the topic rather than just 1-2.\n"
        "When the user asks about a legal concept in plain language but the sources use a specific legal term, "
        "include that term explicitly.\n"
        "Answer ONLY with the facts listed in the list of sources below. Do not generate answers that don't use the sources below.\n"
        "Each source is numbered (e.g., [1], [2], [3]) followed by colon and the actual information.\n"
        "Always include the source number for each fact you use in the response. Use square brackets to reference the source.\n"
        "CRITICAL - Citation formatting rules:\n"
        "- Each sentence must end with exactly one citation.\n"
        "- Do NOT combine multiple sources in a single sentence.\n"
        f"\nPossible citations for current question: {citation_str}\n"
    )


def build_user_message(question: str, sources: list[str]) -> str:
    """Build the user message with sources, matching the Jinja2 template."""
    parts = [question, "", "Sources:", ""]
    for s in sources:
        parts.append(s)
        parts.append("")
    return "\n".join(parts)


def main():
    # ── Azure OpenAI client for direct gpt-5.4 calls ───────────────────
    print("Setting up Azure OpenAI client for gpt-5.4 (full)...")
    cred = AzureDeveloperCliCredential(tenant_id=TENANT_ID)
    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_version=API_VERSION,
        azure_ad_token_provider=lambda: cred.get_token("https://cognitiveservices.azure.com/.default").token,
    )

    # ── Verify app is running ───────────────────────────────────────────
    print(f"Checking app at {BASE_URL}...")
    try:
        r = httpx.get(f"{BASE_URL}/", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: App not reachable at {BASE_URL}: {e}")
        sys.exit(1)
    print("App is running.\n")

    chat_payload_base = {
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
        }
    }

    results = []
    mini_scores = []
    full_scores = []

    print(f"Running {len(TESTS)} test cases...\n")
    print(f"{'ID':<12} {'mini%':>6} {'full%':>6} {'Δ':>5}  Question")
    print("-" * 90)

    for i, t in enumerate(TESTS):
        tid = t["id"]
        q = t["question"]
        expected = t["expected_topics"]

        # Step 1: Get answer + sources from app (gpt-5.4-mini)
        payload = dict(chat_payload_base)
        payload["messages"] = [{"content": q, "role": "user"}]

        try:
            with httpx.Client(timeout=120) as http:
                for attempt in range(3):
                    try:
                        resp = http.post(f"{BASE_URL}/chat", json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                        break
                    except Exception:
                        if attempt < 2:
                            time.sleep(10 * (attempt + 1))
                        else:
                            raise

            mini_answer = data.get("message", {}).get("content", "")
            # Extract text sources from the response
            context = data.get("context", {})
            data_points = context.get("data_points", {})
            text_sources = data_points.get("text", [])

            mini_pct, mini_found = score_answer(mini_answer, expected)

        except Exception as e:
            print(f"  {tid:<12} ERROR getting app response: {e}")
            results.append({"id": tid, "mini": -1, "full": -1, "delta": 0, "error": str(e)})
            continue

        if not text_sources:
            print(f"  {tid:<12} SKIP - no sources retrieved")
            results.append({"id": tid, "mini": mini_pct, "full": -1, "delta": 0, "note": "no sources"})
            mini_scores.append(mini_pct)
            continue

        # Step 2: Re-generate answer with gpt-5.4 (full) using same sources
        numbered_sources = []
        for idx, src in enumerate(text_sources, 1):
            numbered_sources.append(f"[{idx}]: {src}")

        system_msg = build_system_prompt(numbered_sources)
        user_msg = build_user_message(q, numbered_sources)

        try:
            completion = client.chat.completions.create(
                model=FULL_MODEL_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_completion_tokens=2048,
                temperature=0.3,
            )
            full_answer = completion.choices[0].message.content or ""
            full_pct, full_found = score_answer(full_answer, expected)

        except Exception as e:
            print(f"  {tid:<12} {mini_pct:>5}%   ERR   ---  {q[:50]}  (gpt-5.4 error: {e})")
            results.append({"id": tid, "mini": mini_pct, "full": -1, "delta": 0, "error": str(e)})
            mini_scores.append(mini_pct)
            continue

        delta = full_pct - mini_pct
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        tag = "▲" if delta > 0 else ("▼" if delta < 0 else "=")

        print(f"  {tid:<12} {mini_pct:>5}% {full_pct:>5}% {delta_str:>4}% {tag}  {q[:55]}")

        mini_scores.append(mini_pct)
        full_scores.append(full_pct)
        results.append({
            "id": tid,
            "question": q,
            "mini_score": mini_pct,
            "full_score": full_pct,
            "delta": delta,
            "expected_topics": expected,
            "mini_found": mini_found,
            "full_found": full_found,
            "num_sources": len(text_sources),
            "mini_answer_len": len(mini_answer),
            "full_answer_len": len(full_answer),
        })

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    valid_mini = [s for s in mini_scores if s >= 0]
    valid_full = [s for s in full_scores if s >= 0]
    mini_avg = sum(valid_mini) / len(valid_mini) if valid_mini else 0
    full_avg = sum(valid_full) / len(valid_full) if valid_full else 0

    print(f"\n  gpt-5.4-mini  avg: {mini_avg:.1f}%  ({len(valid_mini)} tests)")
    print(f"  gpt-5.4 full  avg: {full_avg:.1f}%  ({len(valid_full)} tests)")
    print(f"  Delta:             {full_avg - mini_avg:+.1f}%")

    improved = sum(1 for r in results if r.get("delta", 0) > 0)
    degraded = sum(1 for r in results if r.get("delta", 0) < 0)
    same = sum(1 for r in results if r.get("delta", 0) == 0 and "error" not in r)
    print(f"\n  Improved: {improved}  |  Same: {same}  |  Degraded: {degraded}")

    # Highlight biggest differences
    diffs = sorted([r for r in results if "delta" in r and r["delta"] != 0], key=lambda x: abs(x["delta"]), reverse=True)
    if diffs:
        print(f"\n  Biggest differences:")
        for d in diffs[:5]:
            print(f"    {d['id']:<12} {d.get('mini_score',0)}% → {d.get('full_score',0)}%  (Δ{d['delta']:+d}%)  {d.get('question','')[:60]}")

    # ── High-quality answer comparison: show answer excerpts for biggest deltas ──
    if diffs:
        print(f"\n{'='*90}")
        print("DETAILED ANSWER COMPARISON (top differences)")
        for d in diffs[:3]:
            print(f"\n  --- {d['id']}: {d.get('question','')} ---")
            print(f"  mini topics found: {d.get('mini_found', [])}")
            print(f"  full topics found: {d.get('full_found', [])}")
            missing_in_mini = set(d.get("expected_topics", [])) - set(d.get("mini_found", []))
            missing_in_full = set(d.get("expected_topics", [])) - set(d.get("full_found", []))
            if missing_in_mini:
                print(f"  mini MISSING: {missing_in_mini}")
            if missing_in_full:
                print(f"  full MISSING: {missing_in_full}")

    # ── Save results ────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "model_ab_mini_vs_full_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "mini_avg": round(mini_avg, 1),
                "full_avg": round(full_avg, 1),
                "delta": round(full_avg - mini_avg, 1),
                "improved": improved,
                "same": same,
                "degraded": degraded,
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
