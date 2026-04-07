"""A/B test: gpt-5.4 vs gpt-5.4-nano for query rewrite quality.

Compares both models on the same query rewrite prompt + tool schema,
measuring: search_query quality, legal_concept_analysis depth,
related_aspects coverage, subsection_hint accuracy, and latency.
"""

import json
import time

from azure.identity import AzureDeveloperCliCredential
from openai import AzureOpenAI

ENDPOINT = "https://cog-gz2m4s637t5me-us2.openai.azure.com/"
API_VERSION = "2024-12-01-preview"
TENANT_ID = "3bfe16b2-5fcc-4565-b1f1-15271d20fecf"

cred = AzureDeveloperCliCredential(tenant_id=TENANT_ID)
client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_version=API_VERSION,
    azure_ad_token_provider=lambda: cred.get_token("https://cognitiveservices.azure.com/.default").token,
)

# Load the actual prompt and tools from the repo
with open("app/backend/approaches/prompts/query_rewrite.system.jinja2") as f:
    SYSTEM_TEMPLATE = f.read()

with open("app/backend/approaches/prompts/chat_query_rewrite_tools.json") as f:
    TOOLS = json.load(f)

# Strip Jinja2 conditionals for a standalone test — use the fallback (no available_sources)
# Replace the Jinja2 block with the static fallback text
import re
# Remove Jinja2 tags, keep the else-branch content
system_prompt = SYSTEM_TEMPLATE
system_prompt = re.sub(
    r"\{%\s*if available_sources\s*%\}.*?\{%\s*else\s*%\}(.*?)\{%\s*endif\s*%\}",
    r"\1",
    system_prompt,
    flags=re.DOTALL,
)
# Clean remaining Jinja2 tags
system_prompt = re.sub(r"\{%.*?%\}", "", system_prompt)
system_prompt = re.sub(r"\{\{.*?\}\}", "", system_prompt)

MODELS = ["gpt-5.4", "gpt-5.4-nano"]

# Test cases: diverse query types that stress different rewrite capabilities
TESTS = [
    # Abbreviation disambiguation
    {"id": "ABBREV-1", "q": "What is PAD?", "expect_query": "CPR 31.16", "expect_not": "Pre-Action Protocol"},
    {"id": "ABBREV-2", "q": "Tell me about RFS", "expect_query": "CPR 3.9", "expect_not": None},
    {"id": "ABBREV-3", "q": "What is SJ?", "expect_query": "Part 24", "expect_not": None},
    # Broad topics needing related_aspects
    {"id": "BROAD-1", "q": "What documents do I have to share with the other side in a lawsuit?", "expect_query": "disclosure", "expect_not": None},
    {"id": "BROAD-2", "q": "What are the rules about expert witnesses?", "expect_query": "Part 35", "expect_not": None},
    {"id": "BROAD-3", "q": "What steps before starting a court claim?", "expect_query": "pre-action protocol", "expect_not": None},
    # Court-guide-specific queries needing naming convention
    {"id": "COURT-1", "q": "What does section B.3 of the Commercial Court Guide say?", "expect_query": "B.3", "expect_not": None},
    {"id": "COURT-2", "q": "What does the TCC Guide say about expert evidence?", "expect_query": "Section 13", "expect_not": None},
    {"id": "COURT-3", "q": "How does the Chancery Guide handle disclosure?", "expect_query": "Chapter 7", "expect_not": None},
    # Disambiguation traps
    {"id": "DISAMB-1", "q": "Tell me about thorough disclosure", "expect_query": "PD 57AD|Practice Direction 57AD", "expect_not": "CPR 31.6"},
    {"id": "DISAMB-2", "q": "Tell me about pre-action disclosure", "expect_query": "CPR 31.16", "expect_not": "Pre-Action Protocol"},
    # Subsection-specific
    {"id": "SUB-1", "q": "What does CPR 3.9 say about relief from sanctions?", "expect_query": "3.9", "expect_not": None},
    {"id": "SUB-2", "q": "What is rule 24.2?", "expect_query": "24.2", "expect_not": None},
    # Broad cross-cutting
    {"id": "CROSS-1", "q": "How do I appeal a court decision?", "expect_query": "Part 52", "expect_not": None},
    {"id": "CROSS-2", "q": "When can a freezing order be obtained?", "expect_query": "Part 25", "expect_not": None},
    {"id": "CROSS-3", "q": "How do I serve court documents?", "expect_query": "Part 6", "expect_not": None},
    # Non-legal (should still produce a query)
    {"id": "OFFTOPIC-1", "q": "What are my health plans?", "expect_query": "health plan", "expect_not": None},
]

print(f"{'ID':<12} {'Model':<14} {'ms':>6} {'Query OK':>8} {'Avoid OK':>9} {'Aspects':>8} {'Sub':>5} | search_query (truncated)")
print("-" * 120)

results = {m: {"pass": 0, "fail": 0, "avoid_pass": 0, "avoid_fail": 0, "has_aspects": 0, "has_sub": 0, "times": []} for m in MODELS}

for t in TESTS:
    for model in MODELS:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": t["q"]},
        ]
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                tools=TOOLS,
                tool_choice={"type": "function", "function": {"name": "search_sources"}},
                max_completion_tokens=500,
                temperature=0.0,
                reasoning_effort="none" if "5.4" in model else "low",
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(f"  {t['id']:<12} {model:<14} ERROR: {e}")
            continue

        results[model]["times"].append(elapsed_ms)

        # Parse tool call
        args = {}
        if resp.choices[0].message.tool_calls:
            raw = resp.choices[0].message.tool_calls[0].function.arguments
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                pass

        search_query = args.get("search_query", resp.choices[0].message.content or "")
        related = args.get("related_aspects", "")
        subsection = args.get("subsection_hint", "")

        # Check expect_query (any alternative separated by |)
        expect_alts = t["expect_query"].split("|")
        query_ok = any(alt.lower() in search_query.lower() for alt in expect_alts)
        if query_ok:
            results[model]["pass"] += 1
        else:
            results[model]["fail"] += 1

        # Check expect_not
        avoid_ok = True
        if t["expect_not"]:
            avoid_alts = t["expect_not"].split("|")
            if any(alt.lower() in search_query.lower() for alt in avoid_alts):
                avoid_ok = False
                results[model]["avoid_fail"] += 1
            else:
                results[model]["avoid_pass"] += 1

        has_aspects = bool(related and related.strip())
        has_sub = bool(subsection and subsection.strip())
        if has_aspects:
            results[model]["has_aspects"] += 1
        if has_sub:
            results[model]["has_sub"] += 1

        q_tag = "✓" if query_ok else "✗"
        a_tag = "✓" if avoid_ok else "✗"
        asp_tag = "Y" if has_aspects else "-"
        sub_tag = "Y" if has_sub else "-"
        trunc_q = search_query[:60]
        print(f"  {t['id']:<12} {model:<14} {elapsed_ms:>5.0f}  {q_tag:>8} {a_tag:>9} {asp_tag:>8} {sub_tag:>5} | {trunc_q}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
n = len(TESTS)
for model in MODELS:
    r = results[model]
    avg_ms = sum(r["times"]) / len(r["times"]) if r["times"] else 0
    p50 = sorted(r["times"])[len(r["times"]) // 2] if r["times"] else 0
    total_avoid = r["avoid_pass"] + r["avoid_fail"]
    avoid_str = f"{r['avoid_pass']}/{total_avoid}" if total_avoid > 0 else "n/a"
    print(f"\n  {model}:")
    print(f"    Query accuracy:  {r['pass']}/{n} ({100*r['pass']/n:.0f}%)")
    print(f"    Avoid traps:     {avoid_str}")
    print(f"    Has aspects:     {r['has_aspects']}/{n}")
    print(f"    Has subsection:  {r['has_sub']}/{n}")
    print(f"    Latency avg:     {avg_ms:.0f} ms")
    print(f"    Latency p50:     {p50:.0f} ms")

# Token cost comparison
if all(results[m]["times"] for m in MODELS):
    full_avg = sum(results["gpt-5.4"]["times"]) / len(results["gpt-5.4"]["times"])
    nano_avg = sum(results["gpt-5.4-nano"]["times"]) / len(results["gpt-5.4-nano"]["times"])
    print(f"\n  Latency reduction: {full_avg - nano_avg:.0f} ms ({100*(full_avg-nano_avg)/full_avg:.0f}% faster)")
    print(f"  Cost reduction:    ~90% (nano is ~$0.10/1M in vs $2.50/1M in for full)")
