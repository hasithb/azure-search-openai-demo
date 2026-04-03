#!/usr/bin/env python3
"""A/B comparison of old vs new query rewrite prompts.

Calls Azure OpenAI directly with both prompt variants on the same set
of test questions, so we can measure query quality improvement without
being affected by search index behaviour.
"""

import asyncio
import json
import os
import re
import sys
import time

import dotenv

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from azure.identity import AzureDeveloperCliCredential
from openai import AsyncAzureOpenAI

# ---------- config ----------

ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT", "gpt-5.4-nano")

# ---------- prompts ----------

OLD_SYSTEM = """\
Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base.
You have access to an Azure AI Search index containing English civil court procedure documents: the Civil Procedure Rules (CPR Parts 1-89), Practice Directions, Pre-Action Protocols, and Court Guides (Commercial, Chancery, King's Bench, TCC, Patents, Circuit Commercial).
Generate a search query based on the conversation and the new question.
Do not include cited source filenames and document names e.g. info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Do not include any special characters like '+'.
If the question is not in English, translate the question to English before generating the search query.
If you cannot generate a search query, return just the number 0.
Use the search_sources function to return the query.
Set subsection_hint to an empty string unless the user names a specific rule, paragraph, or subsection.

IMPORTANT - Legal term disambiguation:
When the user uses an abbreviation, acronym, or shorthand, expand it to the precise legal concept and target the correct CPR rule or Practice Direction. Common examples:
- "PAD" or "pre-action disclosure" = disclosure before proceedings have started under CPR 31.16 (NOT Pre-Action Protocols, which are a different concept about pre-litigation conduct)
- "summary judgment" or "SJ" = application under CPR Part 24 where there is no real prospect of success
- "default judgment" or "DJ" = judgment without trial under CPR Part 12
- "unless order" = order with automatic sanction under CPR 3.1
- "relief from sanctions" or "RFS" = application under CPR 3.9
- "strike out" = CPR 3.4 power to strike out statements of case
- "Norwich Pharmacal" or "NPO" = third party disclosure order, CPR 31.18
- "freezing order" or "freezing injunction" = interim remedy under CPR Part 25
- "WPS" or "without prejudice save as to costs" = Part 36 offers and related costs provisions
- "CMC" = case management conference under CPR Part 29
- "PTR" = pre-trial review
- "ADR" = alternative dispute resolution
Be careful not to confuse similarly named concepts. For example, "pre-action disclosure" (a court application to get documents before issuing proceedings, CPR 31.16) is entirely different from "Pre-Action Protocols" (codes of conduct about steps before litigation).

Generate search query for: How did crypto do last year?

Search query: Summarize Cryptocurrency Market Dynamics from last year

Generate search query for: What are my health plans?

Search query: Show available health plans

Generate search query for: What is PAD?

Search query: CPR 31.16 pre-action disclosure application before proceedings have started

Generate search query for: Tell me about pre-action disclosure

Search query: CPR 31.16 disclosure before proceedings have started application for pre-action disclosure

Generate search query for: What are the rules on summary judgment?

Search query: CPR Part 24 summary judgment no real prospect of succeeding compelling reason for trial

Generate search query for: How do I get relief from sanctions?

Search query: CPR 3.9 relief from sanctions application unless order"""

OLD_TOOLS = json.loads("""\
[{
    "type": "function",
    "function": {
        "name": "search_sources",
        "description": "Retrieve sources from the Azure AI Search index containing CPR Parts, Practice Directions, Pre-Action Protocols, and Court Guides. Always expand abbreviations and acronyms to their full legal meaning with the relevant CPR rule number.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "Query string to retrieve documents from azure search. Include the specific CPR rule or Part number when known."
                },
                "subsection_hint": {
                    "type": "string",
                    "description": "Specific subsection identifier. Use an empty string when no precise subsection is named."
                }
            },
            "required": ["search_query"]
        }
    }
}]""")

NEW_SYSTEM = """\
Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base.
You have access to an Azure AI Search index containing English civil court procedure documents: the Civil Procedure Rules (CPR Parts 1-89), Practice Directions, Pre-Action Protocols, and Court Guides (Commercial, Chancery, King's Bench, TCC, Patents, Circuit Commercial).
Generate a search query based on the conversation and the new question.
Do not include cited source filenames and document names e.g. info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Do not include any special characters like '+'.
If the question is not in English, translate the question to English before generating the search query.
If you cannot generate a search query, return just the number 0.
Use the search_sources function to return the query.
Set subsection_hint to an empty string unless the user names a specific rule, paragraph, or subsection.

CRITICAL - Knowledge-grounded query generation:
You MUST use the legal_concept_analysis parameter to reason about the question BEFORE writing the search query.
Apply your general knowledge of English civil procedure to:
1. Identify the SPECIFIC CPR rule, Practice Direction, or court guide section that is the PRIMARY authoritative source for the topic
2. List key legal terms, tests, or definitions that would appear in that authoritative source document
3. Note any common confusions with similarly-named documents or concepts that the search should AVOID matching

Your search_query MUST then target the specific authoritative source you identified, using the precise CPR rule numbers and distinctive terms from your analysis. This ensures the search retrieves the primary source rather than documents that merely mention the topic in passing.

IMPORTANT - Legal term disambiguation:
When the user uses an abbreviation, acronym, or shorthand, expand it to the precise legal concept and target the correct CPR rule or Practice Direction. Common examples:
- "PAD" or "pre-action disclosure" = disclosure before proceedings have started under CPR 31.16 (NOT Pre-Action Protocols, which are a different concept about pre-litigation conduct)
- "summary judgment" or "SJ" = application under CPR Part 24 where there is no real prospect of success
- "default judgment" or "DJ" = judgment without trial under CPR Part 12
- "unless order" = order with automatic sanction under CPR 3.1
- "relief from sanctions" or "RFS" = application under CPR 3.9
- "strike out" = CPR 3.4 power to strike out statements of case
- "Norwich Pharmacal" or "NPO" = third party disclosure order, CPR 31.18
- "freezing order" or "freezing injunction" = interim remedy under CPR Part 25
- "WPS" or "without prejudice save as to costs" = Part 36 offers and related costs provisions
- "CMC" = case management conference under CPR Part 29
- "PTR" = pre-trial review
- "ADR" = alternative dispute resolution
Be careful not to confuse similarly named concepts. For example, "pre-action disclosure" (a court application to get documents before issuing proceedings, CPR 31.16) is entirely different from "Pre-Action Protocols" (codes of conduct about steps before litigation).

Generate search query for: How did crypto do last year?

Search query: Summarize Cryptocurrency Market Dynamics from last year

Generate search query for: What are my health plans?

Search query: Show available health plans

Generate search query for: What is PAD?

Search query: CPR 31.16 pre-action disclosure application before proceedings have started

Generate search query for: Tell me about pre-action disclosure

Search query: CPR 31.16 disclosure before proceedings have started application for pre-action disclosure

Generate search query for: What are the rules on summary judgment?

Search query: CPR Part 24 summary judgment no real prospect of succeeding compelling reason for trial

Generate search query for: How do I get relief from sanctions?

Search query: CPR 3.9 relief from sanctions application unless order"""

NEW_TOOLS = json.loads("""\
[{
    "type": "function",
    "function": {
        "name": "search_sources",
        "description": "Retrieve sources from the Azure AI Search index containing CPR Parts, Practice Directions, Pre-Action Protocols, and Court Guides. Always expand abbreviations and acronyms to their full legal meaning with the relevant CPR rule number.",
        "parameters": {
            "type": "object",
            "properties": {
                "legal_concept_analysis": {
                    "type": "string",
                    "description": "REQUIRED FIRST STEP: Before writing the search query, briefly state (1) what specific legal concept/rule/provision the user is asking about, (2) the specific CPR rule number or Practice Direction that is the PRIMARY authoritative source, and (3) key terms that distinguish this source from similar-sounding documents."
                },
                "search_query": {
                    "type": "string",
                    "description": "Query string to retrieve documents from azure search. MUST be informed by your legal_concept_analysis. Include the specific CPR rule or Part number identified in your analysis. Target the PRIMARY authoritative source, not documents that merely mention the topic."
                },
                "subsection_hint": {
                    "type": "string",
                    "description": "Specific subsection identifier. Use an empty string when no precise subsection is named."
                }
            },
            "required": ["legal_concept_analysis", "search_query"]
        }
    }
}]""")

# ---------- test cases ----------
# Each test: (question, expected_terms, avoid_terms, label)
# expected_terms: at least one must appear in the query (case-insensitive)
# avoid_terms: none should appear in the query

TEST_CASES = [
    # --- Confusable concepts (the main target) ---
    ("What is PAD?",
     ["31.16", "pre-action disclosure"],
     ["protocol"],
     "PAD abbreviation"),
    ("Tell me about pre-action disclosure",
     ["31.16"],
     ["protocol"],
     "PAD explicit"),
    ("Tell me about PAD requirements",
     ["31.16", "pre-action disclosure"],
     ["protocol"],
     "PAD requirements"),
    ("What are the rules on pre-action disclosure applications?",
     ["31.16"],
     ["protocol"],
     "PAD applications"),
    ("What are Pre-Action Protocols?",
     ["pre-action protocol", "protocol"],
     [],
     "Protocols (should NOT target 31.16)"),
    ("What is the standard disclosure process?",
     ["31.6", "standard disclosure"],
     ["57ad"],
     "Standard disclosure vs extended"),
    ("Tell me about extended disclosure",
     ["57ad", "extended disclosure"],
     [],
     "Extended disclosure PD57AD"),
    ("What is the difference between striking out and summary judgment?",
     ["3.4", "24", "strike", "summary"],
     [],
     "Strike out vs SJ distinction"),

    # --- CPR Part queries ---
    ("What are the rules on summary judgment?",
     ["part 24", "24", "summary judgment"],
     [],
     "Summary judgment"),
    ("How do I get default judgment?",
     ["part 12", "12", "default judgment"],
     [],
     "Default judgment"),
    ("Tell me about relief from sanctions",
     ["3.9", "relief from sanctions"],
     [],
     "Relief from sanctions"),
    ("What are the rules on expert evidence?",
     ["part 35", "35", "expert"],
     [],
     "Expert evidence"),
    ("How do appeals work?",
     ["part 52", "52", "appeal"],
     [],
     "Appeals"),
    ("What is a freezing order?",
     ["part 25", "25", "freezing"],
     [],
     "Freezing orders"),
    ("Tell me about costs budgeting",
     ["3e", "3.15", "costs budget", "budget"],
     [],
     "Costs budgeting"),
    ("What are Norwich Pharmacal orders?",
     ["31.18", "norwich pharmacal", "third party disclosure"],
     [],
     "NPO"),

    # --- Acronyms ---
    ("What is SJ?",
     ["24", "summary judgment"],
     [],
     "SJ acronym"),
    ("What is DJ?",
     ["12", "default judgment"],
     [],
     "DJ acronym"),
    ("What is RFS?",
     ["3.9", "relief from sanctions"],
     [],
     "RFS acronym"),
    ("Tell me about CMC",
     ["29", "case management conference"],
     [],
     "CMC acronym"),

    # --- Ambiguous queries ---
    ("What are the time limits for appeals?",
     ["52", "appeal", "time"],
     [],
     "Time limits for appeals"),
    ("How do I serve documents abroad?",
     ["part 6", "6", "service", "jurisdiction"],
     [],
     "Service abroad"),
    ("What are the disclosure rules in the Business and Property Courts?",
     ["57ad", "extended disclosure", "business and property"],
     [],
     "BPC disclosure"),
    ("Tell me about unless orders",
     ["3.1", "unless order", "sanction"],
     [],
     "Unless orders"),
]


async def run_ab_test():
    credential = AzureDeveloperCliCredential(tenant_id=os.environ.get("AZURE_TENANT_ID"))
    token = credential.get_token("https://cognitiveservices.azure.com/.default")

    client = AsyncAzureOpenAI(
        azure_endpoint=ENDPOINT,
        azure_ad_token=token.token,
        api_version="2025-04-01-preview",
    )

    results = []

    for i, (question, expected, avoid, label) in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] {label}: {question}")

        row = {"label": label, "question": question, "expected": expected, "avoid": avoid}

        for variant, sys_prompt, tools, max_tok in [
            ("OLD", OLD_SYSTEM, OLD_TOOLS, 100),
            ("NEW", NEW_SYSTEM, NEW_TOOLS, 300),
        ]:
            t0 = time.time()
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": question},
                    ],
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "search_sources"}},
                    max_completion_tokens=max_tok,
                    temperature=0,
                )
                elapsed = time.time() - t0

                # Extract tool call
                tc = resp.choices[0].message.tool_calls
                if tc:
                    args = json.loads(tc[0].function.arguments)
                    query = args.get("search_query", "")
                    analysis = args.get("legal_concept_analysis", "")
                else:
                    query = ""
                    analysis = ""

                # Score
                q_lower = query.lower()
                has_expected = any(t.lower() in q_lower for t in expected)
                has_avoid = any(t.lower() in q_lower for t in avoid) if avoid else False

                row[f"{variant}_query"] = query
                row[f"{variant}_analysis"] = analysis
                row[f"{variant}_expected_hit"] = has_expected
                row[f"{variant}_avoid_hit"] = has_avoid
                row[f"{variant}_latency"] = round(elapsed, 2)

                status = "✓" if has_expected and not has_avoid else "✗"
                print(f"  {variant}: {status} query={query[:80]}")
                if has_avoid:
                    print(f"  {variant}: ⚠ AVOID term found in query!")

            except Exception as e:
                elapsed = time.time() - t0
                row[f"{variant}_query"] = f"ERROR: {e}"
                row[f"{variant}_analysis"] = ""
                row[f"{variant}_expected_hit"] = False
                row[f"{variant}_avoid_hit"] = False
                row[f"{variant}_latency"] = round(elapsed, 2)
                print(f"  {variant}: ERROR {e}")

        results.append(row)

    # ---------- summary ----------
    print("\n" + "=" * 80)
    print("A/B COMPARISON RESULTS")
    print("=" * 80)

    for variant in ["OLD", "NEW"]:
        expected_hits = sum(1 for r in results if r[f"{variant}_expected_hit"])
        avoid_hits = sum(1 for r in results if r[f"{variant}_avoid_hit"])
        correct = sum(1 for r in results if r[f"{variant}_expected_hit"] and not r[f"{variant}_avoid_hit"])
        avg_lat = sum(r[f"{variant}_latency"] for r in results) / len(results)
        print(f"\n{variant} PROMPT:")
        print(f"  Expected terms hit: {expected_hits}/{len(results)} ({100*expected_hits/len(results):.1f}%)")
        print(f"  Avoid terms hit (BAD): {avoid_hits}/{len(results)} ({100*avoid_hits/len(results):.1f}%)")
        print(f"  Fully correct: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
        print(f"  Avg latency: {avg_lat:.2f}s")

    # Show where OLD and NEW differ
    print("\n" + "-" * 80)
    print("DIFFERENCES (OLD != NEW):")
    print("-" * 80)
    diffs = 0
    for r in results:
        old_ok = r["OLD_expected_hit"] and not r["OLD_avoid_hit"]
        new_ok = r["NEW_expected_hit"] and not r["NEW_avoid_hit"]
        if old_ok != new_ok:
            diffs += 1
            marker = "NEW wins" if new_ok else "OLD wins"
            print(f"\n  [{marker}] {r['label']}: {r['question']}")
            print(f"    OLD: {r['OLD_query'][:80]}")
            print(f"    NEW: {r['NEW_query'][:80]}")
            if r.get("NEW_analysis"):
                print(f"    Analysis: {r['NEW_analysis'][:120]}")
    if diffs == 0:
        print("  No differences in correctness between OLD and NEW prompts.")

    # Save results
    outfile = os.path.join(os.path.dirname(__file__), "prompt_ab_results.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {outfile}")


if __name__ == "__main__":
    asyncio.run(run_ab_test())
