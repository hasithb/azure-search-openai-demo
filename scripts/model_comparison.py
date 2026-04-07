"""Full pipeline comparison: query_rewrite + answer_gen latency per model."""
import time, json, sys
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

endpoint = "https://cog-gz2m4s637t5me-us2.openai.azure.com/"
tp = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
client = AzureOpenAI(azure_endpoint=endpoint, azure_ad_token_provider=tp, api_version="2024-12-01-preview")

models = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"]

rewrite_sys = """You have access to an Azure AI Search index with CPR Parts, Practice Directions, Pre-Action Protocols, and Court Guides.
Generate a search query. Use search_sources function.
"pre-action disclosure" = CPR 31.16. "summary judgment" = CPR Part 24."""

tools = [{"type": "function", "function": {
    "name": "search_sources", "parameters": {"type": "object", "properties": {
        "legal_concept_analysis": {"type": "string"},
        "search_query": {"type": "string"},
    }, "required": ["legal_concept_analysis", "search_query"]}
}}]

answer_sys = """Answer using ONLY sources below. Cite every fact as [number]. If insufficient, say so."""

sources_pad = """[1] (Chancery Guide, p.60): CPR 31.16 two-stage test: (a) four-part jurisdictional test, (b) court discretion.
[2] (Chancery Guide, p.60): CPR 31.16 requires standard disclosure scope under CPR 31.6.
[3] (KBD Guide, p.41): CPR31.16 provisions for disclosure before proceedings started."""

sources_sj = """[1] (CPR Part 24): 24.2 Summary judgment if no real prospect of succeeding/defending.
[2] (PD 24): Procedure and evidence requirements."""

sources_dental = """[1] (Northwind Health Plus): Dental plan covers routine and emergency care.
[2] (Northwind Standard): Basic dental coverage, two cleanings per year."""

QUERIES = [
    ("What are the requirements for pre-action disclosure", sources_pad),
    ("What are the rules on summary judgment", sources_sj),
    ("What is the dental plan?", sources_dental),
]

print(f"{'Model':<18} {'Query':<45} {'Rewrite':>8} {'Answer':>8} {'Total':>8}")
print("-" * 95)
sys.stdout.flush()

for model in models:
    for query, ctx in QUERIES:
        rw_start = time.time()
        rw = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": rewrite_sys},
                      {"role": "user", "content": f"Generate search query for: {query}"}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "search_sources"}},
            temperature=0.0, max_completion_tokens=200,
        )
        rw_time = time.time() - rw_start

        ans_start = time.time()
        ans = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": answer_sys},
                      {"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {query}"}],
            temperature=0.0, max_completion_tokens=500,
        )
        ans_time = time.time() - ans_start

        total = rw_time + ans_time
        print(f"{model:<18} {query[:43]:<45} {rw_time:>7.2f}s {ans_time:>7.2f}s {total:>7.2f}s")
        sys.stdout.flush()
    print()
    sys.stdout.flush()

print("\nNote: Search/retrieval adds ~2-3s constant overhead (same for all models)")
