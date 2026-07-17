"""Local agentic retrieval test against the newly created knowledge base."""

import asyncio
import time

from azure.identity.aio import AzureDeveloperCliCredential
from azure.search.documents.knowledgebases.aio import KnowledgeBaseRetrievalClient
from azure.search.documents.knowledgebases.models import (
    KnowledgeBaseMessage,
    KnowledgeBaseMessageTextContent,
    KnowledgeBaseRetrievalRequest,
    KnowledgeRetrievalLowReasoningEffort,
    SearchIndexKnowledgeSourceParams,
)

ENDPOINT = "https://gptkb-gz2m4s637t5me.search.windows.net"
KB_NAME = "legal-court-rag-kb"
TENANT_ID = "3bfe16b2-5fcc-4565-b1f1-15271d20fecf"
KNOWLEDGE_SOURCE_NAME = "legal-court-rag-index-v3"  # Must match the KB source name (= index name)


async def test_query(client, query: str):
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print("=" * 80)

    request = KnowledgeBaseRetrievalRequest(
        messages=[
            KnowledgeBaseMessage(
                role="user",
                content=[KnowledgeBaseMessageTextContent(text=query)],
            )
        ],
        knowledge_source_params=[
            SearchIndexKnowledgeSourceParams(
                knowledge_source_name=KNOWLEDGE_SOURCE_NAME,
                include_references=True,
                include_reference_source_data=True,
                reranker_threshold=1.5,
            )
        ],
        include_activity=True,
        retrieval_reasoning_effort=KnowledgeRetrievalLowReasoningEffort(),
        output_mode="extractiveData",
    )

    t0 = time.perf_counter()
    response = await client.retrieve(retrieval_request=request)
    elapsed = time.perf_counter() - t0

    print(f"Elapsed: {elapsed:.2f}s")

    # Answer / response messages
    if response.response:
        for msg in response.response:
            print(f"\nResponse role={msg.role}")
            if msg.content:
                for c in msg.content:
                    text = getattr(c, "text", None)
                    if text:
                        print(f"  text: {text[:600]}")
    else:
        print("No response messages")

    # Activity
    activities = response.activity or []
    print(f"\nActivities: {len(activities)}")
    for a in activities:
        status = getattr(a, "status", "n/a")
        print(f"  id={a.id} type={a.type} status={status}")
        if hasattr(a, "search_index_arguments") and a.search_index_arguments:
            print(f"    search: {a.search_index_arguments.search}")
            if a.search_index_arguments.filter:
                print(f"    filter: {a.search_index_arguments.filter}")

    # References
    refs = response.references or []
    print(f"\nReferences: {len(refs)}")
    for r in refs[:5]:
        print(f"  id={r.id} type={type(r).__name__}")
        if hasattr(r, "doc_key"):
            print(f"    doc_key: {r.doc_key}")
        if hasattr(r, "source_data") and r.source_data:
            sd = r.source_data
            if isinstance(sd, dict):
                sp = sd.get("sourcepage", "")
                ct = sd.get("content", "")
            else:
                sp = getattr(sd, "sourcepage", "")
                ct = getattr(sd, "content", "")
            if sp:
                print(f"    sourcepage: {sp}")
            if ct:
                print(f"    content preview: {str(ct)[:200]}")

    return elapsed


async def main():
    cred = AzureDeveloperCliCredential(tenant_id=TENANT_ID)
    client = KnowledgeBaseRetrievalClient(
        endpoint=ENDPOINT, knowledge_base_name=KB_NAME, credential=cred
    )

    queries = [
        "What is the test for relief from sanctions under CPR 3.9?",
        "What are the requirements for summary judgment under Part 24?",
        "How does the Commercial Court Guide address disclosure?",
    ]

    try:
        timings = []
        for q in queries:
            elapsed = await test_query(client, q)
            timings.append((q, elapsed))

        print(f"\n{'='*80}")
        print("TIMING SUMMARY")
        print("=" * 80)
        for q, t in timings:
            print(f"  {t:.2f}s  {q[:60]}")
        avg = sum(t for _, t in timings) / len(timings)
        print(f"\n  Average: {avg:.2f}s")
        print(f"  (Compare to standard RAG: ~13.9s total, ~1.05s search-only)")
    finally:
        await client.close()
        await cred.close()


if __name__ == "__main__":
    asyncio.run(main())
