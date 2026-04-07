import asyncio
import json
import pathlib
import time

import httpx

TARGETS = {
    "local": "http://localhost:50505/chat",
    "deployed": "https://capps-backend-ot6tupm5qi5wy.delightfulground-1a2f1220.eastus2.azurecontainerapps.io/chat",
}

CASES = [
    {
        "label": "CPR pre-action disclosure",
        "question": "What are the requirements for pre action disclosure?",
        "anchors": ["31.16", "supported by evidence"],
    },
    {
        "label": "CPR relief from sanctions",
        "question": "What is the test for relief from sanctions under CPR 3.9?",
        "anchors": ["3.9", "all the circumstances"],
    },
    {
        "label": "PD appeals",
        "question": "What does Practice Direction 52B provide regarding appeals in the County Court and High Court?",
        "anchors": ["52B", "appeals"],
    },
    {
        "label": "PD costs management",
        "question": "What is Practice Direction 3D about costs management?",
        "anchors": ["3D", "costs management"],
    },
    {
        "label": "Commercial Court guide",
        "question": "What does the Commercial Court Guide say about case management conferences?",
        "anchors": ["Commercial Court", "case management conference"],
    },
    {
        "label": "Patents Court guide",
        "question": "What does the Patents Court Guide say about urgent applications?",
        "anchors": ["Patents Court", "urgent applications"],
    },
    {
        "label": "Chancery guide",
        "question": "What guidance does the Chancery Guide give about Part 8 claims and alternative dispute resolution?",
        "anchors": ["Chancery", "Part 8", "alternative dispute resolution"],
    },
]

PAYLOAD_TEMPLATE = {
    "context": {
        "overrides": {
            "top": 5,
            "use_agentic_knowledgebase": True,
            "retrieval_reasoning_effort": "low",
            "send_text_sources": True,
            "send_image_sources": False,
        }
    },
    "stream": False,
    "session_state": None,
}


async def fetch(client: httpx.AsyncClient, url: str, question: str):
    payload = dict(PAYLOAD_TEMPLATE)
    payload["messages"] = [{"role": "user", "content": question}]
    start = time.time()
    response = await client.post(url, json=payload)
    response.raise_for_status()
    elapsed = round(time.time() - start, 2)
    return elapsed, response.json()


async def main():
    results = []
    async with httpx.AsyncClient(timeout=120) as client:
        for case in CASES:
            row = {"label": case["label"], "question": case["question"]}
            fetched = await asyncio.gather(*[fetch(client, url, case["question"]) for url in TARGETS.values()])
            for (name, _url), (elapsed, data) in zip(TARGETS.items(), fetched):
                answer = data.get("message", {}).get("content", "") or ""
                sources = data.get("context", {}).get("data_points", {}).get("text", []) or []
                top_sources = []
                top_categories = []
                for src in sources[:5]:
                    if isinstance(src, dict):
                        top_sources.append(src.get("sourcefile", "?"))
                        top_categories.append(src.get("category", "?"))
                normalized_answer = answer.lower()
                anchor_hits = sum(1 for anchor in case["anchors"] if anchor.lower() in normalized_answer)
                row[name] = {
                    "latency": elapsed,
                    "anchor_hits": anchor_hits,
                    "top_sources": top_sources,
                    "top_categories": top_categories,
                    "answer_preview": answer[:320].replace("\n", " "),
                }
            results.append(row)

    output_path = pathlib.Path("/tmp/local_vs_deployed_quality.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    asyncio.run(main())
