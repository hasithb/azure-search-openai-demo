import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Probe:
    label: str
    category: str
    question: str


DEFAULT_PROBES = [
    Probe(
        label="CPR deadlines",
        category="Civil Procedure Rules and Practice Directions",
        question="When must particulars of claim be served if they are not served with the claim form?",
    ),
    Probe(
        label="Commercial Court CMC",
        category="Commercial Court",
        question="What does the Commercial Court Guide say about case management conferences?",
    ),
    Probe(
        label="Chancery ADR",
        category="Chancery Division",
        question="What guidance does the Chancery Guide give about Part 8 claims and alternative dispute resolution?",
    ),
    Probe(
        label="KBD triage",
        category="King's Bench Division",
        question="How are cases triaged in the King's Bench Division Guide?",
    ),
    Probe(
        label="Patents urgent",
        category="Patents Court",
        question="What does the Patents Court Guide say about urgent applications?",
    ),
    Probe(
        label="TCC statements",
        category="Technology and Construction Court",
        question="What does the Technology and Construction Court Guide say about statements of case?",
    ),
    Probe(
        label="Cross-category filing",
        category="",
        question="What are the deadlines for filing in the Circuit Commercial Court?",
    ),
]


def post_chat(base_url: str, probe: Probe) -> dict[str, Any]:
    payload = {
        "messages": [{"content": probe.question, "role": "user"}],
        "context": {
            "overrides": {
                "retrieval_mode": "text",
                "send_text_sources": True,
                "send_image_sources": False,
                "search_text_embeddings": True,
                "search_image_embeddings": False,
                "include_category": probe.category,
            }
        },
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def summarize_response(probe: Probe, response: dict[str, Any]) -> dict[str, Any]:
    text_sources = response.get("context", {}).get("data_points", {}).get("text", []) or []
    citations = [item.get("citation") for item in text_sources[:5] if isinstance(item, dict)]
    answer = response.get("message", {}).get("content", "")
    return {
        "label": probe.label,
        "category": probe.category or "All Sources",
        "question": probe.question,
        "answer": answer,
        "text_count": len(text_sources),
        "citations": citations,
        "idk": answer.strip().lower().startswith("i don't know") or answer.strip().lower().startswith("i don’t know"),
    }


def print_text_report(results: list[dict[str, Any]]) -> None:
    idk_count = sum(1 for result in results if result["idk"])
    mean_sources = round(sum(result["text_count"] for result in results) / len(results), 2) if results else 0
    print(f"Probes: {len(results)}")
    print(f"I-don't-know rate: {idk_count}/{len(results)}")
    print(f"Average text source count: {mean_sources}")
    print()
    for result in results:
        print(f"== {result['label']} ==")
        print(f"Category: {result['category']}")
        print(f"Question: {result['question']}")
        print(f"Text sources: {result['text_count']}")
        print(f"Top citations: {result['citations']}")
        print(f"Answer: {result['answer']}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live retrieval probes against the local chat backend.")
    parser.add_argument("--base-url", default="http://127.0.0.1:50505", help="Base URL for the local backend")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text report")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    for probe in DEFAULT_PROBES:
        try:
            response = post_chat(args.base_url, probe)
        except urllib.error.URLError as exc:
            print(f"Probe failed for {probe.label}: {exc}", file=sys.stderr)
            return 1
        results.append(summarize_response(probe, response))

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print_text_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())