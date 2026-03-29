"""Quick performance test for the new GPT-5.4 family models."""
import json
import time
import urllib.request

BASE_URL = "http://localhost:50505"

def test_chat(question: str, label: str, overrides: dict | None = None):
    default_overrides = {
        "top": 3,
        "retrieval_mode": "hybrid",
        "semantic_ranker": True,
        "reasoning_effort": "low",
        "use_agentic_knowledgebase": False,
        "seed": 1,
    }
    if overrides:
        default_overrides.update(overrides)

    payload = {
        "messages": [{"role": "user", "content": question}],
        "stream": False,
        "context": {"overrides": default_overrides},
    }

    req = urllib.request.Request(
        f"{BASE_URL}/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        elapsed = time.perf_counter() - start
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"\n{'='*60}")
        print(f"[{label}] FAILED after {elapsed:.1f}s: {e}")
        return None

    msg = data.get("message", {})
    content = msg.get("content", "")
    ctx = data.get("context", {})
    thoughts = ctx.get("thoughts", [])
    data_points = ctx.get("data_points", {}).get("text", [])

    # Extract token usage
    token_info = ""
    for t in thoughts:
        desc = str(t.get("description", ""))
        if "prompt_tokens" in desc or "completion_tokens" in desc:
            token_info = desc[:300]

    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"  Question: {question[:80]}")
    print(f"  Latency:  {elapsed:.2f}s")
    print(f"  Response:  {len(content)} chars")
    print(f"  Sources:   {len(data_points)}")
    if token_info:
        print(f"  Tokens:    {token_info}")
    print(f"  Preview:   {content[:300]}...")
    return {"label": label, "latency": elapsed, "chars": len(content), "sources": len(data_points)}


def main():
    # Check server is up
    try:
        with urllib.request.urlopen(f"{BASE_URL}/config", timeout=5) as r:
            config = json.loads(r.read())
        print(f"Server running. Reasoning enabled: {config.get('showReasoningEffortOption')}")
    except Exception as e:
        print(f"Server not reachable at {BASE_URL}: {e}")
        return

    results = []

    # Test 1: Basic legal question
    r = test_chat(
        "What is the overriding objective in the CPR?",
        "Basic CPR question",
    )
    if r:
        results.append(r)

    # Test 2: Specific rule lookup
    r = test_chat(
        "What are the requirements for summary judgment under Part 24?",
        "Part 24 summary judgment",
    )
    if r:
        results.append(r)

    # Test 3: Cross-reference question
    r = test_chat(
        "What is the time limit for filing an appeal and what permission is needed?",
        "Appeals time limit",
    )
    if r:
        results.append(r)

    # Test 4: With higher reasoning effort
    r = test_chat(
        "Compare the disclosure requirements in the Commercial Court Guide with standard CPR disclosure rules.",
        "Complex comparison (medium effort)",
        overrides={"reasoning_effort": "medium"},
    )
    if r:
        results.append(r)

    # Summary
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        avg_latency = sum(r["latency"] for r in results) / len(results)
        avg_chars = sum(r["chars"] for r in results) / len(results)
        print(f"  Tests run:      {len(results)}")
        print(f"  Avg latency:    {avg_latency:.2f}s")
        print(f"  Avg response:   {avg_chars:.0f} chars")
        for r in results:
            status = "OK" if r["chars"] > 50 and r["sources"] > 0 else "WARN"
            print(f"  [{status}] {r['label']:40s}  {r['latency']:5.2f}s  {r['chars']:5d} chars  {r['sources']} sources")


if __name__ == "__main__":
    main()
