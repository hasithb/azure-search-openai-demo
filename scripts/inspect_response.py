"""Quick inspection of a single chat response to understand citation format."""
import asyncio
import json
import re
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:50505/chat",
            json={
                "messages": [{"role": "user", "content": "What is CPR Part 1 about?"}],
                "context": {"overrides": {"retrieval_mode": "hybrid", "semantic_ranker": True, "top": 3, "query_rewriting": True}},
                "stream": False,
            },
            timeout=60.0,
        )
        data = resp.json()
        msg = data.get("message", {}).get("content", "")

        print("=== ANSWER (first 800 chars) ===")
        print(msg[:800])
        print()

        # Try different citation patterns
        print("=== CITATION PATTERNS ===")
        patterns = {
            "bracket_num": r"\[(\d+)\]",
            "bracket_file": r"\[[^\]]+\.\w+[^\]]*\]",
            "paren_cite": r"\(([^)]+\.(?:pdf|html|txt|md)[^)]*)\)",
            "any_bracket": r"\[([^\]]+)\]",
        }
        for name, pat in patterns.items():
            matches = re.findall(pat, msg)
            if matches:
                print(f"  {name}: {matches[:10]}")

        print()
        print("=== CONTEXT STRUCTURE ===")
        ctx = data.get("context", {})
        print(f"  Keys: {list(ctx.keys())}")
        dp = ctx.get("data_points", {})
        print(f"  data_points keys: {list(dp.keys())}")
        text_dp = dp.get("text", [])
        print(f"  text data_points count: {len(text_dp)}")
        if text_dp:
            print(f"  First data_point (200 chars): {str(text_dp[0])[:200]}")

        # Save full response for inspection
        with open("scripts/retrieval_depth_test_results/sample_response.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
        print("\nFull response saved to sample_response.json")

asyncio.run(main())
