#!/usr/bin/env python3
"""Diagnosis 3: Test whether the LLM recommends Thorough for complex queries,
and investigate follow-up question generation."""
import httpx
import json
import re

BASE = "http://localhost:50505"

def chat(q, **overrides):
    payload = {
        "messages": [{"content": q, "role": "user"}],
        "context": {"overrides": {
            "use_agentic_knowledgebase": True,
            "retrieval_reasoning_effort": "minimal",
            "suggest_followup_questions": True,
            **overrides,
        }}
    }
    return httpx.post(f"{BASE}/chat", json=payload, timeout=120).json()


# Test: A question where the answer is genuinely thin at Quick depth
print("=" * 60)
print("TEST A: Thin results at Quick depth")
print("=" * 60)
result = chat("What are all the different types of interim remedies available under CPR Part 25 and how do they interact with the Commercial Court Guide's provisions on freezing orders?")
answer = result["message"]["content"]
sources = result.get("context", {}).get("data_points", {}).get("text", [])
print(f"Sources: {len(sources)}")
print(f"Answer length: {len(answer)}")
print(f"Mentions 'search depth': {'search depth' in answer.lower()}")
print(f"Mentions 'Thorough': {'thorough' in answer.lower()}")
print(f"Mentions 'comprehensive': {'comprehensive' in answer.lower()}")
print(f"\nAnswer tail:\n{answer[-500:]}")

# Test: Question where available info is clearly insufficient
print("\n" + "=" * 60)
print("TEST B: Clearly insufficient info at Quick depth")
print("=" * 60)
result2 = chat("Compare the approaches to case management conferences across the Commercial Court Guide, TCC Guide, and Patents Court Guide. Include specific paragraph references.")
answer2 = result2["message"]["content"]
sources2 = result2.get("context", {}).get("data_points", {}).get("text", [])
print(f"Sources: {len(sources2)}")
src_cats = set()
for s in sources2:
    if isinstance(s, dict):
        src_cats.add(s.get("category", "?"))
print(f"Source categories: {sorted(src_cats)}")
print(f"Answer length: {len(answer2)}")
print(f"Mentions 'search depth': {'search depth' in answer2.lower()}")
print(f"Mentions 'Thorough': {'thorough' in answer2.lower()}")
print(f"\nAnswer tail:\n{answer2[-500:]}")
