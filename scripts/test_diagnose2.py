#!/usr/bin/env python3
"""Quick diagnosis - search depth in prompt and follow-up questions."""
import httpx
import json

BASE = "http://localhost:50505"

def chat(q, **overrides):
    payload = {
        "messages": [{"content": q, "role": "user"}],
        "context": {"overrides": {
            "use_agentic_knowledgebase": True,
            "retrieval_reasoning_effort": "low",
            "suggest_followup_questions": True,
            **overrides,
        }}
    }
    return httpx.post(f"{BASE}/chat", json=payload, timeout=120).json()


# 1. Simple question at Quick depth — check if search_depth appears in prompt
print("=" * 60)
print("TEST: Simple question at Quick depth — check prompt")
print("=" * 60)
result = chat("What is CPR Part 31?", retrieval_reasoning_effort="minimal")
thoughts = result.get("context", {}).get("thoughts", [])
all_text = json.dumps(thoughts)

print(f"'Quick' in thoughts: {'Quick' in all_text}")
print(f"'currently using' in thoughts: {'currently using' in all_text}")
print(f"'Thorough' in thoughts: {'Thorough' in all_text}")

if "currently using" in all_text:
    idx = all_text.index("currently using")
    print(f"  Context around 'currently using': ...{all_text[max(0,idx-20):idx+100]}...")

answer = result["message"]["content"]
print(f"\nAnswer preview: {answer[:300]}")
print(f"\ndepth mention in answer: {'search depth' in answer.lower() or 'thorough' in answer.lower()}")

# 2. Check follow-up questions with a simpler query
print("\n" + "=" * 60)
print("TEST: Follow-up questions generation")
print("=" * 60)
result2 = chat("What is standard disclosure?")
answer2 = result2["message"]["content"]
print(f"Answer length: {len(answer2)}")
print(f"Contains '<<': {'<<' in answer2}")
if "<<" in answer2:
    import re
    followups = re.findall(r"<<(.+?)>>", answer2)
    print(f"Follow-ups: {followups}")
print(f"Answer tail: ...{answer2[-400:]}")
