#!/usr/bin/env python3
"""Quick diagnosis script for prompt issues."""
import httpx
import json
import re

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
    return httpx.post(f"{BASE}/chat", json=payload, timeout=60).json()


# 1. Check follow-up questions
print("=" * 60)
print("DIAGNOSIS 1: Follow-up questions")
print("=" * 60)
result = chat("What is CPR Part 1?")
answer = result["message"]["content"]
followups = re.findall(r"<<(.+?)>>", answer)
print(f"Follow-ups found: {len(followups)}")
print(f"Answer ends with: ...{answer[-200:]}")

# Check if the prompt contains the follow-up instruction
thoughts = result.get("context", {}).get("thoughts", [])
for t in thoughts:
    desc = str(t.get("description", ""))
    if "Generate 3" in desc:
        print("\nPrompt DOES contain follow-up instruction")
        break
    if "angle brackets" in desc:
        print("\nPrompt DOES contain follow-up instruction (angle brackets)")
        break
else:
    # Look everywhere in thought props too
    all_text = json.dumps(thoughts)
    if "Generate 3" in all_text:
        print("\nPrompt contains follow-up instruction (found in thoughts)")
    else:
        print("\nPrompt does NOT contain follow-up instruction")


# 2. Check search depth in prompt
print("\n" + "=" * 60)
print("DIAGNOSIS 2: Search depth in prompt")
print("=" * 60)
result2 = chat("Compare disclosure rules across courts", retrieval_reasoning_effort="minimal")
thoughts2 = result2.get("context", {}).get("thoughts", [])
all_text2 = json.dumps(thoughts2)
print(f"'Quick' in prompt: {'Quick' in all_text2}")
print(f"'currently using' in prompt: {'currently using' in all_text2}")
print(f"'search depth' in prompt: {'search depth' in all_text2.lower()}")
# Find the exact search depth mention
if "currently using" in all_text2:
    idx = all_text2.index("currently using")
    print(f"Context: ...{all_text2[idx:idx+80]}...")
answer2 = result2["message"]["content"]
print(f"\nAnswer mentions 'search depth': {'search depth' in answer2.lower()}")
print(f"Answer mentions 'Thorough': {'thorough' in answer2.lower()}")
print(f"Answer tail: ...{answer2[-400:]}")


# 3. Check disambiguation for "disclosure"
print("\n" + "=" * 60)
print("DIAGNOSIS 3: Disclosure disambiguation")
print("=" * 60)
result3 = chat("Tell me about disclosure")
answer3 = result3["message"]["content"]
print(f"'standard' in answer: {'standard' in answer3.lower()}")
print(f"'extended' in answer: {'extended' in answer3.lower()}")
print(f"'PD 57AD' in answer: {'57AD' in answer3}")
print(f"'CPR 31' in answer: {'CPR 31' in answer3}")
print(f"Full answer length: {len(answer3)}")
# Look at sources to see if both types are in retrieved docs
sources3 = result3.get("context", {}).get("data_points", {}).get("text", [])
source_cats = set()
for s in sources3:
    if isinstance(s, dict):
        c = s.get("category", "")
        sp = s.get("sourcepage", "")
        source_cats.add(f"{c}: {sp}")
print(f"\nSource categories/pages:")
for sc in sorted(source_cats):
    print(f"  - {sc}")


# 4. Check source mismatch when filtered
print("\n" + "=" * 60)
print("DIAGNOSIS 4: Source mismatch detection (filtered)")
print("=" * 60)
result4 = chat(
    "What are the specific procedures for starting a claim in the Commercial Court?",
    include_category="Civil Procedure Rules and Practice Directions"
)
answer4 = result4["message"]["content"]
print(f"'Commercial Court Guide' mentioned: {'commercial court guide' in answer4.lower()}")
print(f"'recommend' mentioned: {'recommend' in answer4.lower()}")
print(f"'suggest' mentioned: {'suggest' in answer4.lower()}")
print(f"'also check' mentioned: {'also check' in answer4.lower()}")
# Actually, the user asked about Commercial Court, and it IS a CPR Part 58 topic
# Let's check what the answer says
print(f"\nAnswer preview: {answer4[:500]}")
