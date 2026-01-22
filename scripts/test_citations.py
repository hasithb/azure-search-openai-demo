#!/usr/bin/env python3
"""Test local chat endpoint and check for citations."""
import json
import requests

url = "http://localhost:50505/chat"
payload = {
    "messages": [{"content": "What is the overriding objective?", "role": "user"}],
    "context": {
        "overrides": {
            "top": 5,
            "retrieval_mode": "hybrid",
            "semantic_ranker": True,
            "use_agentic_retrieval": True
        }
    }
}

print("Testing chat endpoint...")
try:
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    
    print("\n=== MESSAGE ===")
    msg = data.get("message", {})
    content = msg.get("content", "")
    print(f"Content length: {len(content)}")
    print(f"Content (first 1500 chars):\n{content[:1500]}")
    
    print("\n=== CONTEXT KEYS ===")
    ctx = data.get("context", {})
    if isinstance(ctx, dict):
        print(list(ctx.keys()))
    else:
        print(f"context type: {type(ctx)}")
    
    print("\n=== DATA_POINTS (first 3) ===")
    dp = ctx.get("data_points", {}) if isinstance(ctx, dict) else {}
    print(f"data_points type: {type(dp)}")
    if isinstance(dp, dict):
        print(f"data_points keys: {list(dp.keys())}")
        texts = dp.get("text", [])
        print(f"text type: {type(texts)}")
        print(f"text count: {len(texts)}")
        for i, t in enumerate(texts[:3]):
            if isinstance(t, dict):
                print(f"\n  --- Text {i+1} ---")
                print(f"  sourcepage: {t.get('sourcepage')}")
                print(f"  sourcefile: {t.get('sourcefile')}")
                print(f"  category: {t.get('category')}")
                print(f"  citation: {t.get('citation')}")
                print(f"  storageUrl: {t.get('storageUrl', 'N/A')}")
                content = str(t.get('content', ''))[:200]
                print(f"  content (first 200): {content}...")
    
    print("\n=== CITATION_MAP (all) ===")
    cm = ctx.get("citation_map", {}) if isinstance(ctx, dict) else {}
    if isinstance(cm, dict):
        print(f"citation_map count: {len(cm)}")
        for key, value in list(cm.items()):
            print(f"  {key} -> {value}")
    
    print("\n=== ENHANCED_CITATIONS ===")
    ec = ctx.get("enhanced_citations", []) if isinstance(ctx, dict) else []
    print(f"enhanced_citations count: {len(ec)}")
    if ec:
        print(f"First: {json.dumps(ec[0], indent=2)[:400]}")
    
    print("\n=== THOUGHTS - Query Generation Details ===")
    thoughts = ctx.get("thoughts", []) if isinstance(ctx, dict) else []
    
    for i, t in enumerate(thoughts):
        if isinstance(t, dict):
            title = t.get("title", "")
            print(f"\n--- Thought {i}: {title} ---")
            if "search query" in title.lower() or "search using" in title.lower():
                desc = str(t.get("description", ""))
                print(f"Description: {desc[:1000]}")
            elif "results" in title.lower():
                desc = str(t.get("description", ""))
                print(f"Results (first 500): {desc[:500]}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
