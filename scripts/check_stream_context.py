#!/usr/bin/env python3
import json
import requests

payload = {
    "messages": [{"content": "What is disclosure under the CPR?", "role": "user"}],
    "context": {
        "overrides": {
            "top": 5,
            "retrieval_mode": "hybrid",
            "semantic_ranker": True,
            "query_rewriting": True,
            "use_agentic_retrieval": True
        }
    }
}

resp = requests.post("http://localhost:50505/chat/stream", json=payload, stream=True)
resp.raise_for_status()

last_ctx = {}
for line in resp.iter_lines(decode_unicode=True):
    if not line:
        continue
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "context" in event:
        last_ctx.update(event.get("context") or {})

print("Final context keys:", list(last_ctx.keys()))
print("citation_map:", last_ctx.get("citation_map"))
print("enhanced_citations:", last_ctx.get("enhanced_citations"))
data_points = last_ctx.get("data_points") or {}
text_points = data_points.get("text") or []
print("data_points.text len:", len(text_points))
if text_points:
    print("first data_point:", text_points[0])
