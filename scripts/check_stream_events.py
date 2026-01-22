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

print("Events:")
for i, line in enumerate(resp.iter_lines(decode_unicode=True)):
    if not line:
        continue
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    has_ctx = "context" in event
    has_dp = bool(event.get("context", {}).get("data_points")) if has_ctx else False
    has_cm = bool(event.get("context", {}).get("citation_map")) if has_ctx else False
    has_ec = bool(event.get("context", {}).get("enhanced_citations")) if has_ctx else False
    has_delta = bool(event.get("delta", {}).get("content"))
    print(f"{i:03d}: ctx={has_ctx} dp={has_dp} cm={has_cm} ec={has_ec} delta={has_delta}")
