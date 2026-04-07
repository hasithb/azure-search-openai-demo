import json
import sys
import time

import requests

question = sys.argv[1]

payload = {
    "messages": [{"role": "user", "content": question}],
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

targets = {
    "local": "http://localhost:50505/chat",
    "deployed": "https://capps-backend-ot6tupm5qi5wy.delightfulground-1a2f1220.eastus2.azurecontainerapps.io/chat",
}

for name, url in targets.items():
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=45)
        elapsed = round(time.time() - start, 2)
        data = response.json()
        print(f"=== {name} ({elapsed}s, status={response.status_code}) ===")
        for src in data.get("context", {}).get("data_points", {}).get("text", [])[:5]:
            if isinstance(src, dict):
                print(f"  - {src.get('sourcefile', '?')} | {src.get('subsection_id', '?')} | {src.get('category', '?')}")
        answer = data.get("message", {}).get("content", "")
        print(answer[:500].replace("\n", " "))
        print()
    except Exception as exc:
        elapsed = round(time.time() - start, 2)
        print(f"=== {name} ({elapsed}s) ERROR ===")
        print(str(exc))
        print()
