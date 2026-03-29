#!/usr/bin/env python3
"""
Create a v3 knowledge source and update the knowledge base to use it.
This fixes the 'Knowledge Source Params target Knowledge Source name must match' error.
"""
import os
import sys
import json
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "backend"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from azure.identity import AzureDeveloperCliCredential

cred = AzureDeveloperCliCredential(process_timeout=60)
endpoint = f"https://{os.environ['AZURE_SEARCH_SERVICE']}.search.windows.net"
api_ver = "2025-11-01-preview"
kb_name = os.getenv("AZURE_SEARCH_AGENT") or os.getenv("AZURE_SEARCH_KNOWLEDGEBASE_NAME", "")
new_index = os.environ["AZURE_SEARCH_INDEX"]  # legal-court-rag-index-v3

token = cred.get_token("https://search.azure.com/.default")
headers = {
    "Authorization": f"Bearer {token.token}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

print(f"Endpoint: {endpoint}")
print(f"Knowledge base: {kb_name}")
print(f"New index: {new_index}")

# Step 1: Create knowledge source for v3 index
print(f"\n=== Step 1: Create knowledge source '{new_index}' ===")
ks_body = {
    "name": new_index,
    "kind": "searchIndex",
    "description": f"Knowledge source for {new_index} (v3 index with court guides)",
    "searchIndexParameters": {
        "searchIndexName": new_index,
        "semanticConfigurationName": None,
        "sourceDataFields": [
            {"name": "id"},
            {"name": "sourcepage"},
            {"name": "sourcefile"},
            {"name": "content"},
            {"name": "category"},
        ],
        "searchFields": [],
    },
}

resp = requests.put(
    f"{endpoint}/knowledgesources/{new_index}?api-version={api_ver}",
    headers=headers,
    json=ks_body,
)
print(f"PUT status: {resp.status_code}")
if resp.ok:
    print(f"✅ Knowledge source '{new_index}' created/updated")
    print(json.dumps(resp.json(), indent=2))
else:
    print(f"❌ Error: {resp.text}")
    sys.exit(1)

# Step 2: Update knowledge base to use v3 knowledge source
print(f"\n=== Step 2: Update knowledge base '{kb_name}' ===")
resp2 = requests.get(f"{endpoint}/knowledgebases/{kb_name}?api-version={api_ver}", headers=headers)
assert resp2.ok, f"GET KB failed: {resp2.text}"
kb_data = resp2.json()

old_sources = [s["name"] for s in kb_data.get("knowledgeSources", [])]
print(f"Old knowledge sources: {old_sources}")

# Update
kb_data["knowledgeSources"] = [{"name": new_index}]
for key in list(kb_data.keys()):
    if key.startswith("@odata"):
        del kb_data[key]

resp3 = requests.put(
    f"{endpoint}/knowledgebases/{kb_name}?api-version={api_ver}",
    headers=headers,
    json=kb_data,
)
print(f"PUT status: {resp3.status_code}")
if resp3.ok:
    result = resp3.json()
    new_sources = [s["name"] for s in result.get("knowledgeSources", [])]
    print(f"✅ Knowledge base updated: {old_sources} -> {new_sources}")
    print(json.dumps(result, indent=2))
else:
    print(f"❌ Error: {resp3.text}")
    sys.exit(1)

# Step 3: Verify
print(f"\n=== Step 3: Verify ===")
resp4 = requests.get(f"{endpoint}/knowledgebases/{kb_name}?api-version={api_ver}", headers=headers)
if resp4.ok:
    verified = resp4.json()
    sources = [s["name"] for s in verified.get("knowledgeSources", [])]
    print(f"Verified knowledge sources: {sources}")
    if new_index in sources:
        print(f"✅ All good! Knowledge base now uses '{new_index}'")
    else:
        print(f"❌ Verification failed - '{new_index}' not in sources")
else:
    print(f"Verification GET failed: {resp4.text}")
