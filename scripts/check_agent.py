#!/usr/bin/env python
"""Check agent configuration for knowledge source settings."""

import os
import sys
import json
import requests
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "backend"))

load_dotenv()

search_endpoint = "https://cpr-rag.search.windows.net"
agent_name = os.getenv("AZURE_SEARCH_AGENT", "legal-court-rag-index-agent")

print(f"Checking agent: {agent_name}")
print(f"Endpoint: {search_endpoint}")

# Use admin key if available
search_key = os.getenv("AZURE_SEARCH_KEY")

if search_key:
    print("Using admin key")
    headers = {"api-key": search_key, "Content-Type": "application/json"}
else:
    print("No AZURE_SEARCH_KEY found, trying Azure AD")
    from azure.identity import DefaultAzureCredential
    credential = DefaultAzureCredential()
    token = credential.get_token("https://search.azure.com/.default")
    headers = {"Authorization": f"Bearer {token.token}", "Content-Type": "application/json"}

# Get agent config
resp = requests.get(
    f"{search_endpoint}/agents/{agent_name}?api-version=2025-01-01-preview",
    headers=headers
)

print(f"\nStatus: {resp.status_code}")

if resp.status_code == 200:
    data = resp.json()
    print(json.dumps(data, indent=2))
    
    # Check knowledge sources
    if "knowledgeSources" in data:
        print("\n--- Knowledge Sources ---")
        for ks in data["knowledgeSources"]:
            print(f"  Name: {ks.get('name')}")
            if "indexSource" in ks:
                print(f"  Index: {ks['indexSource'].get('indexName')}")
else:
    print(f"Error: {resp.text}")
