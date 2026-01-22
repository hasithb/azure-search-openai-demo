#!/usr/bin/env python3
"""Quick test to see what the v2 index returns for overriding objective."""
import os
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

# Load environment
os.chdir("/Users/HasithB/Downloads/PROJECTS/azure-search-openai-demo-2/app/backend")
load_dotenv(".env")

service_endpoint = os.environ.get("AZURE_SEARCH_SERVICE", "https://cpr-rag.search.windows.net")
index_name = os.environ.get("AZURE_SEARCH_INDEX", "legal-court-rag-index-v2")

print(f"Searching {index_name} for 'overriding objective'...")

credential = DefaultAzureCredential()
client = SearchClient(service_endpoint, index_name, credential)

# Same fields as in approach.py search
select_fields = ["id", "content", "category", "sourcepage", "sourcefile", "storageUrl", "updated", "oids", "groups"]

results = client.search(
    search_text="overriding objective CPR 1.1 part 1",
    top=5,
    select=select_fields
)

print(f"\n=== Search Results ===")
for i, doc in enumerate(results):
    print(f"\n--- Document {i+1} ---")
    print(f"id: {doc.get('id')}")
    print(f"sourcepage: {doc.get('sourcepage')}")
    print(f"sourcefile: {doc.get('sourcefile')}")
    print(f"category: {doc.get('category')}")
    print(f"storageUrl: {doc.get('storageUrl')}")
    print(f"updated: {doc.get('updated')}")
    content = doc.get("content", "")[:200]
    print(f"content (first 200): {content}")
