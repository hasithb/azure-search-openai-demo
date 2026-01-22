#!/usr/bin/env python3
"""Search v2 index for overriding objective content."""
import os
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

# Load environment
load_dotenv("app/backend/.env")

service_endpoint = os.environ.get("AZURE_SEARCH_SERVICE", "https://cpr-rag.search.windows.net")
index_name = os.environ.get("AZURE_SEARCH_INDEX", "legal-court-rag-index-v2")

print(f"Searching {service_endpoint}/{index_name} for 'overriding objective'...")

credential = DefaultAzureCredential()
client = SearchClient(service_endpoint, index_name, credential)

results = client.search(
    search_text="overriding objective",
    top=5,
    select=["sourcepage", "sourcefile", "category", "content"]
)

print(f"\n=== Search Results ===")
for i, doc in enumerate(results):
    print(f"\n{i+1}. {doc.get('sourcepage')} | {doc.get('sourcefile')} | {doc.get('category')}")
    content = doc.get("content", "")[:500]
    print(f"   Content: {content}...")
