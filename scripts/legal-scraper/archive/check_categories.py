#!/usr/bin/env python
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
import os
import sys

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from config import Config
except ImportError:
    class Config:
        AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")

def get_search_client():
    service_name = Config.AZURE_SEARCH_SERVICE
    if not service_name:
         import subprocess
         try:
            res = subprocess.run("azd env get-values", shell=True, capture_output=True, text=True)
            for line in res.stdout.splitlines():
                if "AZURE_SEARCH_SERVICE" in line:
                    service_name = line.split("=")[1].strip('"')
         except:
             pass
    
    endpoint = f"https://{service_name}.search.windows.net"
    credential = DefaultAzureCredential()
    return SearchClient(endpoint=endpoint, index_name="legal-court-rag-index", credential=credential)

client = get_search_client()
results = client.search(search_text="*", facets=["category"])
print("Categories in V1 Index:")
if results.get_facets():
    for cat in results.get_facets().get("category", []):
        print(f" - {cat['value']} (count: {cat['count']})")
