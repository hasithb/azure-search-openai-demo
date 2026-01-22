from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
import json

# Manual config import
import sys
sys.path.insert(0, 'scripts/legal-scraper')
from config import Config

def inspect():
    endpoint = f'https://{Config.AZURE_SEARCH_SERVICE}.search.windows.net'
    index_name = 'legal-court-rag-index-v2'
    
    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=DefaultAzureCredential())
    
    target_id = "Part_1___Overriding_Objective"
    print(f"Fetching {target_id}...")
    
    try:
        doc = client.get_document(key=target_id)
        print(json.dumps(doc, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
