from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

endpoint = 'https://cpr-rag.search.windows.net'
index_name = 'legal-court-rag-index-v2'
credential = DefaultAzureCredential()

client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

def search_rule():
    search_text = "31.16"
    # Note: sourcefile might store the full filename like "Part_31_....pdf" or similar.
    # It's safer to not strictly filter by exact match "Part 31" unless we know the format.
    # However, user asked for it. I will try a filter if possible, but search text is strong enough for "31.16".
    # Let's try searching first.
    
    print(f"Searching index '{index_name}' for '{search_text}'...")
    
    # Selecting likely fields
    results = client.search(
        search_text=search_text,
        select=['id', 'content', 'sourcefile', 'sourcepage', 'category'],
        top=5
    )
    
    found = False
    for result in results:
        sourcefile = result.get('sourcefile', '')
        content = result.get('content', '')
        
        # Check if sourcefile looks like Part 31 (case insensitive)
        if "Part 31" in sourcefile or "Part_31" in sourcefile:
            print("-" * 80)
            print(f"ID: {result['id']}")
            print(f"Source File: {sourcefile}")
            print(f"Source Page: {result.get('sourcepage')}")
            print("-" * 80)
            print("CONTENT START")
            print(content)
            print("CONTENT END")
            print("-" * 80)
            
            if "31.16" in content and "Disclosure before proceedings start" in content:
                found = True
                print(">>> SUCCESS: Found Rule 31.16 with title.")
            elif "31.16" in content:
                 print(">>> PARTIAL: Found '31.16' but maybe not title.")
    
    if not found:
        print(">>> WARNING: Did not find Rule 31.16 with title in 'Part 31' documents.")

if __name__ == "__main__":
    search_rule()
