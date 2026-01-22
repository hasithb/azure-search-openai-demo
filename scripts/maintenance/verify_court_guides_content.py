import os
import sys
import asyncio
from azure.identity.aio import AzureDeveloperCliCredential
from azure.search.documents.aio import SearchClient

# Add app/backend to path to import load_azd_env
sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'backend'))
from load_azd_env import load_azd_env

async def verify_guides():
    load_azd_env()
    service_name = os.environ.get("AZURE_SEARCH_SERVICE")
    index_name = os.environ.get("AZURE_SEARCH_INDEX")
    
    if not service_name or not index_name:
        print("Error: AZURE_SEARCH_SERVICE or AZURE_SEARCH_INDEX not set.")
        return

    endpoint = f"https://{service_name}.search.windows.net"
    credential = AzureDeveloperCliCredential()

    print(f"Connecting to {endpoint}, index: {index_name}")

    guide_categories = [
        "Commercial Court",
        "King''s Bench Division", # Escaped for OData
        "Chancery Division",
        "Patents Court",
        "Technology and Construction Court"
    ]

    async with SearchClient(endpoint=endpoint, index_name=index_name, credential=credential) as client:
        print("\n=== Verifying Court Guides Content ===")
        
        for category in guide_categories:
            display_name = category.replace("''", "'")
            print(f"\nScanning: {display_name}")
            
            # Search with count=True
            try:
                results = await client.search(
                    search_text="*",
                    filter=f"category eq '{category}'",
                    include_total_count=True,
                    select=["id", "sourcepage", "content", "storageUrl", "sourcefile"],
                    top=3
                )
                
                count = await results.get_count()
                print(f"  Total Documents: {count}")
                
                if count == 0:
                    print(f"  ❌ ERROR: No documents found for {display_name}")
                    continue
                    
                print("  Sample Documents:")
                async for doc in results:
                    print(f"    - ID: {doc['id']}")
                    print(f"      Source: {doc['sourcepage']}")
                    print(f"      File: {doc['sourcefile']}")
                    print(f"      URL: {doc['storageUrl']}")
                    
                    content_preview = doc['content'][:100].replace('\n', ' ') if doc['content'] else "[EMPTY]"
                    print(f"      Content: {content_preview}...")
                    
                    # Validation checks
                    if not doc['content']:
                        print("      ❌ Content is empty") 
                    if not doc['storageUrl']:
                        print("      ⚠️ Storage URL is missing")
                        
            except Exception as e:
                print(f"  ❌ Error querying {display_name}: {e}")

    await credential.close()

if __name__ == "__main__":
    asyncio.run(verify_guides())
