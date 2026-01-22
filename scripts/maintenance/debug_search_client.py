import asyncio
import os
from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient

async def main():
    service_name = os.getenv("AZURE_SEARCH_SERVICE", "cpr-rag") 
    index_name = os.getenv("AZURE_SEARCH_INDEX", "legal-court-rag-index-v2")
    endpoint = f"https://{service_name}.search.windows.net"
    credential = DefaultAzureCredential()

    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
    
    print(f"Index: {index_name}")

    try:
        print("Searching for 'Part 35' (sourcefile field)...")
        # Trying to find documents that originated from Part 35
        # Note: Filter might be case sensitive or fuzzy. 
        results = await client.search(search_text="*", filter="sourcefile eq 'Part 35'", top=5)
        
        count = 0
        async for result in results:
             count += 1
             print(f"\nID: {result.get('id')}")
             print("Content Start:")
             print(repr(result.get('content')[:300])) # Use repr to see newlines
             print("-" * 20)
        
        if count == 0:
            print("No documents found with sourcefile='Part 35'. Trying free text search 'Part 35'")
            
            results = await client.search(search_text="Part 35", top=5)
            async for result in results:
                 print(f"\nID: {result.get('id')}")
                 print(f"Sourcefile: {result.get('sourcefile')}")
                 print("Content Start:")
                 print(repr(result.get('content')[:300]))
                 print("-" * 20)

    except Exception as e:
        print(f"Error: {e}")

    await client.close()
    await credential.close()

if __name__ == "__main__":
    asyncio.run(main())
