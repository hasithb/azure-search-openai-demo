import asyncio
import os
from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient

async def main():
    # Use environment variable or default to 'cpr-rag'
    service_name = os.getenv("AZURE_SEARCH_SERVICE", "cpr-rag")
    index_name = "legal-court-rag-index-v2"
    endpoint = f"https://{service_name}.search.windows.net"
    credential = DefaultAzureCredential()

    print(f"Connecting to {endpoint}...\nIndex: {index_name}")
    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    try:
        # TASK 1: Search for 'Practice Direction 53B' and Inspect Paragraph 9
        print("\n" + "="*60)
        print("TASK 1: Inspecting 'Practice Direction 53B' & Paragraph 9")
        print("="*60)
        
        # Searching for the specific document logic
        results = await client.search(
            search_text="Practice Direction 53B", 
            select=["id", "content", "sourcefile"], 
            top=20
        )
        
        found_pd53b = False
        found_para9_match = False
        
        async for result in results:
            content = result.get('content', '')
            source = result.get('sourcefile', '')
            doc_id = result.get('id')

            # Filter mostly for relevant source file
            if "53B" in source:
                found_pd53b = True
                
                # Check for "Data Protection" section (Paragraph 9)
                if "Data Protection" in content and "9" in content:
                    print(f"\n[POTENTIAL MATCH] ID: {doc_id}")
                    print(f"Source: {source}")
                    print("-" * 20 + " CONTENT SNIPPET " + "-" * 20)
                    # Print relevant chunk to check formatting
                    print(repr(content[:600])) 
                    print("-" * 60)
                    found_para9_match = True

        if not found_pd53b:
            print("WARNING: No documents found with '53B' in the source filename.")
        elif not found_para9_match:
            print("WARNING: Found PD 53B documents, but none contained both 'Data Protection' and '9' in the same chunk.")

        # TASK 2: Search for Pre-Action Protocol
        print("\n" + "="*60)
        print("TASK 2: Searching for 'Pre-Action Protocol' Documents")
        print("="*60)
        
        query_pap = "\"Pre-Action Protocol\" OR \"Practice Direction Pre-Action Conduct\""
        results_pap = await client.search(search_text=query_pap, select=["id", "sourcefile"], top=10)
        
        pap_count = 0
        print(f"Query: {query_pap}\n")
        async for result in results_pap:
            pap_count += 1
            print(f"{pap_count}. ID: {result.get('id')}")
            print(f"   Source: {result.get('sourcefile')}")

        if pap_count == 0:
            print("\nRESULT: No documents found matching the Pre-Action Protocol query.")
        else:
            print(f"\nRESULT: Found {pap_count} matches (top 10 displayed).")

    except Exception as e:
        print(f"\nERROR: {e}")
    finally:
        await client.close()
        await credential.close()

if __name__ == "__main__":
    asyncio.run(main())
