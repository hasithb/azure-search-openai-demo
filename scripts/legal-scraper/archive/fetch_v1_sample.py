import os
import json
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

def run():
    service_name = "cpr-rag"
    index_name = "legal-court-rag-index"
    endpoint = f"https://{service_name}.search.windows.net"
    
    # Use standard Azure credential
    credential = DefaultAzureCredential()
    
    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
    
    # print(f"Searching {index_name} at {service_name}...")
    try:
        results = client.search(
            search_text="Part 2 Application and Interpretation", 
            filter="category eq 'Civil Procedure Rules and Practice Directions'",
            top=1
        )
        
        found = False
        for res in results:
            found = True
            # Clean fields for readable output
            for k in list(res.keys()):
                if k.startswith('@search'):
                    del res[k]
                    
            if 'embedding' in res:
                res['embedding'] = "(vector hidden)"
            if 'imageEmbedding' in res:
                del res['imageEmbedding']
            
            print(json.dumps(res, indent=2))
            break
            
        if not found:
            print("No results found matching filter criteria.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run()
