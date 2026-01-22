"""Check for duplicate IDs in Azure Search index."""
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from collections import Counter
import sys
sys.path.insert(0, 'scripts/legal-scraper')
from config import Config

def check_duplicates(index_name: str):
    """Check for duplicate document IDs in the index."""
    endpoint = f'https://{Config.AZURE_SEARCH_SERVICE}.search.windows.net'
    
    client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(Config.AZURE_SEARCH_KEY)
    )
    
    print(f"\n{'='*60}")
    print(f"Checking for duplicates in: {index_name}")
    print(f"{'='*60}\n")
    
    # Fetch all document IDs
    all_ids = []
    results = client.search('*', select='id')
    
    for result in results:
        all_ids.append(result['id'])
    
    total_docs = len(all_ids)
    unique_ids = len(set(all_ids))
    
    print(f"Total documents: {total_docs}")
    print(f"Unique IDs: {unique_ids}")
    
    if total_docs == unique_ids:
        print("✅ NO DUPLICATES FOUND - Index is clean!")
        return True
    else:
        print(f"❌ DUPLICATES DETECTED: {total_docs - unique_ids} duplicate documents")
        
        # Find and report duplicates
        id_counts = Counter(all_ids)
        duplicates = {id_val: count for id_val, count in id_counts.items() if count > 1}
        
        print(f"\n{'='*60}")
        print(f"Duplicate IDs (showing first 10):")
        print(f"{'='*60}")
        for id_val, count in list(duplicates.items())[:10]:
            print(f"  {id_val}: {count} copies")
        
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more duplicates")
        
        return False

if __name__ == "__main__":
    print("\n🔍 Azure Search Index Duplicate Check")
    print("=" * 60)
    
    # Check both indexes
    v2_clean = check_duplicates('legal-court-rag-index-v2')
    
    print("\n" + "="*60)
    if v2_clean:
        print("✅ Index validation passed - ready for Phase 2 testing")
    else:
        print("❌ Index has duplicates - needs cleanup before Phase 2")
    print("="*60 + "\n")
