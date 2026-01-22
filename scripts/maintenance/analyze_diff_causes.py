#!/usr/bin/env python3
"""
Analyze why so many documents are showing as "changed"
"""
import sys
sys.path.insert(0, 'scripts/legal-scraper')

from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
from config import Config
import json
from pathlib import Path

def main():
    # Initialize search client
    endpoint = f'https://{Config.AZURE_SEARCH_SERVICE}.search.windows.net'
    client = SearchClient(
        endpoint=endpoint,
        index_name='legal-court-rag-index-v2',
        credential=DefaultAzureCredential()
    )
    
    # Load scraped documents  
    scraped_path = Path('./data/legal-scraper/processed/Upload/civil_procedure_rules_review.json')
    with open(scraped_path, 'r') as f:
        scraped_docs = json.load(f)
    
    print(f"Loaded {len(scraped_docs)} scraped documents\n")
    
    # Analyze first 20 documents
    print("="*80)
    print("DETAILED DIFFERENCE ANALYSIS")
    print("="*80)
    
    analyzed = 0
    field_diffs = {}
    
    for scraped_doc in scraped_docs[:20]:
        # Convert ID format (spaces to underscores for query)
        scraped_id_raw = scraped_doc.get('id', '')
        indexed_id = scraped_id_raw.replace(' ', '___').replace('–', '___').replace('—', '___')
        
        try:
            # Query index
            results = list(client.search(
                '*',
                filter=f"id eq '{indexed_id}'",
                top=1
            ))
            
            if not results:
                print(f"\n{analyzed+1}. SKIPPED: {scraped_id_raw[:60]} (not in index)")
                continue
            
            indexed_doc = results[0]
            analyzed += 1
            
            # Compare each field
            print(f"\n{analyzed}. {scraped_id_raw[:60]}")
            print("-" * 80)
            
            has_diff = False
            for field in ['id', 'updated', 'category', 'sourcefile', 'storageUrl']:
                scraped_val = scraped_doc.get(field, '')
                indexed_val = indexed_doc.get(field, '')
                
                if str(scraped_val) != str(indexed_val):
                    has_diff = True
                    if field not in field_diffs:
                        field_diffs[field] = 0
                    field_diffs[field] += 1
                    
                    print(f"   {field}:")
                    print(f"      Scraped:  {repr(scraped_val)[:100]}")
                    print(f"      Indexed:  {repr(indexed_val)[:100]}")
            
            if not has_diff:
                print("   ✅ No metadata differences")
            
            if analyzed >= 10:
                break
                
        except Exception as e:
            print(f"\nError processing {scraped_id_raw}: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nDocuments analyzed: {analyzed}")
    print(f"\nField differences found:")
    for field, count in sorted(field_diffs.items(), key=lambda x: x[1], reverse=True):
        pct = (count / analyzed) * 100 if analyzed > 0 else 0
        print(f"  {field}: {count}/{analyzed} ({pct:.1f}%)")

if __name__ == '__main__':
    main()
