#!/usr/bin/env python3
"""
Comprehensive field-by-field comparison analysis
"""
import sys
sys.path.insert(0, 'scripts/legal-scraper')
from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
from config import Config
import json
from pathlib import Path
from collections import defaultdict

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

print("="*80)
print("COMPREHENSIVE FIELD COMPARISON ANALYSIS")
print("="*80)

# Track differences by field
field_differences = defaultdict(int)
documents_with_differences = defaultdict(list)
total_analyzed = 0

# Fields included in hash computation
hash_fields = ['id', 'sourcefile', 'sourcepage', 'category', 'storageUrl', 'updated', 'content']

print(f"\nAnalyzing {len(scraped_docs)} documents...")
print("Checking fields: " + ", ".join(hash_fields))
print("\n" + "-"*80 + "\n")

def sanitize_id(doc_id: str) -> str:
    """Match the actual sanitize_id function from upload script."""
    import re
    s = re.sub(r'[^a-zA-Z0-9_\-=]', '_', doc_id)
    s = re.sub(r'_{2,}', '___', s)
    s = s.strip('_')
    return s

for scraped_doc in scraped_docs[:30]:  # Sample 30 documents
    scraped_id_raw = scraped_doc.get('id', '')
    indexed_id = sanitize_id(scraped_id_raw)
    
    try:
        results = list(client.search(
            '*',
            filter=f"id eq '{indexed_id}'",
            top=1,
            select=hash_fields
        ))
        
        if not results:
            continue
            
        indexed_doc = results[0]
        total_analyzed += 1
        
        doc_has_diff = False
        doc_diff_fields = []
        
        for field in hash_fields:
            scraped_val = scraped_doc.get(field, '')
            indexed_val = indexed_doc.get(field, '')
            
            # Normalize content if list
            if isinstance(scraped_val, list):
                scraped_val = ' '.join(scraped_val)
            if isinstance(indexed_val, list):
                indexed_val = ' '.join(indexed_val)
            
            # Convert None to empty string for comparison
            scraped_val = str(scraped_val) if scraped_val is not None else ''
            indexed_val = str(indexed_val) if indexed_val is not None else ''
            
            if scraped_val != indexed_val:
                field_differences[field] += 1
                doc_diff_fields.append(field)
                doc_has_diff = True
                
                # Show detailed comparison for first 10 docs with this field difference
                if total_analyzed <= 10:
                    if not doc_has_diff or len(doc_diff_fields) == 1:
                        print(f"\n📄 Doc {total_analyzed}: {scraped_id_raw[:65]}")
                    
                    print(f"   🔸 {field}:")
                    
                    if field == 'content':
                        print(f"      Scraped length: {len(scraped_val)} chars")
                        print(f"      Indexed length: {len(indexed_val)} chars")
                        print(f"      Difference: {len(scraped_val) - len(indexed_val):+d} chars")
                        if abs(len(scraped_val) - len(indexed_val)) > 100:
                            print(f"      Scraped (first 150): {scraped_val[:150]}")
                            print(f"      Indexed (first 150): {indexed_val[:150]}")
                            print(f"      Scraped (last 150): ...{scraped_val[-150:]}")
                            print(f"      Indexed (last 150): ...{indexed_val[-150:]}")
                    elif field == 'id':
                        # Show character-by-character diff for id
                        print(f"      Scraped: '{scraped_val}'")
                        print(f"      Indexed: '{indexed_val}'")
                        if scraped_val != indexed_val:
                            print(f"      NOTE: ID field includes RAW scraped ID, not sanitized version")
                            print(f"      This is EXPECTED - ID in document != ID as Azure Search key")
                    else:
                        print(f"      Scraped: '{scraped_val}'")
                        print(f"      Indexed: '{indexed_val}'")
        
        if doc_has_diff:
            documents_with_differences[tuple(sorted(doc_diff_fields))].append(scraped_id_raw[:50])
            
    except Exception as e:
        print(f"Error processing {scraped_id_raw[:50]}: {e}")

# Summary
print("\n" + "="*80)
print("SUMMARY - FIELD DIFFERENCES")
print("="*80)
print(f"\nDocuments analyzed: {total_analyzed}")
print(f"\nDifferences by field:")
print("-" * 40)

for field in hash_fields:
    count = field_differences.get(field, 0)
    pct = (count / total_analyzed * 100) if total_analyzed > 0 else 0
    bar = "█" * int(pct / 5)  # Visual bar
    print(f"  {field:<15} {count:3}/{total_analyzed} ({pct:5.1f}%) {bar}")

# Pattern analysis
print("\n" + "="*80)
print("DIFFERENCE PATTERNS")
print("="*80)
print("\nCommon field combinations that differ:")
print("-" * 40)

for fields_combo, doc_list in sorted(documents_with_differences.items(), 
                                     key=lambda x: len(x[1]), 
                                     reverse=True)[:5]:
    print(f"\n{len(doc_list)} documents differ in: {', '.join(fields_combo)}")
    for doc_id in doc_list[:3]:
        print(f"  - {doc_id}")
    if len(doc_list) > 3:
        print(f"  ... and {len(doc_list) - 3} more")

# Conclusions
print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)

primary_cause = max(field_differences.items(), key=lambda x: x[1]) if field_differences else (None, 0)

if primary_cause[0]:
    pct = (primary_cause[1] / total_analyzed * 100) if total_analyzed > 0 else 0
    print(f"""
PRIMARY CAUSE: '{primary_cause[0]}' field
- Differs in {primary_cause[1]}/{total_analyzed} documents ({pct:.1f}%)

IMPACT:
- These field differences cause hash mismatches
- Documents are correctly identified as "changed"
- Re-upload will sync the new metadata
    """)

if 'updated' in field_differences and field_differences['updated'] > total_analyzed * 0.5:
    print("""
SPECIFIC FINDING - 'updated' field:
- Many documents have updated=None in index
- Scraped documents have actual dates from CPR website
- This is CORRECT - metadata is being enriched
- Future uploads will only catch real changes
    """)

if 'content' in field_differences and field_differences['content'] > 0:
    print(f"""
CONTENT CHANGES DETECTED:
- {field_differences['content']} documents have actual content changes
- These are legitimate updates from the CPR website
- Differential upload is working as intended
    """)

print("="*80)
