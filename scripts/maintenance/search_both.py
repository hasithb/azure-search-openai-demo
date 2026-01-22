#!/usr/bin/env python3
from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential

endpoint = 'https://cpr-rag.search.windows.net'
cred = DefaultAzureCredential()

print('='*80)
print('V1 INDEX: legal-court-rag-index')
print('='*80)
client_v1 = SearchClient(endpoint=endpoint, index_name='legal-court-rag-index', credential=cred)
results = list(client_v1.search('Part 2 chunk_005', select=['id', 'content', 'category'], top=5))
print(f'Found {len(results)} results\n')
for i, r in enumerate(results, 1):
    print(f"{i}. ID: {r['id']}")
    print(f"   Category: {r.get('category', 'N/A')}")
    print(f"   Content: {r['content'][:200]}...\n")

print('='*80)
print('V2 INDEX: legal-court-rag-index-v2')  
print('='*80)
client_v2 = SearchClient(endpoint=endpoint, index_name='legal-court-rag-index-v2', credential=cred)
results2 = list(client_v2.search('PART 2 APPLICATION INTERPRETATION Footnotes', select=['id', 'content', 'category'], top=5))
print(f'Found {len(results2)} results\n')
for i, r in enumerate(results2, 1):
    print(f"{i}. ID: {r['id']}")
    print(f"   Category: {r.get('category', 'N/A')}")
    print(f"   Content: {r['content'][:200]}...\n")

print('='*80)
print('KEY FINDINGS')
print('='*80)
print(f"✓ V1 has {len(results)} Part 2 documents")
print(f"✓ V2 has {len(results2)} Part 2 related documents")
print("⚠️  v2 does NOT have subsection_id or subsections fields in schema yet")
print("⚠️  These fields need to be added before re-uploading documents")
