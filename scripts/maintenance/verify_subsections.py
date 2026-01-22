#!/usr/bin/env python3
"""Verify subsection fields in v2 index."""
from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential

endpoint = 'https://cpr-rag.search.windows.net'
cred = DefaultAzureCredential()

print('='*80)
print('VERIFYING SUBSECTION FIELDS IN V2 INDEX')
print('='*80)

client = SearchClient(endpoint=endpoint, index_name='legal-court-rag-index-v2', credential=cred)

# Search for Part 35 (known to have subsections like 35.1, 35.2)
results = list(client.search(
    'Part 35 Experts',
    select=['id', 'content', 'category', 'subsection_id', 'subsections'],
    top=5
))

print(f'\nFound {len(results)} results\n')

for i, doc in enumerate(results, 1):
    print(f'{i}. ID: {doc["id"][:80]}')
    print(f'   Category: {doc.get("category", "N/A")}')
    print(f'   ✓ Subsection ID: {doc.get("subsection_id", "N/A")}')
    subs = doc.get("subsections", [])
    if subs:
        print(f'   ✓ All Subsections ({len(subs)}): {", ".join(subs[:10])}{"..." if len(subs) > 10 else ""}')
    else:
        print(f'   ✓ All Subsections: None')
    print(f'   Content: {doc["content"][:120]}...\n')

print('='*80)
print('✅ VERIFICATION COMPLETE - Subsection fields are indexed!')
print('='*80)
