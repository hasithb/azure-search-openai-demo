#!/usr/bin/env python3
"""Validate migration and test filterable id field."""

from azure.search.documents import SearchClient
from azure.identity import AzureCliCredential

credential = AzureCliCredential()

# Test old index (should fail - id not filterable)
old_client = SearchClient(
    endpoint='https://cpr-rag.search.windows.net',
    index_name='legal-court-rag-index',
    credential=credential
)

# Test new index (should work - id is filterable)
new_client = SearchClient(
    endpoint='https://cpr-rag.search.windows.net',
    index_name='legal-court-rag-index-v2',
    credential=credential
)

print("=" * 80)
print("MIGRATION VALIDATION TEST")
print("=" * 80)

# Get a sample ID from old index
sample = list(old_client.search(search_text="*", top=1))[0]
sample_id = sample['id']

print(f"\nTest ID: {sample_id}")

# Test 1: Try filtering old index (should fail)
print("\n1️⃣ Test OLD index (id NOT filterable):")
try:
    results = list(old_client.search(
        search_text="*",
        filter=f"id eq '{sample_id}'",
        top=1
    ))
    print(f"   ❌ Unexpected: Filter worked (should have failed)")
except Exception as e:
    print(f"   ✅ Expected error: {str(e)[:100]}...")

# Test 2: Try filtering new index (should work)
print("\n2️⃣ Test NEW index (id IS filterable):")
try:
    results = list(new_client.search(
        search_text="*",
        filter=f"id eq '{sample_id}'",
        top=1
    ))
    
    if len(results) == 1 and results[0]['id'] == sample_id:
        print(f"   ✅ Filter successful! Found: {sample_id}")
    else:
        print(f"   ❌ Filter returned wrong results")
        
except Exception as e:
    print(f"   ❌ Unexpected error: {e}")

# Test 3: Batch filter multiple IDs
print("\n3️⃣ Test batch filter (multiple IDs):")
sample_ids = [doc['id'] for doc in old_client.search(search_text="*", top=3)]
filter_query = " or ".join([f"id eq '{id}'" for id in sample_ids])

try:
    results = list(new_client.search(
        search_text="*",
        filter=filter_query,
        top=10
    ))
    
    print(f"   ✅ Batch filter successful! Found {len(results)}/3 documents")
    for r in results:
        print(f"      - {r['id']}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Final count check
old_count = old_client.search(search_text="*", include_total_count=True, top=1).get_count()
new_count = new_client.search(search_text="*", include_total_count=True, top=1).get_count()

print(f"\n{'=' * 80}")
print(f"FINAL VALIDATION")
print(f"{'=' * 80}")
print(f"Old index: {old_count} documents")
print(f"New index: {new_count} documents")

if old_count == new_count == 1127:
    print(f"\n✅ MIGRATION SUCCESSFUL!")
    print(f"\nNext steps:")
    print(f"  1. Update app backend to use: legal-court-rag-index-v2")
    print(f"  2. Update GitHub workflow secrets")
    print(f"  3. Update upload script to use id-based diff checking")
else:
    print(f"\n⚠️  Document count mismatch")

old_client.close()
new_client.close()
