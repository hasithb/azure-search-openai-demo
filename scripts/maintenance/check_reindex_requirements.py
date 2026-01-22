#!/usr/bin/env python3
"""Check if reindexing is required to make 'id' field filterable."""

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.identity import AzureCliCredential

credential = AzureCliCredential()
index_client = SearchIndexClient(
    endpoint='https://cpr-rag.search.windows.net',
    credential=credential
)

# Get current index definition
index = index_client.get_index('legal-court-rag-index')

print("=" * 80)
print("CURRENT INDEX SCHEMA - 'id' field")
print("=" * 80)

# Find the id field
for field in index.fields:
    if field.name == 'id':
        print(f"\nField: {field.name}")
        print(f"  Type: {field.type}")
        print(f"  Key: {field.key}")
        print(f"  Searchable: {field.searchable}")
        print(f"  Filterable: {field.filterable}")
        print(f"  Sortable: {field.sortable}")
        print(f"  Facetable: {field.facetable}")
        break

print("\n" + "=" * 80)
print("AZURE SEARCH SCHEMA CHANGE RULES")
print("=" * 80)

print("""
Making 'id' filterable requires understanding Azure Search schema constraints:

1. ✅ CAN MODIFY WITHOUT REINDEXING:
   - Adding NEW fields
   - Changing 'retrievable' attribute
   - Adding/removing suggesters
   - Modifying scoring profiles

2. ❌ REQUIRES REINDEXING (Breaking Changes):
   - Changing field TYPE (e.g., Edm.String -> Edm.Int32)
   - Changing 'key' attribute
   - Changing 'searchable', 'filterable', 'sortable', 'facetable' on EXISTING fields
   - Deleting fields
   - Changing analyzer on existing fields

3. ⚠️ FOR KEY FIELDS (id):
   - Key field is typically NOT filterable by default
   - Making key field filterable IS a breaking change
   - Requires reindexing all documents
""")

print("\n" + "=" * 80)
print("REINDEXING OPTIONS")
print("=" * 80)

print("""
Option A: DROP AND RECREATE (Simple but destructive)
  1. Delete existing index
  2. Create new index with filterable 'id'
  3. Re-upload all documents
  ⏱️ Time: ~30-60 minutes
  ⚠️ Risk: Downtime during rebuild

Option B: CREATE NEW INDEX + MIGRATE (Zero downtime)
  1. Create new index 'legal-court-rag-index-v2' with filterable 'id'
  2. Copy all documents from old to new index
  3. Switch application to use new index
  4. Delete old index
  ⏱️ Time: ~1-2 hours
  ✅ Benefit: No downtime

Option C: LIVE MIGRATION (Complex but safe)
  1. Create new index with filterable 'id'
  2. Dual-write: Upload new docs to both indexes
  3. Backfill old documents to new index
  4. Switch application to use new index
  5. Delete old index
  ⏱️ Time: 1-3 days (gradual)
  ✅ Benefit: Safest, can rollback
""")

print("\n" + "=" * 80)
print("CHECKING INDEX SIZE")
print("=" * 80)

search_client = SearchClient(
    endpoint='https://cpr-rag.search.windows.net',
    index_name='legal-court-rag-index',
    credential=credential
)

# Get document count
results = search_client.search(search_text='*', include_total_count=True)
doc_count = results.get_count()

print(f"\nTotal documents in index: {doc_count}")
print(f"Estimated reindex time (at 10 docs/sec): {doc_count / 10 / 60:.1f} minutes")
print(f"Estimated reindex time (at 50 docs/sec): {doc_count / 50 / 60:.1f} minutes")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

print(f"""
✅ YES, REINDEXING IS REQUIRED

Making the 'id' field filterable is a BREAKING CHANGE in Azure Search.
You cannot simply update the schema on the existing index.

Given your index has {doc_count} documents, recommended approach:

🎯 OPTION B: CREATE NEW INDEX + MIGRATE (Recommended)

Why this is best:
  ✅ Zero downtime - old index stays operational
  ✅ Can validate new index before switching
  ✅ Easy rollback if issues found
  ✅ Clean migration process
  ✅ Reasonable time investment ({doc_count / 50 / 60:.1f}-{doc_count / 10 / 60:.1f} minutes)

Next steps:
  1. Create migration script to:
     - Create new index with filterable 'id'
     - Copy all {doc_count} documents
     - Validate migration
  2. Update application to use new index
  3. Update GitHub workflow secrets
  4. Delete old index when confirmed working

Would you like me to create the migration script?
""")
