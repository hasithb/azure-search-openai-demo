#!/usr/bin/env python3
"""Compare filterable fields between index and scraped data."""

from azure.search.documents import SearchClient
from azure.identity import AzureCliCredential
import json
from pathlib import Path

# Connect to index
credential = AzureCliCredential()
search_client = SearchClient(
    endpoint='https://cpr-rag.search.windows.net',
    index_name='legal-court-rag-index',
    credential=credential
)

print("=" * 80)
print("TESTING FILTERABLE FIELDS FOR DIFF CHECKING")
print("=" * 80)

# Get sample from index
print("\n📊 Sample from INDEX (first 3 CPR documents):")
print("-" * 80)
results = search_client.search(
    search_text='Part',
    select='id,sourcefile,sourcepage,category,storageUrl,parent_id,updated',
    top=3
)

for i, doc in enumerate(results, 1):
    print(f"\nDocument {i}:")
    print(f"  id: {doc.get('id')}")
    print(f"  sourcefile: {doc.get('sourcefile')}")
    print(f"  sourcepage: {doc.get('sourcepage')}")
    print(f"  category: {doc.get('category')}")
    print(f"  storageUrl: {doc.get('storageUrl', 'None')[:60]}...")
    print(f"  parent_id: {doc.get('parent_id')}")
    print(f"  updated: {doc.get('updated')}")

# Get sample from scraped data
print("\n" + "=" * 80)
print("📁 Sample from SCRAPED DATA:")
print("-" * 80)

scraped_dir = Path('data/legal-scraper/processed/Upload')
sample_files = ['Part 1.json', 'Part 44.json', 'Practice Direction 54A.json']

for filename in sample_files:
    filepath = scraped_dir / filename
    if filepath.exists():
        print(f"\n{filename}:")
        with open(filepath, 'r') as f:
            data = json.load(f)
            print(f"  id: {data.get('id')}")
            print(f"  sourcefile: {data.get('sourcefile')}")
            print(f"  sourcepage: {data.get('sourcepage')}")
            print(f"  category: {data.get('category')}")
            print(f"  storageUrl: {data.get('storageUrl', 'None')[:60]}...")
            print(f"  parent_id: {data.get('parent_id')}")
            print(f"  updated: {data.get('updated')}")

# Test different filter strategies
print("\n" + "=" * 80)
print("🔍 TESTING FILTER STRATEGIES:")
print("=" * 80)

# Strategy 1: Filter by storageUrl
print("\n1️⃣ Filter by storageUrl (exact match)")
test_url = "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part01"
try:
    results = list(search_client.search(
        search_text='*',
        filter=f"storageUrl eq '{test_url}'",
        select='id,sourcefile',
        top=5
    ))
    print(f"   Found {len(results)} documents with storageUrl = {test_url}")
    for r in results[:3]:
        print(f"   - {r.get('id')} (sourcefile: {r.get('sourcefile')})")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Strategy 2: Filter by category
print("\n2️⃣ Filter by category")
try:
    results = list(search_client.search(
        search_text='*',
        filter="category eq 'Civil Procedure Rules and Practice Directions'",
        select='id,sourcefile',
        top=5
    ))
    print(f"   Found {len(results)} CPR documents")
    for r in results[:3]:
        print(f"   - {r.get('id')} (sourcefile: {r.get('sourcefile')})")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Strategy 3: Filter by parent_id
print("\n3️⃣ Filter by parent_id")
try:
    results = list(search_client.search(
        search_text='*',
        filter="parent_id eq 'Part 1 – Overriding Objective'",
        select='id,parent_id',
        top=3
    ))
    print(f"   Found {len(results)} documents with parent_id = 'Part 1 – Overriding Objective'")
    for r in results[:3]:
        print(f"   - {r.get('id')}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Strategy 4: Batch filter by sourcefile (like we planned)
print("\n4️⃣ Batch filter by sourcefile (our planned approach)")
test_sourcefiles = ['Part 1', 'Part 44', 'Practice Direction 54A']
filter_parts = [f"sourcefile eq '{sf}'" for sf in test_sourcefiles]
filter_query = " or ".join(filter_parts)
try:
    results = list(search_client.search(
        search_text='*',
        filter=filter_query,
        select='id,sourcefile',
        top=20
    ))
    print(f"   Filter: {filter_query[:80]}...")
    print(f"   Found {len(results)} documents")
    for r in results[:5]:
        print(f"   - {r.get('id')} (sourcefile: {r.get('sourcefile')})")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 80)
print("✅ Field comparison complete!")
print("=" * 80)
