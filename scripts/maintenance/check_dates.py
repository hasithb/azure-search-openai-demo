#!/usr/bin/env python3
import sys
sys.path.insert(0, 'scripts/legal-scraper')
from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
from config import Config

endpoint = f'https://{Config.AZURE_SEARCH_SERVICE}.search.windows.net'
client = SearchClient(
    endpoint=endpoint,
    index_name='legal-court-rag-index-v2',
    credential=DefaultAzureCredential()
)

test_ids = [
    "Notes_on_Practice_Directions",
    "Part_1___Overriding_Objective",
    "Practice_Direction_1A__participation_of_vulnerable_parties_or_witnesses",
    "Part_2___Application_and_Interpretation_of_the_Rules"
]

print("INDEXED DOCUMENTS - Updated dates:\n")
for doc_id in test_ids:
    results = list(client.search('*', filter=f"id eq '{doc_id}'", top=1, select=['id', 'updated']))
    if results:
        doc = results[0]
        print(f"{doc_id[:50]:<52} updated: {doc.get('updated')}")
    else:
        print(f"{doc_id:<50} NOT FOUND")
