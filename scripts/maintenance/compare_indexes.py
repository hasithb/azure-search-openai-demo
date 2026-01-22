#!/usr/bin/env python3
"""Compare how the same document appears in v1 vs v2 indexes."""

from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
import os
import json

def main():
    # Get service name from environment
    service = os.environ.get('AZURE_SEARCH_SERVICE', 'cpr-rag')
    endpoint = f"https://{service}.search.windows.net"
    
    print(f"Using endpoint: {endpoint}\n")
    
    credential = DefaultAzureCredential()
    
    # The exact ID from the JSON file
    doc_id = "Part 2 – Application And Interpretation Of The Rules\n\n[Part_chunk_005"
    
    print("=" * 80)
    print("SEARCHING FOR DOCUMENT IN BOTH INDEXES")
    print("=" * 80)
    print(f"Document ID: {doc_id}")
    print()
    
    # Search v1 index
    print("=" * 80)
    print("V1 INDEX (legal-court-rag-index)")
    print("=" * 80)
    
    client_v1 = SearchClient(
        endpoint=endpoint, 
        index_name='legal-court-rag-index', 
        credential=credential
    )
    
    results_v1 = list(client_v1.search(
        search_text='Footnotes 1976 c.36 2002 c.38 1983 c.2',
        select=['id', 'content', 'category', 'sourcepage', 'sourcefile'],
        top=5
    ))
    
    if results_v1:
        # Find the exact matching document
        doc = None
        for result in results_v1:
            if result['id'] == doc_id:
                doc = result
                break
        
        if doc:
            print(f"✓ FOUND IN V1")
            print(f"ID: {doc['id']}")
            print(f"Category: {doc.get('category', 'N/A')}")
            print(f"Sourcepage: {doc.get('sourcepage', 'N/A')}")
            print(f"Sourcefile: {doc.get('sourcefile', 'N/A')}")
            print(f"\nContent (first 500 chars):")
            print(doc['content'][:500])
            print("\n(Note: v1 does NOT have subsection_id or subsections fields)")
        else:
            print(f"✗ NOT FOUND IN V1 (searched 5 results)")
    else:
        print("✗ NOT FOUND IN V1")
    
    print("\n" + "=" * 80)
    print("V2 INDEX (legal-court-rag-index-v2)")
    print("=" * 80)
    
    # Search v2 index
    client_v2 = SearchClient(
        endpoint=endpoint, 
        index_name='legal-court-rag-index-v2', 
        credential=credential
    )
    
    results_v2 = list(client_v2.search(
        search_text='Footnotes 1976 c.36 2002 c.38 1983 c.2',
        select=['id', 'content', 'category', 'sourcepage', 'sourcefile'],
        top=5
    ))
    
    if results_v2:
        # Find the exact matching document
        doc = None
        for result in results_v2:
            if result['id'] == doc_id:
                doc = result
                break
        
        if doc:
            print(f"✓ FOUND IN V2")
            print(f"ID: {doc['id']}")
            print(f"Category: {doc.get('category', 'N/A')}")
            print(f"Sourcepage: {doc.get('sourcepage', 'N/A')}")
            print(f"Sourcefile: {doc.get('sourcefile', 'N/A')}")
            print(f"⚠️  NOTE: subsection_id and subsections fields DO NOT EXIST in v2 schema yet")
            print(f"\nContent (first 500 chars):")
            print(doc['content'][:500])
        else:
            print(f"✗ NOT FOUND IN V2 (searched 5 results)")
    else:
        print("✗ NOT FOUND IN V2")
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    # Find matching docs in both results
    v1_doc = None
    v2_doc = None
    
    if results_v1:
        for result in results_v1:
            if result['id'] == doc_id:
                v1_doc = result
                break
    
    if results_v2:
        for result in results_v2:
            if result['id'] == doc_id:
                v2_doc = result
                break
    
    if v1_doc and v2_doc:
        print("✓ Document exists in BOTH indexes")
        print("\nKey differences:")
        print("- v1: Original format (no subsection fields)")
        print("- v2: Includes subsection_id and subsections[] fields")
        
        # Check if content matches
        if v1_doc['content'] == v2_doc['content']:
            print("- Content is IDENTICAL between v1 and v2")
        else:
            print("- ⚠️  Content DIFFERS between v1 and v2")
    elif v1_doc:
        print("⚠️  Document only in v1 (NOT migrated to v2)")
    elif v2_doc:
        print("⚠️  Document only in v2 (new document)")
    else:
        print("✗ Document not found in either index")

if __name__ == "__main__":
    main()
