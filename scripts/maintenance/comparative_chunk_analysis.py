#!/usr/bin/env python3
"""
Phase 2 Comparative Analysis: Every Chunk (V1 vs V2)
Iterates through all locally generated V2 chunks and compares them against the V1 Azure Search Index.
"""
import sys
import os
import glob
import json
import re
from pathlib import Path
from collections import defaultdict
from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

# Add script directory to path to import config
sys.path.insert(0, 'scripts/legal-scraper')

# Configuration - Hardcoded to ensure we hit the right index
from config import Config
AZURE_SEARCH_SERVICE = Config.AZURE_SEARCH_SERVICE
INDEX_NAME = 'legal-court-rag-index-v2'  # The production index
UPLOAD_DIR = "data/legal-scraper/processed/Upload"

def sanitize_id(doc_id: str) -> str:
    """
    Mimic the legacy V1 sanitization logic to find the matching document.
    V1 replaced spaces/symbols with underscores.
    """
    # This regex mimics the typical safe_filename logic used in V1 headers/ids
    # But we found V1 IDs are like 'Part_1___Overriding_Objective'
    # While V2 raw IDs are 'Part 1 – Overriding Objective'
    
    # Logic observed in comprehensive_field_analysis.py:
    s = re.sub(r'[^a-zA-Z0-9_\-=]', '_', doc_id)
    s = re.sub(r'_{2,}', '___', s)
    s = s.strip('_')
    return s

def compare_docs():
    print(f"Connecting to Index: {INDEX_NAME} @ {AZURE_SEARCH_SERVICE}...")
    
    endpoint = f'https://{AZURE_SEARCH_SERVICE}.search.windows.net'
    client = SearchClient(
        endpoint=endpoint,
        index_name=INDEX_NAME,
        credential=DefaultAzureCredential()
    )

    files = glob.glob(os.path.join(UPLOAD_DIR, "*.json"))
    files.sort()
    
    print(f"Found {len(files)} V2 chunks to compare.")
    print("="*100)
    print(f"{'V2 Scraped ID':<50} | {'V1 Index ID':<50} | {'Status':<10} | {'Diffs'}")
    print("-" * 140)

    stats = {
        'total': 0,
        'matched': 0,
        'missing_in_v1': 0,
        'perfect_match': 0,
        'diffs': 0,
        'field_diffs': defaultdict(int)
    }

    fields_to_compare = ['content', 'category', 'sourcefile', 'sourcepage', 'oids', 'groups', 'storageUrl', 'updated']

    # Limit output lines if too many
    printed_lines = 0
    MAX_LINES = 100

    for file_path in files:
        stats['total'] += 1
        
        with open(file_path, 'r', encoding='utf-8') as f:
            v2_doc = json.load(f)

        v2_id = v2_doc.get('id')
        
        # Attempt 1: Direct Sanitization
        v1_id_candidate = sanitize_id(v2_id)
        
        # Fetch from Index
        v1_doc = None
        try:
            v1_doc = client.get_document(key=v1_id_candidate)
        except ResourceNotFoundError:
            # Fallback for chunked files? 
            # If V2 is "Part 2" but V1 was "Part_2", we found it.
            # If V2 is "Part 2 chunk 005", V1 might be "Part_2_chunk_005"
            pass

        if not v1_doc:
            stats['missing_in_v1'] += 1
            if printed_lines < MAX_LINES:
                print(f"{v2_id[:48]:<50} | {v1_id_candidate[:48]:<50} | ❌ MISS  | Doc not found in V1 Index")
                printed_lines += 1
            continue

        stats['matched'] += 1
        
        # Compare Fields
        diffs = []
        for field in fields_to_compare:
            v1_val = v1_doc.get(field)
            v2_val = v2_doc.get(field)

            # Normalization for comparison
            if field == 'groups':
                # Sort lists for comparison
                v1_val = sorted(v1_val) if v1_val else []
                v2_val = sorted(v2_val) if v2_val else []
            elif field == 'oids':
                 v1_val = sorted(v1_val) if v1_val else []
                 v2_val = sorted(v2_val) if v2_val else []
            elif field == 'updated':
                 # Relaxed check: V1 had None/Empty, V2 has date. Treat as improvement, not error?
                 # But strict diff will flag it.
                 v1_val = str(v1_val) if v1_val else "EMPTY"
                 v2_val = str(v2_val) if v2_val else "EMPTY"

            if v1_val != v2_val:
                # Content Diff Handling (ignore whitespace diffs?)
                if field == 'content':
                    val1_clean = "".join(str(v1_val).split())
                    val2_clean = "".join(str(v2_val).split())
                    if val1_clean == val2_clean:
                        continue # Ignore whitespace diffs
                    
                    diff_len = abs(len(str(v1_val)) - len(str(v2_val)))
                    diffs.append(f"content(len_diff={diff_len})")
                
                elif field == 'id':
                    # We expect ID diffs because V1 is sanitized, V2 starts with raw.
                    # But here the key we looked up is v1_id_candidate.
                    pass
                else:
                    diffs.append(f"{field}")
                
                stats['field_diffs'][field] += 1

        if not diffs:
            stats['perfect_match'] += 1
            # print(f"{v2_id[:48]:<50} | {v1_id_candidate[:48]:<50} | ✅ MATCH | Identical")
        else:
            stats['diffs'] += 1
            if printed_lines < MAX_LINES:
                print(f"{v2_id[:48]:<50} | {v1_id_candidate[:48]:<50} | ⚠️ DIFF  | {', '.join(diffs)}")
                printed_lines += 1

    print("="*100)
    print("\nSummary Statistics:")
    print(f"Total V2 Documents Processed: {stats['total']}")
    print(f"Documents Aligned in V1:      {stats['matched']}")
    print(f"Documents Missing in V1:      {stats['missing_in_v1']}")
    print(f"Perfect Content Matches:      {stats['perfect_match']}")
    print(f"Documents with Diffs:         {stats['diffs']}")
    print("\nField Discrepancies (Count of Documents with Diff):")
    for field, count in stats['field_diffs'].items():
        print(f" - {field}: {count}")

if __name__ == "__main__":
    compare_docs()
