#!/usr/bin/env python
"""
Deep Analysis of V1 vs V2 content discrepancies.
Focuses on identifying naming mismatches and scope differences.
"""
import os
import sys
import json
import re
import logging
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from config import Config
except ImportError:
    class Config:
        AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
        PROCESSED_DIR = "data/legal-scraper/processed"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def normalize_title(title):
    """Normalize title for fuzzy comparison."""
    if not title:
        return ""
    # Remove file extensions
    t = re.sub(r'\.json$', '', title, flags=re.IGNORECASE)
    t = re.sub(r'\.pdf$', '', t, flags=re.IGNORECASE)
    # Remove common prefixes
    t = t.lower()
    t = t.replace("practice direction", "pd")
    t = t.replace("civil procedure rules", "")
    t = t.replace("part", "")
    # Remove non-alphanumeric
    t = re.sub(r'[^a-z0-9]', '', t)
    return t

def get_search_client(index_name):
    service_name = Config.AZURE_SEARCH_SERVICE
    if not service_name:
         # Try to get from environment
         import subprocess
         try:
            res = subprocess.run("azd env get-values", shell=True, capture_output=True, text=True)
            for line in res.stdout.splitlines():
                if "AZURE_SEARCH_SERVICE" in line:
                    service_name = line.split("=")[1].strip('"')
         except:
             pass
    
    if not service_name:
        logger.error("Could not find AZURE_SEARCH_SERVICE")
        return None

    endpoint = f"https://{service_name}.search.windows.net"
    credential = DefaultAzureCredential()
    return SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

def analyze():
    # 1. Load Local V2 Data
    local_dir = os.path.join(Config.PROCESSED_DIR, "Upload")
    local_metadata = {} # normalized -> {original, file}
    
    import glob
    files = glob.glob(os.path.join(local_dir, "*.json"))
    logger.info(f"Loading {len(files)} local V2 files...")
    
    for f in files:
        try:
            with open(f, 'r') as fd:
                # Handle potentially corrupt files (concatenated JSON) by reading only first valid JSON
                content = fd.read()
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract first object
                    match = re.search(r'\{(?:[^{}]|(?R))*\}', content)
                    if not match:
                         # Simple brace counting fallback
                         idx = content.find('}{')
                         if idx != -1:
                             content = content[:idx+1]
                             data = json.loads(content)
                         else:
                             continue
                    else:
                         continue
                
                if isinstance(data, list):
                    items = data
                else:
                    items = [data]
                
                for item in items:
                    title = item.get('sourcepage') or item.get('title') or ""
                    norm = normalize_title(title)
                    if norm:
                        local_metadata[norm] = {'original': title, 'id': item.get('id')}
                        
        except Exception as e:
            # logger.warning(f"Skipping {f}: {e}")
            pass

    logger.info(f"Found {len(local_metadata)} unique normalized titles in V2.")

    # 2. Fetch V1 Data
    v1_index = "legal-court-rag-index"
    client = get_search_client(v1_index)
    if not client:
        return

    target_category = "Civil Procedure Rules and Practice Directions"
    logger.info(f"Fetching V1 index headers from {v1_index} (Filter: Category='{target_category}')...")
    v1_metadata = {} # normalized -> original
    total_v1 = 0
    
    # Filter by category
    results = client.search(
        search_text="*", 
        filter=f"category eq '{target_category}'",
        select=["sourcepage", "id"], 
        top=3000
    )
    
    print("\n--- DEBUG: RAW DATA SAMPLES ---")
    print("V1 'sourcepage' samples:")
    
    v1_samples = []
    
    for res in results:
        total_v1 += 1
        title = res.get('sourcepage')
        if len(v1_samples) < 5:
            v1_samples.append(title)
            
        if title:
            norm = normalize_title(title)
            if len(v1_samples) <= 5: 
                print(f"  Raw: '{title}' -> Norm: '{norm}'")
                
            if norm:
                v1_metadata[norm] = {'original': title, 'id': res.get('id')}

    logger.info(f"Dimensions: V1 has {len(v1_metadata)} unique normalized titles. V2 has {len(local_metadata)}.")
    
    # 3. Intersection Analysis
    v1_keys = list(v1_metadata.keys())
    v2_keys = list(local_metadata.keys())
    
    # Missing in V2 (i.e. present in V1 but not V2)
    missing_in_v2_strict = set(v1_keys) - set(v2_keys)
    
    # Fuzzy Matching Logic
    confirmed_missing = []
    fuzzy_matches = []
    
    for v1_k in missing_in_v2_strict:
        found_match = False
        for v2_k in v2_keys:
            # Check if one is contained in the other
            if v1_k in v2_k or v2_k in v1_k:
                found_match = True
                fuzzy_matches.append((v1_metadata[v1_k]['original'], v1_k, v2_k))
                break
        
        if not found_match:
            confirmed_missing.append(v1_metadata[v1_k]['original'])

    logger.info(f"\n--- GAP ANALYSIS (Fuzzy Matched) ---")
    logger.info(f"Total V1 Items: {len(v1_metadata)}")
    logger.info(f"Fuzzy Matches: {len(fuzzy_matches)}")
    logger.info(f"Confirmed Missing in V2: {len(confirmed_missing)}")
    
    if confirmed_missing:
        logger.info("\n--- EXAMPLES OF CONFIRMED MISSING ---")
        for m in sorted(confirmed_missing)[:20]:
            logger.info(f" - {m}")
    
    # Categorize the missing ones
    categories = {'files': 0, 'parts': 0, 'pds': 0, 'other': 0}
    examples = []
    
    for k in missing_in_v2_strict:
        orig = v1_metadata[k]['original']
        if len(examples) < 20:
            examples.append(orig)
            
        if orig.startswith('file-'):
            categories['files'] += 1
        elif 'part' in orig.lower():
            categories['parts'] += 1
        elif 'practice' in orig.lower():
            categories['pds'] += 1
        else:
            categories['other'] += 1
            
    logger.info("\nBreakdown of docs found in V1 but NOT in V2:")
    logger.info(f"  Files (file-*): {categories['files']}")
    logger.info(f"  Parts: {categories['parts']}")
    logger.info(f"  Practice Directions: {categories['pds']}")
    logger.info(f"  Other: {categories['other']}")
    
    print("\n--- EXAMPLES OF V1 DOCS MISSING IN V2 ---")
    for ex in sorted(examples):
        print(f" - {ex}")

    # Check for near matches (e.g. Part 06 vs Part 6)
    logger.info("\nChecking for formatting mismatches (e.g. '01' vs '1')...")
    mismatches = 0
    for k in missing_in_v2_strict:
        # Try stripping leading zeros from numbers
        # e.g. pd06a -> pd6a
        relaxed = re.sub(r'0+(\d)', r'\1', k)
        if relaxed in v2_keys:
            v2_orig = local_metadata[relaxed]['original']
            v1_orig = v1_metadata[k]['original']
            if mismatches < 5:
                print(f"  MISMATCH: V1='{v1_orig}'  vs  V2='{v2_orig}'")
            mismatches += 1
            
    if mismatches > 0:
        logger.info(f"  Found {mismatches} documents that exist in both but are named differently (e.g. leading zeros).")
        logger.info(f"  True Missing Count is approx: {len(missing_in_v2) - mismatches}")

if __name__ == "__main__":
    analyze()
