#!/usr/bin/env python
"""
Compare Local Scraped Data (New Architecture) vs V1 Production Index.
"""
import os
import sys
import json
import logging
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import Config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def get_search_client(index_name):
    endpoint = f"https://{Config.AZURE_SEARCH_SERVICE}.search.windows.net"
    credential = DefaultAzureCredential()
    return SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

def analyze_differences():
    v1_index_name = "legal-court-rag-index"
    logger.info(f"Connecting to V1 Index: {v1_index_name}")
    
    client = get_search_client(v1_index_name)
    
    # 1. Get V1 Stats
    v1_count = client.get_document_count()
    logger.info(f"V1 Index Total Documents: {v1_count}")
    
    # 2. Sample V1 IDs
    logger.info("Sampling V1 Document IDs...")
    v1_map = {} # Title -> Count
    v1_samples = []
    
    # fetch all content is too heavy. Let's fetch select fields
    # We iterate using a simple search "*"
    
    total_fetched = 0
    results = client.search(search_text="*", select=["id", "sourcepage", "content"], top=2000)
    
    for res in results:
        total_fetched += 1
        title = res.get('sourcepage')
        if title:
            v1_map[title] = v1_map.get(title, 0) + 1
            
        if len(v1_samples) < 3:
            v1_samples.append(res)
            
    logger.info(f"Analyzed {total_fetched} documents from V1.")
    
    # 3. Load Local Documents (V2 candidate)
    local_dir = os.path.join(Config.PROCESSED_DIR, "Upload")
    local_docs = []
    local_map = {} # Title -> Count
    import glob
    files = glob.glob(os.path.join(local_dir, "*.json"))
    for f in files:
        try:
            with open(f, 'r') as fd:
                data = json.load(fd)
                if isinstance(data, list):
                    docs = data
                else:
                    docs = [data]
                
                for d in docs:
                    local_docs.append(d)
                    t = d.get('sourcepage')
                    if t:
                        local_map[t] = local_map.get(t, 0) + 1
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")

    logger.info(f"Local (V2) Documents: {len(local_docs)}")

    # 4. Compare
    logger.info("\n--- COMPARISON ---")
    
    v1_titles = set(v1_map.keys())
    local_titles = set(local_map.keys())
    
    common = v1_titles.intersection(local_titles)
    missing_in_v1 = local_titles - v1_titles
    missing_in_local = v1_titles - local_titles
    
    logger.info(f"Common Titles (Source Pages): {len(common)}")
    logger.info(f"Titles Only in Local (New): {len(missing_in_v1)}")
    logger.info(f"Titles Only in V1 (Old/Renamed): {len(missing_in_local)}")
    
    # Granularity Comparison
    if common:
        sample_title = list(common)[0]
        logger.info(f"\ngranularity Check for: '{sample_title}'")
        logger.info(f"  V1 Chunk Count: {v1_map[sample_title]}")
        logger.info(f"  V2 Chunk Count: {local_map[sample_title]}")
        
    
if __name__ == "__main__":
    analyze_differences()
