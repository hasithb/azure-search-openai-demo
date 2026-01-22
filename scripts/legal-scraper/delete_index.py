#!/usr/bin/env python
import os
import logging
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def delete_index():
    service_name = Config.AZURE_SEARCH_SERVICE
    if not service_name:
        logger.error("AZURE_SEARCH_SERVICE not set")
        return

    endpoint = f"https://{service_name}.search.windows.net"
    index_name = os.getenv("AZURE_SEARCH_INDEX", "legal-court-rag-index-v2")
    
    logger.info(f"Connecting to {endpoint} to delete index '{index_name}'...")
    
    credential = DefaultAzureCredential()
    client = SearchIndexClient(endpoint=endpoint, credential=credential)
    
    try:
        client.delete_index(index_name)
        logger.info(f"✅ Successfully deleted index: {index_name}")
    except Exception as e:
        logger.error(f"❌ Failed to delete index: {e}")

if __name__ == "__main__":
    delete_index()
