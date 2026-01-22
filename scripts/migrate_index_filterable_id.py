#!/usr/bin/env python3
"""
Migrate Azure Search index to make 'id' field filterable.

This script:
1. Creates a new index with filterable 'id' field
2. Copies all documents from old to new index
3. Validates the migration
4. Provides rollback instructions
"""

import logging
import time
from typing import List, Dict, Any
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchAlgorithmConfiguration,
    HnswAlgorithmConfiguration,
    HnswParameters,
    BinaryQuantizationCompression,
    VectorSearchCompression,
    RescoringOptions,
    VectorSearchCompressionRescoreStorageMethod,
    SemanticSearch,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
)
from azure.identity import AzureCliCredential
from azure.core.exceptions import ResourceNotFoundError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
OLD_INDEX_NAME = "legal-court-rag-index"
NEW_INDEX_NAME = "legal-court-rag-index-v2"
SEARCH_ENDPOINT = "https://cpr-rag.search.windows.net"
BATCH_SIZE = 100  # Azure Search batch upload limit is 1000, use 100 for safety


def get_existing_index_schema(index_client: SearchIndexClient) -> SearchIndex:
    """Get the schema of the existing index."""
    logger.info(f"Fetching schema for index: {OLD_INDEX_NAME}")
    return index_client.get_index(OLD_INDEX_NAME)


def create_new_index_with_filterable_id(index_client: SearchIndexClient, old_index: SearchIndex) -> None:
    """Create new index with same schema but filterable 'id' field."""
    logger.info(f"Creating new index: {NEW_INDEX_NAME}")
    
    # Clone all fields from old index
    new_fields = []
    for field in old_index.fields:
        if field.name == 'id':
            # Make id filterable (breaking change)
            new_field = SearchField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,  # This is the critical change
                sortable=True,
                facetable=False,
                searchable=False,
            )
            new_fields.append(new_field)
            logger.info("✅ Created 'id' field with filterable=True")
        else:
            # Copy field as-is
            new_fields.append(field)
    
    # Create new index with same configuration
    new_index = SearchIndex(
        name=NEW_INDEX_NAME,
        fields=new_fields,
        vector_search=old_index.vector_search,
        semantic_search=old_index.semantic_search,
        scoring_profiles=old_index.scoring_profiles,
        cors_options=old_index.cors_options,
        suggesters=old_index.suggesters,
        analyzers=old_index.analyzers,
        tokenizers=old_index.tokenizers,
        token_filters=old_index.token_filters,
        char_filters=old_index.char_filters,
    )
    
    # Check if new index already exists
    try:
        existing_index = index_client.get_index(NEW_INDEX_NAME)
        logger.warning(f"Index {NEW_INDEX_NAME} already exists!")
        response = input("Delete and recreate? (yes/no): ")
        if response.lower() == 'yes':
            index_client.delete_index(NEW_INDEX_NAME)
            logger.info(f"Deleted existing index: {NEW_INDEX_NAME}")
        else:
            logger.info("Using existing index")
            return
    except ResourceNotFoundError:
        pass
    
    # Create the new index
    index_client.create_index(new_index)
    logger.info(f"✅ Created new index: {NEW_INDEX_NAME}")


def copy_documents_batch(
    old_search_client: SearchClient,
    new_search_client: SearchClient,
    batch_size: int = BATCH_SIZE
) -> tuple[int, int]:
    """Copy all documents from old to new index in batches."""
    logger.info(f"Starting document migration from {OLD_INDEX_NAME} to {NEW_INDEX_NAME}")
    
    # Get all documents from old index
    results = old_search_client.search(
        search_text="*",
        include_total_count=True,
        top=batch_size
    )
    
    total_count = results.get_count()
    logger.info(f"Total documents to migrate: {total_count}")
    
    migrated_count = 0
    error_count = 0
    batch_num = 0
    
    # Fetch all documents (paginated)
    all_documents = []
    skip = 0
    while skip < total_count:
        batch_results = old_search_client.search(
            search_text="*",
            top=batch_size,
            skip=skip
        )
        
        batch_docs = []
        for doc in batch_results:
            # Convert document to dict and upload
            doc_dict = {k: v for k, v in doc.items() if v is not None}
            batch_docs.append(doc_dict)
        
        all_documents.extend(batch_docs)
        skip += batch_size
        logger.info(f"Fetched {len(all_documents)}/{total_count} documents...")
    
    # Upload in batches
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i + batch_size]
        batch_num += 1
        
        try:
            result = new_search_client.upload_documents(documents=batch)
            
            # Count successes and failures
            for item in result:
                if item.succeeded:
                    migrated_count += 1
                else:
                    error_count += 1
                    logger.error(f"Failed to upload document {item.key}: {item.error_message}")
            
            logger.info(f"Batch {batch_num}: Uploaded {len([r for r in result if r.succeeded])}/{len(batch)} documents")
            
        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {e}")
            error_count += len(batch)
    
    return migrated_count, error_count


def validate_migration(
    old_search_client: SearchClient,
    new_search_client: SearchClient
) -> bool:
    """Validate that all documents were migrated correctly."""
    logger.info("Validating migration...")
    
    # Check document counts
    old_results = old_search_client.search(search_text="*", include_total_count=True, top=1)
    new_results = new_search_client.search(search_text="*", include_total_count=True, top=1)
    
    old_count = old_results.get_count()
    new_count = new_results.get_count()
    
    logger.info(f"Old index: {old_count} documents")
    logger.info(f"New index: {new_count} documents")
    
    if old_count != new_count:
        logger.error(f"❌ Document count mismatch! Old: {old_count}, New: {new_count}")
        return False
    
    logger.info(f"✅ Document counts match: {new_count}")
    
    # Test filterable id field
    logger.info("Testing filterable 'id' field...")
    
    # Get a sample document
    sample = list(old_search_client.search(search_text="*", top=1))[0]
    sample_id = sample['id']
    
    try:
        # Try to filter by id in new index
        filter_results = list(new_search_client.search(
            search_text="*",
            filter=f"id eq '{sample_id}'",
            top=1
        ))
        
        if len(filter_results) == 1 and filter_results[0]['id'] == sample_id:
            logger.info(f"✅ Successfully filtered by id: {sample_id}")
        else:
            logger.error(f"❌ Filter returned unexpected results")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to filter by id: {e}")
        return False
    
    return True


def main():
    """Main migration workflow."""
    print("=" * 80)
    print("AZURE SEARCH INDEX MIGRATION")
    print("=" * 80)
    print(f"\nOld Index: {OLD_INDEX_NAME}")
    print(f"New Index: {NEW_INDEX_NAME}")
    print(f"Endpoint: {SEARCH_ENDPOINT}")
    print("\nThis script will:")
    print("  1. Create new index with filterable 'id' field")
    print("  2. Copy all documents from old to new index")
    print("  3. Validate the migration")
    print("\n⚠️  WARNING: This is a one-way migration. Old index will remain unchanged.")
    print("=" * 80)
    
    response = input("\nProceed with migration? (yes/no): ")
    if response.lower() != 'yes':
        logger.info("Migration cancelled")
        return
    
    credential = AzureCliCredential()
    
    index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=credential)
    
    # Step 1: Get existing schema
    old_index = get_existing_index_schema(index_client)
    
    # Step 2: Create new index
    create_new_index_with_filterable_id(index_client, old_index)
    
    # Give Azure Search time to finalize index creation
    logger.info("Waiting for index to be ready...")
    time.sleep(5)
    
    index_client.close()
    
    # Step 3: Copy documents
    old_search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=OLD_INDEX_NAME,
        credential=credential
    )
    
    new_search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=NEW_INDEX_NAME,
        credential=credential
    )
    
    migrated, errors = copy_documents_batch(old_search_client, new_search_client)
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Migration complete!")
    logger.info(f"  Migrated: {migrated} documents")
    logger.info(f"  Errors: {errors} documents")
    logger.info(f"{'=' * 80}")
    
    # Step 4: Validate
    if validate_migration(old_search_client, new_search_client):
        logger.info("\n✅ MIGRATION SUCCESSFUL!")
        logger.info("\nNext steps:")
        logger.info(f"  1. Update application config to use: {NEW_INDEX_NAME}")
        logger.info(f"  2. Update GitHub workflow secret AZURE_SEARCH_INDEX to: {NEW_INDEX_NAME}")
        logger.info(f"  3. Test the application with new index")
        logger.info(f"  4. Once confirmed working, delete old index: {OLD_INDEX_NAME}")
        logger.info("\nRollback instructions:")
        logger.info(f"  - Change config back to: {OLD_INDEX_NAME}")
        logger.info(f"  - Delete new index if needed: az search index delete --name {NEW_INDEX_NAME}")
    else:
        logger.error("\n❌ MIGRATION VALIDATION FAILED!")
        logger.error(f"Please review errors and consider deleting new index: {NEW_INDEX_NAME}")
    
    # Close clients
    old_search_client.close()
    new_search_client.close()


if __name__ == "__main__":
    main()
