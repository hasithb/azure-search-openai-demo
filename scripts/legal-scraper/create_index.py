#!/usr/bin/env python
"""
Create Azure Search index for legal documents.
"""
import os
import sys
import logging
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    HnswParameters,
    SemanticSearch,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField
)

# Add scripts to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_index():
    endpoint = Config.AZURE_SEARCH_SERVICE
    if not endpoint.startswith("https://"):
        endpoint = f"https://{endpoint}.search.windows.net"

    credential = DefaultAzureCredential()
    if Config.AZURE_SEARCH_KEY:
        credential = AzureKeyCredential(Config.AZURE_SEARCH_KEY)

    client = SearchIndexClient(endpoint=endpoint, credential=credential)
    index_name = Config.AZURE_SEARCH_INDEX

    logger.info(f"Creating index {index_name} at {endpoint}...")

    # Define fields
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=Config.EMBEDDING_DIMENSIONS,
            vector_search_profile_name="embedding-profile"
        ),
        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="sourcepage", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="sourcefile", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="storageUrl", type=SearchFieldDataType.String, filterable=True),
        # Access control fields
        SimpleField(name="oids", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
        SimpleField(name="groups", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
        # Parent ID for hierarchy
        SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
        # Subsection fields for accurate citation navigation
        SimpleField(name="subsection_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="subsections", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
        # Timestamp for differential upload tracking
        SimpleField(name="updated", type=SearchFieldDataType.String, filterable=True, sortable=True)
    ]

    # Vector search configuration
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="embedding-profile",
                algorithm_configuration_name="hnsw-config"
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters=HnswParameters(
                    metric="cosine" 
                )
            )
        ]
    )

    # Semantic search configuration
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="default",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="sourcepage"),
                    content_fields=[SemanticField(field_name="content")]
                )
            )
        ]
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search
    )

    result = client.create_or_update_index(index)
    logger.info(f"Index {result.name} created successfully.")

if __name__ == "__main__":
    create_index()
