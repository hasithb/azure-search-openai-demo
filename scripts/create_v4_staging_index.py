"""Create the Search schema for an immutable v4 staging index."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.upload_v4_staging import (
    EMBEDDING_DIMENSIONS,
    PRODUCTION_INDEX,
    validate_staging_target,
)

MAX_INDEX_COUNT = 50


def build_index(index_name: str):
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        HnswParameters,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SearchIndexPermissionFilterOption,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="embedding-profile",
        ),
        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="sourcepage", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="sourcefile", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="storageUrl", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="oids", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
        SimpleField(name="groups", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
        SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="subsection_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="subsections", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
        SimpleField(name="child_window", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="child_window_count", type=SearchFieldDataType.Int32, filterable=True),
        SearchableField(name="section_title", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        SearchableField(name="hierarchy_path", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        SearchField(
            name="legal_references",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            analyzer_name="standard.lucene",
        ),
        SearchableField(name="embedding_text", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        SimpleField(name="updated", type=SearchFieldDataType.String, filterable=True, sortable=True),
    ]
    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="embedding-profile", algorithm_configuration_name="hnsw-config")],
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-config", parameters=HnswParameters(metric="cosine"))],
    )
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="default",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="section_title"),
                    content_fields=[SemanticField(field_name="hierarchy_path"), SemanticField(field_name="embedding_text")],
                ),
            )
        ]
    )
    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
        permission_filter_option=SearchIndexPermissionFilterOption.ENABLED,
    )


def check_capacity(client, index_name: str, max_indexes: int = MAX_INDEX_COUNT) -> int:
    index_names = list(client.list_index_names())
    if index_name in index_names:
        raise ValueError(f"Staging index already exists: {index_name}")
    if len(index_names) >= max_indexes:
        protected = f"{PRODUCTION_INDEX} and rollback indexes"
        raise ValueError(
            f"Azure AI Search index quota exhausted: {len(index_names)}/{max_indexes} indexes exist; "
            f"free a disposable staging index before creating {index_name}. Do not delete {protected}."
        )
    return len(index_names)


def get_client(service: str):
    from azure.identity import DefaultAzureCredential
    from azure.search.documents.indexes import SearchIndexClient

    endpoint = service if service.startswith("https://") else f"https://{service}.search.windows.net"
    return SearchIndexClient(endpoint=endpoint, credential=DefaultAzureCredential())


def provision(index_name: str, service: str) -> None:
    client = get_client(service)
    check_capacity(client, index_name)
    result = client.create_index(build_index(index_name))
    print(json.dumps({"index": result.name, "status": "created"}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--service", default=os.environ.get("AZURE_SEARCH_SERVICE", ""))
    parser.add_argument("--execute", action="store_true", help="Create the index; default is validation-only")
    parser.add_argument("--check-capacity", action="store_true", help="Check index quota without creating an index")
    args = parser.parse_args()
    validate_staging_target(args.index)
    if args.index.casefold() == PRODUCTION_INDEX.casefold():
        raise ValueError("Refusing to provision the production index")
    if args.execute or args.check_capacity:
        if not args.service:
            raise ValueError("--service or AZURE_SEARCH_SERVICE is required with --execute or --check-capacity")
        client = get_client(args.service)
        index_count = check_capacity(client, args.index)
        if args.execute:
            result = client.create_index(build_index(args.index))
            print(json.dumps({"index": result.name, "status": "created", "index_count_before": index_count}))
        else:
            print(json.dumps({"index": args.index, "status": "capacity_available", "index_count": index_count}))
    else:
        print(json.dumps({"index": args.index, "status": "validated", "execute": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())