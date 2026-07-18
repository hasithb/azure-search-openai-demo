"""
Standalone script to create the Azure AI Search Knowledge Base object.

This does NOT touch the search index data — it only creates the KB metadata
that the agentic retrieval client expects. Safe to run against an existing
custom-ingested index.

Usage:
    # Uses values from .azure/cpr-rag/.env by default
    python scripts/create_knowledgebase.py

    # Or override with env vars:
    AZURE_SEARCH_SERVICE=gptkb-gz2m4s637t5me \
    AZURE_SEARCH_INDEX=legal-court-rag-index-v3 \
    AZURE_SEARCH_KNOWLEDGEBASE_NAME=legal-court-rag-index-v3-agent-upgrade \
    AZURE_OPENAI_SERVICE=cog-gz2m4s637t5me-us2 \
    AZURE_OPENAI_KNOWLEDGEBASE_DEPLOYMENT=knowledgebase \
    AZURE_OPENAI_KNOWLEDGEBASE_MODEL=gpt-4.1-mini \
    python scripts/create_knowledgebase.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Allow importing from app/backend
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "backend"))

from azure.identity.aio import AzureCliCredential, AzureDeveloperCliCredential
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizerParameters,
    KnowledgeBase,
    KnowledgeBaseAzureOpenAIModel,
    KnowledgeRetrievalOutputMode,
    KnowledgeSourceReference,
    SearchIndexFieldReference,
    SearchIndexKnowledgeSource,
    SearchIndexKnowledgeSourceParameters,
)


def load_azd_env() -> dict[str, str]:
    """Load .env from .azure/<env>/ if it exists."""
    root = Path(__file__).resolve().parent.parent
    config_path = root / ".azure" / "config.json"
    if config_path.exists():
        import json

        config = json.loads(config_path.read_text())
        env_name = config.get("defaultEnvironment", "")
        env_file = root / ".azure" / env_name / ".env"
        if env_file.exists():
            result: dict[str, str] = {}
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    result[k.strip()] = v
            return result
    return {}


async def create_knowledgebase():
    # Merge azd env with actual env (actual env wins)
    azd_env = load_azd_env()
    def env(key: str, default: str = "") -> str:
        return os.getenv(key) or azd_env.get(key, default)

    search_service = env("AZURE_SEARCH_SERVICE")
    search_index = env("AZURE_SEARCH_INDEX")
    kb_name = env("AZURE_SEARCH_KNOWLEDGEBASE_NAME")
    openai_service = env("AZURE_OPENAI_SERVICE")
    kb_deployment = env("AZURE_OPENAI_KNOWLEDGEBASE_DEPLOYMENT")
    kb_model = env("AZURE_OPENAI_KNOWLEDGEBASE_MODEL")
    tenant_id = env("AZURE_TENANT_ID")

    if not all([search_service, search_index, kb_name, openai_service, kb_deployment, kb_model]):
        print("Missing required env vars. Need:")
        print(f"  AZURE_SEARCH_SERVICE={search_service or '(missing)'}")
        print(f"  AZURE_SEARCH_INDEX={search_index or '(missing)'}")
        print(f"  AZURE_SEARCH_KNOWLEDGEBASE_NAME={kb_name or '(missing)'}")
        print(f"  AZURE_OPENAI_SERVICE={openai_service or '(missing)'}")
        print(f"  AZURE_OPENAI_KNOWLEDGEBASE_DEPLOYMENT={kb_deployment or '(missing)'}")
        print(f"  AZURE_OPENAI_KNOWLEDGEBASE_MODEL={kb_model or '(missing)'}")
        sys.exit(1)

    search_endpoint = f"https://{search_service}.search.windows.net"
    openai_endpoint = f"https://{openai_service}.openai.azure.com/"

    print(f"Search endpoint: {search_endpoint}")
    print(f"Search index:    {search_index}")
    print(f"KB name:         {kb_name}")
    print(f"OpenAI endpoint: {openai_endpoint}")
    print(f"KB deployment:   {kb_deployment}")
    print(f"KB model:        {kb_model}")
    print()

    if os.getenv("GITHUB_ACTIONS") == "true":
        credential = AzureCliCredential(tenant_id=tenant_id) if tenant_id else AzureCliCredential()
    else:
        credential = AzureDeveloperCliCredential(tenant_id=tenant_id, process_timeout=60) if tenant_id else AzureDeveloperCliCredential(process_timeout=60)

    async with SearchIndexClient(endpoint=search_endpoint, credential=credential) as client:
        # Step 1: Create the knowledge source pointing at the existing index
        field_names = ["id", "sourcepage", "sourcefile", "content", "category"]
        source_data_fields = [SearchIndexFieldReference(name=f) for f in field_names]

        knowledge_source = SearchIndexKnowledgeSource(
            name=search_index,
            description="Default knowledge source using the main search index",
            search_index_parameters=SearchIndexKnowledgeSourceParameters(
                search_index_name=search_index,
                source_data_fields=source_data_fields,
            ),
        )
        print(f"Creating knowledge source '{search_index}'...")
        await client.create_or_update_knowledge_source(knowledge_source=knowledge_source)
        print("  Done.")

        # Step 2: Create the knowledge base referencing that source
        kb = KnowledgeBase(
            name=kb_name,
            knowledge_sources=[KnowledgeSourceReference(name=search_index)],
            models=[
                KnowledgeBaseAzureOpenAIModel(
                    azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                        resource_url=openai_endpoint,
                        deployment_name=kb_deployment,
                        model_name=kb_model,
                    )
                )
            ],
            output_mode=KnowledgeRetrievalOutputMode.ANSWER_SYNTHESIS,
        )
        print(f"Creating knowledge base '{kb_name}'...")
        await client.create_or_update_knowledge_base(knowledge_base=kb)
        print("  Done.")

    print()
    print(f"Knowledge base '{kb_name}' is now active on '{search_service}'.")
    print("The deployed app should work on the next chat request.")


if __name__ == "__main__":
    asyncio.run(create_knowledgebase())
