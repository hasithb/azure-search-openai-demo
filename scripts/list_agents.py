#!/usr/bin/env python3
"""List all agents/knowledge bases in Azure Search."""
import asyncio
import aiohttp
from azure.identity import AzureDeveloperCliCredential
from azure.search.documents.indexes.aio import SearchIndexClient

TENANT_ID = "3bfe16b2-5fcc-4565-b1f1-15271d20fecf"
ENDPOINT = "https://srch-p3xd47iwljsue.search.windows.net"

async def main():
    cred = AzureDeveloperCliCredential(tenant_id=TENANT_ID)
    
    # List indexes
    client = SearchIndexClient(ENDPOINT, cred)
    print("=== Indexes ===")
    try:
        async for idx in client.list_indexes():
            print(f"  Index: {idx.name}")
    finally:
        await client.close()
    
    # List agents via REST API
    print("\n=== Agents/Knowledge Bases ===")
    token = cred.get_token("https://search.azure.com/.default")
    headers = {"Authorization": f"Bearer {token.token}"}
    
    async with aiohttp.ClientSession() as session:
        url = f"{ENDPOINT}/agents?api-version=2025-05-01-preview"
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                for agent in data.get("value", []):
                    name = agent.get("name")
                    kb = agent.get("knowledgeBase", {})
                    sources = kb.get("knowledgeSources", [])
                    source_names = [s.get("name", "?") for s in sources]
                    print(f"  Agent: {name}")
                    print(f"    Knowledge Sources: {source_names}")
            else:
                text = await resp.text()
                print(f"  Error {resp.status}: {text[:200]}")

if __name__ == "__main__":
    asyncio.run(main())
