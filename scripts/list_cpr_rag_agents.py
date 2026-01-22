#!/usr/bin/env python3
"""List agents/knowledge bases and their knowledge sources for cpr-rag search service."""
import asyncio
import aiohttp
from azure.identity import AzureDeveloperCliCredential

ENDPOINT = "https://cpr-rag.search.windows.net"
TENANT_ID = "3bfe16b2-5fcc-4565-b1f1-15271d20fecf"

async def main():
    cred = AzureDeveloperCliCredential(tenant_id=TENANT_ID)
    token = cred.get_token("https://search.azure.com/.default")
    headers = {"Authorization": f"Bearer {token.token}"}
    
    async with aiohttp.ClientSession() as session:
        # List agents
        print("=== Agents/Knowledge Bases ===")
        url = f"{ENDPOINT}/agents?api-version=2025-05-01-preview"
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                for agent in data.get("value", []):
                    name = agent.get("name")
                    kb = agent.get("knowledgeBase", {})
                    sources = kb.get("knowledgeSources", [])
                    print(f"\n  Agent: {name}")
                    for source in sources:
                        src_name = source.get("name", "?")
                        idx_params = source.get("indexSource", {}).get("indexParameters", {})
                        idx_name = idx_params.get("indexName", "?")
                        print(f"    Knowledge Source: {src_name}")
                        print(f"    -> Index Name: {idx_name}")
            else:
                text = await resp.text()
                print(f"  Error {resp.status}: {text[:400]}")

        # List knowledge sources
        print("\n\n=== Knowledge Sources ===")
        url = f"{ENDPOINT}/knowledge-sources?api-version=2025-05-01-preview"
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                for ks in data.get("value", []):
                    name = ks.get("name")
                    idx_params = ks.get("indexSource", {}).get("indexParameters", {})
                    idx_name = idx_params.get("indexName", "?")
                    print(f"\n  Knowledge Source: {name}")
                    print(f"    -> Index Name: {idx_name}")
            else:
                text = await resp.text()
                print(f"  Error {resp.status}: {text[:400]}")

if __name__ == "__main__":
    asyncio.run(main())
