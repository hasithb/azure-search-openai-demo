#!/usr/bin/env python3
"""Get details of specific agent/knowledge base in cpr-rag search service."""
import asyncio
import aiohttp
import json
from azure.identity import AzureDeveloperCliCredential

ENDPOINT = "https://cpr-rag.search.windows.net"
TENANT_ID = "3bfe16b2-5fcc-4565-b1f1-15271d20fecf"
AGENT_NAME = "legal-court-rag-index-agent"

async def main():
    cred = AzureDeveloperCliCredential(tenant_id=TENANT_ID)
    token = cred.get_token("https://search.azure.com/.default")
    headers = {"Authorization": f"Bearer {token.token}"}
    
    async with aiohttp.ClientSession() as session:
        # Get specific agent
        print(f"=== Agent: {AGENT_NAME} ===")
        url = f"{ENDPOINT}/agents/{AGENT_NAME}?api-version=2025-08-01-preview"
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(json.dumps(data, indent=2))
            else:
                text = await resp.text()
                print(f"Error {resp.status}: {text}")

if __name__ == "__main__":
    asyncio.run(main())
