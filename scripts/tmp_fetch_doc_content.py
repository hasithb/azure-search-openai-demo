import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient


async def get_client():
    import subprocess

    env = {}
    try:
        result = subprocess.run(["azd", "env", "get-values"], capture_output=True, text=True, check=True)
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, _, value = line.partition("=")
                env[key] = value.strip('"')
    except Exception:
        pass

    search_service = env.get("AZURE_SEARCH_SERVICE", "cpr-rag")
    search_index = env.get("AZURE_SEARCH_INDEX", "legal-court-rag-index-v2")
    endpoint = f"https://{search_service}.search.windows.net"
    credential = DefaultAzureCredential()
    return SearchClient(endpoint=endpoint, index_name=search_index, credential=credential), credential


async def main(doc_id: str):
    search_client, credential = await get_client()
    try:
        async with search_client:
            result = await search_client.get_document(key=doc_id)
            content = result.get("content", "")
            print(content[:2000])
    finally:
        await credential.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("doc_id")
    args = parser.parse_args()
    asyncio.run(main(args.doc_id))
