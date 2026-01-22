import asyncio
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from customizations.subsection_extractor import SubsectionExtractor
from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient


def escape_regexp(value: str) -> str:
    return re.escape(value)


def has_subsection_in_content(content: str, target: str) -> bool:
    if not content or not target:
        return False

    escaped = escape_regexp(target)
    patterns = [
        re.compile(rf"(^|\n)\s*#{{1,6}}\s*{escaped}\s*(\n|\s|$)", re.IGNORECASE),
        re.compile(rf"(^|\n)\s*\[[^\]]*>\s*{escaped}\s*\]\s*(\n|\s|$)", re.IGNORECASE),
        re.compile(rf"(^|\n)\s*{escaped}\s*(\n|\s|$)", re.IGNORECASE),
        re.compile(rf"(^|\n)\s*{escaped}\s*[.:]?\s*(\n|\s|$)", re.IGNORECASE),
        re.compile(rf"(^|\n)\s*{escaped}\s+[A-Za-z]", re.IGNORECASE),
        re.compile(rf"(^|\n)\s*\(?{escaped}\)?\s*[-\s]", re.IGNORECASE),
        re.compile(rf"\b{escaped}\b", re.IGNORECASE),
    ]

    return any(pattern.search(content) for pattern in patterns)


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


async def main():
    max_docs = 1000000
    collected = []
    search_client, credential = await get_client()
    try:
        async with search_client:
            results = await search_client.search(
                search_text="*",
                select=["id", "content", "sourcepage", "sourcefile", "category"],
                top=max_docs,
            )
            async for result in results:
                collected.append(
                    {
                        "id": result.get("id", ""),
                        "content": result.get("content", ""),
                        "sourcepage": result.get("sourcepage", ""),
                        "sourcefile": result.get("sourcefile", ""),
                        "category": result.get("category", ""),
                    }
                )
                if len(collected) >= max_docs:
                    break
    finally:
        await credential.close()

    missing_with_subsection = []
    missing_no_subsection = []
    missing_no_subsection_source = Counter()
    missing_no_subsection_category = Counter()

    for doc in collected:
        content = (doc.get("content") or "").strip()
        subsection_id = SubsectionExtractor.extract_first_subsection(content)
        if subsection_id:
            if not has_subsection_in_content(content, subsection_id):
                missing_with_subsection.append(
                    {
                        **doc,
                        "subsection_id": subsection_id,
                        "preview": content[:240].replace("\n", " "),
                    }
                )
        else:
            missing_no_subsection.append(
                {
                    **doc,
                    "subsection_id": "",
                    "preview": content[:240].replace("\n", " "),
                }
            )
            missing_no_subsection_source[doc.get("sourcefile", "")] += 1
            missing_no_subsection_category[doc.get("category", "")] += 1

    print("docs", len(collected))
    print("missing_with_subsection", len(missing_with_subsection))
    print("missing_no_subsection", len(missing_no_subsection))
    print("top_missing_no_subsection_sourcefile", missing_no_subsection_source.most_common(10))
    print("top_missing_no_subsection_category", missing_no_subsection_category.most_common(10))

    print("\nMissing with subsection:")
    for item in missing_with_subsection[:20]:
        print("---")
        print("id", item["id"])
        print("subsection_id", item["subsection_id"])
        print("source", item.get("sourcefile"))
        print("preview", item["preview"])

    print("\nMissing without subsection:")
    for item in missing_no_subsection[:10]:
        print("---")
        print("id", item["id"])
        print("subsection_id", item["subsection_id"])
        print("source", item.get("sourcefile"))
        print("preview", item["preview"])


if __name__ == "__main__":
    asyncio.run(main())
