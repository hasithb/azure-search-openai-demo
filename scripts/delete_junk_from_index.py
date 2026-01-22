import json
from pathlib import Path
from typing import Iterable
import subprocess

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient


def load_azd_env() -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        result = subprocess.run(["azd", "env", "get-values"], capture_output=True, text=True, check=True)
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, _, value = line.partition("=")
                env[key] = value.strip('"')
    except Exception:
        pass
    return env


def get_search_client() -> SearchClient:
    env = load_azd_env()
    search_service = env.get("AZURE_SEARCH_SERVICE", "cpr-rag")
    search_index = env.get("AZURE_SEARCH_INDEX", "legal-court-rag-index-v2")
    endpoint = f"https://{search_service}.search.windows.net"
    credential = DefaultAzureCredential()
    return SearchClient(endpoint=endpoint, index_name=search_index, credential=credential)


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:
    junk_path = Path("evals/results/v2_highlight_accuracy_junks_full.json")
    junk = json.loads(junk_path.read_text())

    missing_docs = junk.get("missing_without_subsection", [])
    excluded_docs = junk.get("excluded_docs", [])

    ids = [doc.get("id") for doc in missing_docs if doc.get("id")]
    ids.extend([doc.get("id") for doc in excluded_docs if doc.get("id")])
    ids = list(dict.fromkeys(ids))

    if not ids:
        print("No junk doc ids found.")
        return

    client = get_search_client()

    deleted = 0
    for batch in chunked(ids, 1000):
        payload = [{"id": doc_id} for doc_id in batch]
        result = client.delete_documents(documents=payload)
        deleted += len(batch)
        print(f"Deleted batch: {len(batch)}")

    print(f"Deleted total: {deleted}")


if __name__ == "__main__":
    main()
