import json
from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
import subprocess


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


def fetch_doc(client: SearchClient, doc_id: str) -> dict[str, Any]:
    return client.get_document(key=doc_id)


def collect_excluded_doc_ids(client: SearchClient, max_docs: int = 2000) -> list[str]:
    excluded_ids: list[str] = []
    results = client.search(search_text="fresh_vs_index_verification.json", select=["id"], top=max_docs)
    for result in results:
        doc_id = result.get("id")
        if doc_id and str(doc_id).lower().startswith("file-fresh_vs_index_verification_json-"):
            excluded_ids.append(doc_id)
    return excluded_ids


def main() -> None:
    report_path = Path("evals/results/v2_highlight_accuracy.json")
    report = json.loads(report_path.read_text())
    missing = report.get("missing_without_subsection", [])
    missing_ids = [item.get("doc_id") for item in missing if item.get("doc_id")]

    client = get_search_client()

    excluded_ids = collect_excluded_doc_ids(client)

    missing_docs = [fetch_doc(client, doc_id) for doc_id in missing_ids]
    excluded_docs = [fetch_doc(client, doc_id) for doc_id in excluded_ids]

    output = {
        "missing_without_subsection": missing_docs,
        "excluded_docs": excluded_docs,
    }

    output_path = Path("evals/results/v2_highlight_accuracy_junks_full.json")
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")
    print(f"Missing without subsection: {len(missing_docs)}")
    print(f"Excluded docs: {len(excluded_docs)}")


if __name__ == "__main__":
    main()
