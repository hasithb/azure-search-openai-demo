#!/usr/bin/env python3
"""
Test highlight accuracy for v2 index.

Checks whether the subsection referenced by the citation target (subsection_id)
can be located in the chunk content using the same matching logic as the
frontend subsection highlighter.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Add app/backend to path for customizations
ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "app" / "backend"
import sys
sys.path.insert(0, str(BACKEND_DIR))

from customizations.subsection_extractor import SubsectionExtractor

console = Console()


def load_azd_env() -> dict[str, str]:
    from subprocess import run

    try:
        result = run(["azd", "env", "get-values"], capture_output=True, text=True, check=True)
        env = {}
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, _, value = line.partition("=")
                env[key] = value.strip('"')
        return env
    except Exception as e:
        console.print(f"[red]Failed to load azd env: {e}[/red]")
        return {}


async def get_search_client():
    from azure.identity.aio import DefaultAzureCredential
    from azure.search.documents.aio import SearchClient

    env = load_azd_env()
    search_service = env.get("AZURE_SEARCH_SERVICE", "cpr-rag")
    search_index = env.get("AZURE_SEARCH_INDEX", "legal-court-rag-index-v2")

    endpoint = f"https://{search_service}.search.windows.net"
    credential = DefaultAzureCredential()

    return SearchClient(endpoint=endpoint, index_name=search_index, credential=credential), credential


async def retrieve_all_documents(max_docs: int) -> list[dict]:
    console.print("[blue]Connecting to Azure Search...[/blue]")

    async def run_query(select_fields: list[str]) -> list[dict]:
        collected: list[dict] = []
        search_client, credential = await get_search_client()
        try:
            async with search_client:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Retrieving documents...", total=max_docs)
                    results = await search_client.search(search_text="*", select=select_fields, top=max_docs)
                    async for result in results:
                        collected.append(
                            {
                                "id": result.get("id", ""),
                                "content": result.get("content", ""),
                                "sourcepage": result.get("sourcepage", ""),
                                "sourcefile": result.get("sourcefile", ""),
                                "category": result.get("category", ""),
                                "subsection_id": result.get("subsection_id", ""),
                            }
                        )
                        progress.update(task, completed=len(collected))
                        if len(collected) >= max_docs:
                            break
        finally:
            await credential.close()
        return collected

    try:
        documents = await run_query(["id", "content", "sourcepage", "sourcefile", "category", "subsection_id"])
    except Exception as e:
        console.print(f"[yellow]Select with subsection_id failed, retrying without it: {e}[/yellow]")
        documents = await run_query(["id", "content", "sourcepage", "sourcefile", "category"])

    console.print(f"[green]Retrieved {len(documents)} documents[/green]")
    return documents


def escape_regexp(value: str) -> str:
    parts = [re.escape(part) for part in re.split(r"\s+", value.strip()) if part]
    if not parts:
        return ""
    return "[\\s_]+".join(parts)


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


@dataclass
class Summary:
    total_docs: int = 0
    docs_with_subsection_id: int = 0
    highlightable: int = 0
    missing_in_content: int = 0
    missing_no_subsection: int = 0
    excluded_docs: int = 0


def is_excluded_doc(doc: dict) -> bool:
    doc_id = (doc.get("id") or "").lower()
    content = (doc.get("content") or "").lower()
    if doc_id.startswith("file-fresh_vs_index_verification_json-"):
        return True
    if "fresh_vs_index_verification.json" in content:
        return True
    return False


async def main() -> int:
    parser = argparse.ArgumentParser(description="v2 highlight accuracy test")
    parser.add_argument("--max-docs", type=int, default=1000000, help="Maximum documents to test")
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "evals" / "results" / "v2_highlight_accuracy.json"),
        help="Output report file",
    )
    args = parser.parse_args()

    documents = await retrieve_all_documents(args.max_docs)
    if not documents:
        console.print("[red]No documents retrieved. Check Azure Search connection.[/red]")
        return 1

    summary = Summary()
    failures = []
    missing_without_subsection = []
    by_sourcefile = Counter()
    by_category = Counter()

    for doc in documents:
        summary.total_docs += 1
        if is_excluded_doc(doc):
            summary.excluded_docs += 1
            continue
        content = (doc.get("content") or "").strip()
        subsection_id = (doc.get("subsection_id") or "").strip()
        if not subsection_id:
            subsection_id = SubsectionExtractor.extract_first_subsection(content)

        if subsection_id:
            summary.docs_with_subsection_id += 1
            if has_subsection_in_content(content, subsection_id):
                summary.highlightable += 1
            else:
                summary.missing_in_content += 1
                failures.append(
                    {
                        "doc_id": doc.get("id", ""),
                        "sourcefile": doc.get("sourcefile", ""),
                        "sourcepage": doc.get("sourcepage", ""),
                        "category": doc.get("category", ""),
                        "subsection_id": subsection_id,
                        "content_preview": content[:240].replace("\n", " "),
                    }
                )
                by_sourcefile[doc.get("sourcefile", "")] += 1
                by_category[doc.get("category", "")] += 1
        else:
            summary.missing_no_subsection += 1
            missing_without_subsection.append(
                {
                    "doc_id": doc.get("id", ""),
                    "sourcefile": doc.get("sourcefile", ""),
                    "sourcepage": doc.get("sourcepage", ""),
                    "category": doc.get("category", ""),
                    "subsection_id": "",
                    "content_preview": content[:240].replace("\n", " "),
                }
            )

    report = {
        "summary": asdict(summary),
        "by_sourcefile": by_sourcefile.most_common(50),
        "by_category": by_category.most_common(50),
        "missing_with_subsection": failures[:200],
        "missing_without_subsection": missing_without_subsection[:200],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    console.print("\n[bold]Summary[/bold]")
    console.print(f"Total docs: {summary.total_docs}")
    console.print(f"Docs with subsection_id: {summary.docs_with_subsection_id}")
    console.print(f"Highlightable: {summary.highlightable}")
    console.print(f"Missing in content: {summary.missing_in_content}")
    console.print(f"Missing no subsection: {summary.missing_no_subsection}")
    console.print(f"Excluded docs: {summary.excluded_docs}")
    console.print(f"Report saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
