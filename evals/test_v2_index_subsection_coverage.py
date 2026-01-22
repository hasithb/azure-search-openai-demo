#!/usr/bin/env python3
"""
Test subsection coverage for ALL documents in the v2 Azure Search index.

Compares:
- SubsectionExtractor.extract_all_subsections(content)
vs
- Regex sweep of all subsection-like tokens in content.

Outputs a summary and JSON report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter
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

TOKEN_PATTERN = re.compile(
    r"\b("
    r"[A-Z]\.\d+(?:\.\d+)?[A-Z]?"  # A.1, B.7, C.2.3, 5.1A
    r"|[A-Z]\d+\.\d+(?:\.\d+)?[A-Z]?"  # A4.1, B2.3, A1.1A
    r"|\d+\.\d+(?:\.\d+)?[A-Z]?"  # 1.1, 2.3, 1.2.3, 7.3A
    r"|Rule\s+\d+(?:\.\d+)?"  # Rule 3.1
    r"|Para\s+\d+(?:\.\d+)?"  # Para 5.2
    r")\b",
    re.IGNORECASE,
)


@dataclass
class Summary:
    total_docs: int = 0
    total_tokens: int = 0
    total_extracted: int = 0
    missing_tokens: int = 0
    extra_tokens: int = 0
    docs_with_missing: int = 0
    docs_with_extra: int = 0


def normalize_token(token: str) -> str:
    normalized = token.replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized.strip())
    return normalized.rstrip(".: ").upper()


def build_equivalent_set(tokens: list[str]) -> set[str]:
    result = set(tokens)
    for token in list(result):
        match = re.match(r"^(RULE|PARA)\s+(\d+(?:\.\d+)?[A-Z]?)$", token, re.IGNORECASE)
        if match:
            result.add(match.group(2))
    return result


def extract_expected_tokens(content: str) -> list[str]:
    if not content:
        return []
    tokens = [normalize_token(m.group(0)) for m in TOKEN_PATTERN.finditer(content)]
    # Preserve order but unique
    seen = set()
    ordered = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


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
    try:
        search_client, credential = await get_search_client()
    except Exception as e:
        console.print(f"[red]Failed to create search client: {e}[/red]")
        return []

    documents = []
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
                results = await search_client.search(
                    search_text="*",
                    select=["id", "content", "sourcepage", "sourcefile", "category"],
                    top=max_docs,
                )
                async for result in results:
                    documents.append(
                        {
                            "id": result.get("id", ""),
                            "content": result.get("content", ""),
                            "sourcepage": result.get("sourcepage", ""),
                            "sourcefile": result.get("sourcefile", ""),
                            "category": result.get("category", ""),
                        }
                    )
                    progress.update(task, completed=len(documents))
                    if len(documents) >= max_docs:
                        break
    finally:
        await credential.close()

    console.print(f"[green]Retrieved {len(documents)} documents[/green]")
    return documents


def analyze_documents(documents: list[dict]) -> tuple[Summary, list[dict]]:
    summary = Summary()
    failures: list[dict] = []

    for doc in documents:
        content = (doc.get("content") or "").strip()
        expected = extract_expected_tokens(content)
        extracted = SubsectionExtractor.extract_all_subsections(content)

        expected_norm = [normalize_token(t) for t in expected]
        extracted_norm = [normalize_token(t) for t in extracted]

        expected_set = build_equivalent_set(expected_norm)
        extracted_set = build_equivalent_set(extracted_norm)

        missing = [t for t in expected_set if t not in extracted_set]
        extra = [t for t in extracted_set if t not in expected_set]

        summary.total_docs += 1
        summary.total_tokens += len(expected_norm)
        summary.total_extracted += len(extracted_norm)
        summary.missing_tokens += len(missing)
        summary.extra_tokens += len(extra)
        summary.docs_with_missing += 1 if missing else 0
        summary.docs_with_extra += 1 if extra else 0

        if missing or extra:
            failures.append(
                {
                    "doc_id": doc.get("id", ""),
                    "sourcepage": doc.get("sourcepage", ""),
                    "sourcefile": doc.get("sourcefile", ""),
                    "missing_tokens": missing,
                    "extra_tokens": extra,
                    "content_preview": content[:240].replace("\n", " "),
                }
            )

    return summary, failures


async def main() -> int:
    parser = argparse.ArgumentParser(description="v2 index subsection coverage test")
    parser.add_argument("--max-docs", type=int, default=1000000, help="Maximum documents to test")
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "evals" / "results" / "v2_subsection_coverage.json"),
        help="Output report file",
    )
    args = parser.parse_args()

    documents = await retrieve_all_documents(args.max_docs)
    if not documents:
        console.print("[red]No documents retrieved. Check Azure Search connection.[/red]")
        return 1

    summary, failures = analyze_documents(documents)

    report = {
        "summary": asdict(summary),
        "failures": failures[:200],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    console.print("\n[bold]Summary[/bold]")
    console.print(f"Total docs: {summary.total_docs}")
    console.print(f"Total tokens: {summary.total_tokens}")
    console.print(f"Missing tokens: {summary.missing_tokens}")
    console.print(f"Extra tokens: {summary.extra_tokens}")
    console.print(f"Docs with missing: {summary.docs_with_missing}")
    console.print(f"Docs with extra: {summary.docs_with_extra}")
    console.print(f"Report saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
