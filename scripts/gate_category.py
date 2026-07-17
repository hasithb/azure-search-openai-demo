"""Run category discovery and category-filter behavior checks."""

from __future__ import annotations

from typing import Any

import httpx

try:
    from .gate_common import GateFailure, gate_parser, passing_report, post_chat, response_answer, response_sources, run_gate
except ImportError:
    from gate_common import GateFailure, gate_parser, passing_report, post_chat, response_answer, response_sources, run_gate


def validate_category_sources(selected: str, sources: list[dict[str, Any]]) -> None:
    if not sources:
        raise GateFailure(f"Category filter returned no sources: {selected}")
    unexpected = sorted({str(source.get("category") or "") for source in sources if source.get("category") != selected})
    if unexpected:
        raise GateFailure(f"Category filter returned unexpected categories: {', '.join(unexpected)}")


async def run(candidate: str, provenance: dict[str, str]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=90) as client:
        try:
            response = await client.get(f"{candidate}/api/categories")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise GateFailure(f"Category request failed: {error}") from error
        categories = payload.get("categories") if isinstance(payload, dict) else None
        if not isinstance(categories, list) or not categories:
            raise GateFailure("Category response contains no categories")
        values = [item for item in categories if isinstance(item, dict) and item.get("key")]
        if not values:
            raise GateFailure("Category response contains no filterable category")
        filter_checks = []
        for item in values:
            selected = str(item["key"])
            result = await post_chat(
                client,
                candidate,
                "What are the main case management procedures?",
                category=selected,
                top=5,
            )
            response_answer(result)
            sources = response_sources(result)
            validate_category_sources(selected, sources)
            filter_checks.append({"selected": selected, "source_count": len(sources), "status": "PASS"})
    return passing_report(
        "category",
        [
            {"id": "category_discovery", "category_count": len(values), "status": "PASS"},
            {"id": "category_filter", "category_count": len(filter_checks), "checks": filter_checks, "status": "PASS"},
        ],
        provenance=provenance,
    )


def main() -> int:
    args = gate_parser(__doc__).parse_args()
    return run_gate("category", args.output, run, args.candidate_url, args.provenance)


if __name__ == "__main__":
    raise SystemExit(main())
