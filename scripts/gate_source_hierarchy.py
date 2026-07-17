"""Run source hierarchy checks against a v4 candidate."""

from __future__ import annotations

from typing import Any

import httpx

try:
    from .gate_common import GateFailure, gate_parser, passing_report, post_chat, response_answer, response_sources, run_gate
except ImportError:
    from gate_common import GateFailure, gate_parser, passing_report, post_chat, response_answer, response_sources, run_gate

CASES = (
    ("general_cpr", "What are the rules on case management conferences?", "", "Civil Procedure Rules and Practice Directions"),
    ("commercial_guide", "How does the Commercial Court handle case management conferences?", "", "Commercial Court"),
    ("chancery_filter", "How are case management conferences handled?", "Chancery Division", "Chancery Division"),
)


async def run(candidate: str, provenance: dict[str, str]) -> dict[str, Any]:
    checks = []
    async with httpx.AsyncClient(timeout=90) as client:
        for case_id, question, category, expected_category in CASES:
            result = await post_chat(client, candidate, question, category=category, top=7)
            answer = response_answer(result).lower()
            sources = response_sources(result)
            categories = {str(source.get("category", "")) for source in sources if source.get("category")}
            if expected_category not in categories and expected_category.lower() not in answer:
                raise GateFailure(f"{case_id}: expected {expected_category} in sources or answer")
            if category and any(source.get("category") != category for source in sources):
                raise GateFailure(f"{case_id}: filtered response crossed category boundary")
            checks.append({"id": case_id, "categories": sorted(categories), "status": "PASS"})
    return passing_report("source_hierarchy", checks, provenance=provenance)


def main() -> int:
    args = gate_parser(__doc__).parse_args()
    return run_gate("source_hierarchy", args.output, run, args.candidate_url, args.provenance)


if __name__ == "__main__":
    raise SystemExit(main())
