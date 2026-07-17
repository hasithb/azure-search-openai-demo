"""Run citation-to-source consistency checks against a v4 candidate."""

from __future__ import annotations

import re
from typing import Any

import httpx

try:
    from .gate_common import GateFailure, gate_parser, passing_report, post_chat, response_answer, response_sources, run_gate
except ImportError:
    from gate_common import GateFailure, gate_parser, passing_report, post_chat, response_answer, response_sources, run_gate

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def validate_citation_sources(answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate citation ordinals and preserve their exact structured targets."""
    cited = sorted({int(value) for value in CITATION_PATTERN.findall(answer)})
    if not cited:
        raise GateFailure("Answer contains no numeric citations")

    invalid = [number for number in cited if number < 1 or number > len(sources)]
    if invalid:
        raise GateFailure(f"Citations do not resolve to returned sources: {invalid}")

    missing_identity = [
        number
        for number in cited
        if not str(sources[number - 1].get("sourcefile") or "").strip()
        or not str(sources[number - 1].get("sourcepage") or "").strip()
    ]
    if missing_identity:
        raise GateFailure(f"Cited sources are missing sourcefile/sourcepage identity: {missing_identity}")

    targets = [
        {
            "citation_number": number,
            "sourcefile": str(sources[number - 1]["sourcefile"]).strip(),
            "sourcepage": str(sources[number - 1]["sourcepage"]).strip(),
            "subsection_id": str(sources[number - 1].get("subsection_id") or "").strip(),
        }
        for number in cited
    ]
    return {"cited_source_numbers": cited, "cited_targets": targets}


async def run(candidate: str, provenance: dict[str, str]) -> dict[str, Any]:
    question = "What is CPR Part 1 and the overriding objective?"
    async with httpx.AsyncClient(timeout=90) as client:
        result = await post_chat(client, candidate, question, top=7)
    answer = response_answer(result)
    sources = response_sources(result)
    citation_targets = validate_citation_sources(answer, sources)
    return passing_report(
        "citation",
        [
            {
                "id": "numeric_citations_resolve_to_structured_sources",
                "citation_count": len(citation_targets["cited_source_numbers"]),
                "source_count": len(sources),
                "status": "PASS",
            }
        ],
        citation_targets,
        provenance=provenance,
    )


def main() -> int:
    args = gate_parser(__doc__).parse_args()
    return run_gate("citation", args.output, run, args.candidate_url, args.provenance)


if __name__ == "__main__":
    raise SystemExit(main())
