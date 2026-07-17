"""Run curated retrieval behavior checks against a v4 candidate."""

from __future__ import annotations

from typing import Any

import httpx

try:
    from .gate_common import gate_parser, passing_report, post_chat, response_answer, response_sources, run_gate
except ImportError:
    from gate_common import gate_parser, passing_report, post_chat, response_answer, response_sources, run_gate

CASES = (
    ("cpr_part_31", "What is the standard disclosure process under CPR Part 31?", ("part 31", "disclosure")),
    ("cpr_part_52", "What are the time limits for filing an appeal under CPR Part 52?", ("part 52", "appeal")),
    ("commercial_court", "How does the Commercial Court handle case management conferences?", ("commercial court", "case management")),
    ("pre_action", "Can I obtain pre-action disclosure under CPR 31.16?", ("31.16", "pre-action disclosure")),
)


async def run(candidate: str, provenance: dict[str, str]) -> dict[str, Any]:
    checks = []
    async with httpx.AsyncClient(timeout=90) as client:
        for case_id, question, expected_terms in CASES:
            result = await post_chat(client, candidate, question, top=5)
            answer = response_answer(result).lower()
            sources = response_sources(result)
            if not any(term in answer for term in expected_terms):
                raise ValueError(f"{case_id}: answer lacks an expected legal term")
            checks.append({"id": case_id, "source_count": len(sources), "status": "PASS"})
    return passing_report("retrieval", checks, provenance=provenance)


def main() -> int:
    args = gate_parser(__doc__).parse_args()
    return run_gate("retrieval", args.output, run, args.candidate_url, args.provenance)


if __name__ == "__main__":
    raise SystemExit(main())
