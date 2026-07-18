"""Shared helpers for fail-closed v4 application gates."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from application_gate import validate_candidate_url


class GateFailure(ValueError):
    """Raised when a candidate fails an expected application behavior check."""


def gate_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def passing_report(
    gate: str,
    checks: list[dict[str, Any]],
    citation_targets: dict[str, Any] | None = None,
    *,
    provenance: dict[str, str],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "gate": gate,
        "provenance": provenance,
        "checks": checks,
    }
    if citation_targets:
        report.update(citation_targets)
    return report


async def post_chat(
    client: httpx.AsyncClient,
    candidate: str,
    question: str,
    *,
    category: str = "",
    top: int = 5,
) -> dict[str, Any]:
    candidate_url = validate_candidate_url(candidate)
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": question}],
        "overrides": {"top": top},
    }
    if category:
        payload["overrides"]["include_category"] = category
    response = await client.post(f"{candidate_url}/chat", json=payload)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise GateFailure("Candidate chat response must be a JSON object")
    return result


def response_answer(result: dict[str, Any]) -> str:
    answer = result.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    raise GateFailure("Candidate chat response has no usable answer")


def response_sources(result: dict[str, Any]) -> list[dict[str, Any]]:
    sources = result.get("sources", [])
    if not isinstance(sources, list) or any(not isinstance(source, dict) for source in sources):
        raise GateFailure("Candidate chat response has invalid sources")
    return sources


def run_gate(
    gate: str,
    output: Path,
    runner: Callable[..., Awaitable[dict[str, Any]]],
    candidate_url: str,
    provenance_path: Path,
) -> int:
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(provenance, dict):
            raise GateFailure("Gate provenance must be a JSON object")
        report = asyncio.run(runner(validate_candidate_url(candidate_url), provenance))
        if not isinstance(report, dict) or report.get("status") != "PASS":
            raise GateFailure(f"{gate} gate did not produce a passing report")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as temporary:
            temporary.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(output)
        print(json.dumps({"gate": gate, "output": str(output), "status": "PASS"}, sort_keys=True))
        return 0
    except (OSError, ValueError, httpx.HTTPError) as error:
        print(json.dumps({"gate": gate, "status": "FAIL", "error": str(error)}, sort_keys=True))
        return 1