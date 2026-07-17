"""Prove citation navigation and subsection highlighting in a live candidate UI."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

try:
    from .validate_highlight_oracle import validate as validate_oracle
except ImportError:
    from validate_highlight_oracle import validate as validate_oracle


class BrowserGateError(ValueError):
    """Raised when the live browser behavior is not proven."""


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def choose_case(oracle: dict[str, Any]) -> dict[str, Any]:
    cases = [case for case in oracle.get("cases", []) if isinstance(case, dict)]
    if not cases:
        raise BrowserGateError("Highlight oracle contains no browser-checkable cases")
    preferred = [
        case
        for case in cases
        if str(case.get("subsection_id") or "").strip() == "24.2"
        and "part 24" in str(case.get("sourcepage") or "").casefold()
    ]
    return preferred[0] if preferred else min(cases, key=lambda case: len(str(case.get("body_text") or "")))


def run_browser_gate(candidate_url: str, oracle: dict[str, Any], question: str) -> dict[str, Any]:
    target_case = choose_case(oracle)
    expected_body = normalize(str(target_case.get("body_text") or ""))
    expected_heading = normalize(str(target_case.get("expected_heading") or ""))
    next_heading = normalize(str(target_case.get("next_heading") or ""))
    if not expected_body or not expected_heading:
        raise BrowserGateError("Selected highlight oracle case is incomplete")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(candidate_url, wait_until="domcontentloaded", timeout=60_000)
            splash = page.locator("[role='dialog'][aria-modal='true']")
            try:
                splash.wait_for(state="visible", timeout=2_000)
                splash.click()
                splash.wait_for(state="hidden", timeout=5_000)
            except PlaywrightTimeoutError:
                pass

            question_input = page.get_by_placeholder(re.compile(r"Ask a question|Type a new question"))
            question_input.wait_for(state="visible", timeout=30_000)
            question_input.fill(question)
            page.get_by_role("button", name="Submit question").click()

            subsection_selector = f".supContainer[data-subsection-id={json.dumps(str(target_case['subsection_id']))}]"
            citations = page.locator(subsection_selector)
            citations.first.wait_for(state="visible", timeout=120_000)
            citation_count = citations.count()
            citations.first.click()
            page.get_by_text("Supporting content", exact=False).first.wait_for(state="visible", timeout=30_000)

            highlight = page.locator("#highlighted-subsection")
            highlight.wait_for(state="visible", timeout=30_000)
            highlighted_text = normalize(highlight.inner_text())
            if not highlighted_text:
                raise BrowserGateError("Supporting Content rendered an empty highlighted subsection")
            if expected_heading not in highlighted_text and str(target_case["subsection_id"]).casefold() not in highlighted_text:
                raise BrowserGateError("Highlighted subsection does not identify the canonical target heading")
            if expected_body not in highlighted_text and highlighted_text not in expected_body:
                raise BrowserGateError("Highlighted subsection text does not match canonical oracle evidence")
            if next_heading and next_heading in highlighted_text:
                raise BrowserGateError("Highlighted subsection includes the next canonical subsection")

            citation_path = citations.first.get_attribute("data-citation-path") or ""
            return {
                "browser": {
                    "candidate_url": candidate_url,
                    "question": question,
                    "citation_count": citation_count,
                    "clicked_selector": subsection_selector,
                    "supporting_content_visible": True,
                    "highlight_visible": True,
                    "highlighted_text_sha256": hashlib.sha256(highlighted_text.encode("utf-8")).hexdigest(),
                    "citation_path_present": bool(citation_path),
                },
                "case_id": target_case["case_id"],
                "subsection_id": target_case["subsection_id"],
            }
        finally:
            browser.close()


def build_report(candidate_url: str, oracle_path: Path, snapshot_dir: Path, provenance: dict[str, str], question: str) -> dict[str, Any]:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    validated_oracle = validate_oracle(oracle, snapshot_dir, provenance=None)
    browser_evidence = run_browser_gate(candidate_url, oracle, question)
    return {
        "schema_version": 1,
        "gate": "highlight",
        "status": "PASS",
        "oracle_version": validated_oracle["oracle_version"],
        "case_count": validated_oracle["case_count"],
        "source_count": validated_oracle["source_count"],
        "snapshot_manifest_sha256": validated_oracle["snapshot_manifest_sha256"],
        "browser_evidence": browser_evidence,
        "provenance": provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--question", default="What is CPR Part 24 rule 24.2 and the test for summary judgment?")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        report = build_report(args.candidate_url, args.oracle, args.snapshot_dir, provenance, args.question)
    except (OSError, json.JSONDecodeError, BrowserGateError, PlaywrightTimeoutError, ValueError) as error:
        report = {"schema_version": 1, "gate": "highlight", "status": "FAIL", "error": str(error)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as temporary:
        temporary.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())