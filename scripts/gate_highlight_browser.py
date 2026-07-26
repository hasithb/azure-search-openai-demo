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


def normalize_path(value: str) -> str:
    return normalize(value).rstrip("/")


def normalize_sourcepage(value: str) -> str:
    return normalize(re.sub(r"[\u2010-\u2015\u2212-]", " ", value))


def subsection_matches(expected: str, actual: str) -> bool:
    expected_normalized = normalize(expected)
    actual_normalized = normalize(actual)
    if not expected_normalized or not actual_normalized:
        return False
    if expected_normalized == actual_normalized:
        return True
    if expected_normalized == "part 24":
        return actual_normalized.startswith("24.") or actual_normalized == "24"
    return actual_normalized.startswith(f"{expected_normalized}(")


def citation_matches(target_case: dict[str, Any], citation: dict[str, Any]) -> bool:
    expected_sourcefile = normalize_path(str(target_case.get("sourcefile") or ""))
    expected_path = normalize_path(str(target_case.get("citation_path") or ""))
    actual_sourcefile = normalize_path(str(citation.get("sourcefile") or ""))
    actual_path = normalize_path(str(citation.get("citation_path") or ""))
    source_bound = bool(expected_sourcefile and actual_sourcefile == expected_sourcefile)
    path_bound = bool(expected_path and actual_path == expected_path)
    if not source_bound and not path_bound:
        return False
    if not subsection_matches(
        str(target_case.get("subsection_id") or ""),
        str(citation.get("subsection_id") or ""),
    ):
        return False
    expected_category = normalize(str(target_case.get("category") or ""))
    if expected_category and normalize(str(citation.get("category") or "")) != expected_category:
        return False
    expected_sourcepage = normalize_sourcepage(str(target_case.get("sourcepage") or ""))
    actual_sourcepage = normalize_sourcepage(str(citation.get("sourcepage") or ""))
    return not expected_sourcepage or not actual_sourcepage or expected_sourcepage in actual_sourcepage or actual_sourcepage in expected_sourcepage


def select_citation(target_case: dict[str, Any], citations: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [citation for citation in citations if citation_matches(target_case, citation)]
    if len(matches) != 1:
        raise BrowserGateError(
            f"Expected one canonical citation, found {len(matches)}; candidates={json.dumps(citations, sort_keys=True)}"
        )
    return matches[0]


def validate_highlight_identity(
    highlighted_text: str,
    supporting_card_text: str,
    expected_heading: str,
    expected_subsection: str,
) -> None:
    """Require the heading in the selected card and the subsection in its highlight."""
    normalized_highlight = normalize(highlighted_text)
    normalized_card = normalize(supporting_card_text)
    if not normalized_highlight:
        raise BrowserGateError("Supporting Content rendered an empty highlighted subsection")
    if normalize(expected_heading) not in normalized_card:
        raise BrowserGateError(
            "Supporting Content card does not identify the canonical target heading: "
            f"expected_heading={normalize(expected_heading)!r}, card_text={normalized_card[:500]!r}"
        )
    if normalize(expected_subsection) not in normalized_highlight:
        raise BrowserGateError(
            "Highlighted subsection does not identify the canonical target subsection: "
            f"expected_subsection={expected_subsection!r}, highlighted_text={normalized_highlight[:500]!r}"
        )


def css_attribute_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


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
    if preferred:
        return preferred[0]

    part_cases = [
        case
        for case in cases
        if str(case.get("subsection_id") or "").strip().casefold() == "part 24"
    ]
    if part_cases:
        return part_cases[0]

    return max(cases, key=lambda case: len(str(case.get("body_text") or "")))


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
            source_filter = page.get_by_role("combobox", name="Source filter")
            source_filter.wait_for(state="visible", timeout=30_000)
            source_filter.click()
            page.get_by_role("menuitemcheckbox", name="All Sources", exact=True).click()
            page.keyboard.press("Escape")
            question_input.fill(question)
            page.get_by_role("button", name="Submit question").click()

            citations_locator = page.locator(".supContainer")
            citations_locator.first.wait_for(state="visible", timeout=120_000)
            citations = citations_locator.evaluate_all(
                """
                (elements) => elements
                    .map((element, index) => ({
                        index,
                        visible: !!(element.offsetWidth || element.offsetHeight || element.getClientRects().length),
                        subsection_id: element.getAttribute('data-subsection-id') || '',
                        sourcepage: element.getAttribute('data-sourcepage') || '',
                        sourcefile: element.getAttribute('data-sourcefile') || '',
                        citation_path: element.getAttribute('data-citation-path') || '',
                        category: element.getAttribute('data-category') || '',
                        title: element.getAttribute('title') || '',
                        citation_text: element.innerText || ''
                    }))
                    .filter((citation) => citation.visible)
                """
            )
            selected_citation = select_citation(target_case, citations)
            citation_count = len(citations)
            page.locator(".supContainer").nth(selected_citation["index"]).click()
            page.get_by_text("Supporting content", exact=False).first.wait_for(state="visible", timeout=30_000)

            highlight = page.locator("#highlighted-subsection")
            highlight.wait_for(state="visible", timeout=30_000)
            highlighted_text = normalize(highlight.inner_text())
            supporting_card = highlight.locator("xpath=ancestor::*[contains(@class, 'supportingItem')][1]")
            supporting_card.wait_for(state="visible", timeout=5_000)
            supporting_card_text = supporting_card.inner_text()
            validate_highlight_identity(
                highlighted_text,
                supporting_card_text,
                expected_heading,
                str(target_case["subsection_id"]),
            )
            if expected_body not in highlighted_text and highlighted_text not in expected_body:
                raise BrowserGateError("Highlighted subsection text does not match canonical oracle evidence")
            if next_heading and next_heading in highlighted_text:
                raise BrowserGateError("Highlighted subsection includes the next canonical subsection")

            return {
                "browser": {
                    "candidate_url": candidate_url,
                    "question": question,
                    "citation_count": citation_count,
                    "clicked_index": selected_citation["index"],
                    "selected_citation": selected_citation,
                    "visible_citations": citations,
                    "supporting_content_visible": True,
                    "highlight_visible": True,
                    "highlighted_text_sha256": hashlib.sha256(highlighted_text.encode("utf-8")).hexdigest(),
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
        print(f"Browser highlight gate failed: {error}")
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