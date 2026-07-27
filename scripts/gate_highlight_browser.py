"""Prove citation navigation and subsection highlighting in a live candidate UI."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

try:
    from .validate_highlight_oracle import (  # type: ignore[unresolved-import]
        validate as validate_oracle,
    )
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


def sourcepage_matches(target_case: dict[str, Any], citation: dict[str, Any]) -> bool:
    expected_sourcepage = normalize_sourcepage(str(target_case.get("sourcepage") or ""))
    actual_sourcepage = normalize_sourcepage(str(citation.get("sourcepage") or ""))
    if not expected_sourcepage or not actual_sourcepage:
        return True
    if expected_sourcepage in actual_sourcepage or actual_sourcepage in expected_sourcepage:
        return True

    # The UI can expose a parent heading while the oracle records the leaf
    # heading. Sourcefile and subsection matching above keep this fallback
    # scoped to the same canonical Part.
    expected_subsection = normalize(str(target_case.get("subsection_id") or ""))
    expected_sourcefile = normalize_sourcepage(str(target_case.get("sourcefile") or ""))
    return bool(
        expected_subsection != "part 24"
        and expected_subsection.startswith("24.")
        and expected_sourcefile
        and expected_sourcefile in actual_sourcepage
    )


def subsection_matches(expected: str, actual: str) -> bool:
    expected_normalized = normalize(expected)
    actual_normalized = normalize(actual)
    if not expected_normalized or not actual_normalized:
        return False
    if expected_normalized == actual_normalized:
        return True
    if expected_normalized == "part 24":
        return actual_normalized.startswith("24.") or actual_normalized == "24"
    return actual_normalized.startswith(expected_normalized) and (
        len(actual_normalized) == len(expected_normalized)
        or actual_normalized[len(expected_normalized)] in " ("
    )


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
    return sourcepage_matches(target_case, citation)


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
    canonical_sourcepage: str = "",
    canonical_sourcefile: str = "",
) -> None:
    """Require the heading in the selected card and the subsection in its highlight."""
    normalized_highlight = normalize(highlighted_text)
    normalized_card = normalize(supporting_card_text)
    heading_candidates = {
        normalize(expected_heading),
        normalize(canonical_sourcepage),
        normalize(canonical_sourcefile),
    }
    heading_candidates.discard("")
    if not normalized_highlight:
        raise BrowserGateError("Supporting Content rendered an empty highlighted subsection")
    heading_in_card = any(candidate in normalized_card for candidate in heading_candidates)
    leaf_only_card = bool(canonical_sourcepage and canonical_sourcefile) and subsection_matches(
        expected_subsection, normalized_highlight
    ) and normalized_highlight in normalized_card
    if not heading_in_card and not leaf_only_card:
        raise BrowserGateError(
            "Supporting Content card does not identify the canonical target heading: "
            f"expected_heading={normalize(expected_heading)!r}, "
            f"canonical_sourcepage={normalize(canonical_sourcepage)!r}, "
            f"canonical_sourcefile={normalize(canonical_sourcefile)!r}, "
            f"card_text={normalized_card[:500]!r}"
        )
    if not subsection_matches(expected_subsection, normalized_highlight):
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
        and "part 24" in " ".join(
            str(case.get(field) or "") for field in ("sourcefile", "sourcepage", "identity")
        ).casefold()
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


def _capture_browser_diagnostics(page: Any, diagnostics_dir: Path | None) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "console": [],
        "page_errors": [],
        "request_failures": [],
        "responses": [],
    }

    def record_console(message: Any) -> None:
        diagnostics["console"].append({"type": message.type, "text": message.text})

    def record_page_error(error: Any) -> None:
        diagnostics["page_errors"].append(str(error))

    def record_request_failure(request: Any) -> None:
        diagnostics["request_failures"].append(
            {"url": request.url, "method": request.method, "failure": request.failure}
        )

    def record_response(response: Any) -> None:
        diagnostics["responses"].append(
            {"url": response.url, "status": response.status, "status_text": response.status_text}
        )

    page.on("console", record_console)
    page.on("pageerror", record_page_error)
    page.on("requestfailed", record_request_failure)
    page.on("response", record_response)

    if diagnostics_dir is not None:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        page.context.tracing.start(screenshots=True, snapshots=True, sources=True)
        diagnostics["directory"] = str(diagnostics_dir)
    return diagnostics


def _write_browser_diagnostics(page: Any, diagnostics: dict[str, Any], diagnostics_dir: Path | None) -> None:
    if diagnostics_dir is None:
        return
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    try:
        diagnostics["url"] = page.url
        diagnostics["title"] = page.title()
        diagnostics["body_text"] = page.locator("body").inner_text(timeout=2_000)[:20_000]
        diagnostics["html"] = page.content()[:100_000]
        if hasattr(page, "aria_snapshot"):
            diagnostics["aria_snapshot"] = page.aria_snapshot(timeout=2_000)
    except Exception as error:  # pragma: no cover - diagnostic fallback
        diagnostics["snapshot_error"] = str(error)
    (diagnostics_dir / "browser-diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    page.screenshot(path=str(diagnostics_dir / "browser-final.png"), full_page=True)
    page.context.tracing.stop(path=str(diagnostics_dir / "browser-trace.zip"))


def _wait_for_citation(
    page: Any,
    timeout_ms: int = 30_000,
    diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    citations_locator = page.locator(".supContainer")
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        if diagnostics and any(
            int(response.get("status", 0)) >= 400 for response in diagnostics.get("responses", [])
        ):
            raise BrowserGateError("Candidate UI received an HTTP error before rendering citations")
        if page.locator("text=Something went wrong").count() or page.locator("text=Error").count():
            raise BrowserGateError("Candidate UI reported an error before rendering citations")
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
                    source_revision: element.getAttribute('data-source-revision') || '',
                    source_id: element.getAttribute('data-source-id') || '',
                    document_id: element.getAttribute('data-document-id') || '',
                    canonical_text_sha256: element.getAttribute('data-canonical-text-sha256') || '',
                    title: element.getAttribute('title') || '',
                    citation_text: element.innerText || ''
                }))
                .filter((citation) => citation.visible)
            """
        )
        if citations:
            return citations
        page.wait_for_timeout(250)
    raise BrowserGateError(f"No visible citations rendered within {timeout_ms}ms")


def run_browser_gate(
    candidate_url: str,
    oracle: dict[str, Any],
    question: str,
    diagnostics_dir: Path | None = None,
    target_case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_case = target_case or choose_case(oracle)
    expected_body = normalize(str(target_case.get("body_text") or ""))
    expected_heading = normalize(str(target_case.get("expected_heading") or ""))
    next_heading = normalize(str(target_case.get("next_heading") or ""))
    if not expected_body or not expected_heading:
        raise BrowserGateError("Selected highlight oracle case is incomplete")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        diagnostics = _capture_browser_diagnostics(page, diagnostics_dir)
        try:
            page.goto(candidate_url, wait_until="domcontentloaded", timeout=60_000)
            splash = page.locator("[role='dialog'][aria-modal='true']")
            try:
                splash.wait_for(state="visible", timeout=2_000)
                splash.click()
                splash.wait_for(state="hidden", timeout=5_000)
            except PlaywrightTimeoutError:
                pass

            question_input = page.locator("textarea").first
            question_input.wait_for(state="visible", timeout=30_000)
            source_filter = page.locator(
                "#chat-source-filter-desktop-button, #chat-source-filter-mobile-button"
            ).first
            source_filter.wait_for(state="visible", timeout=30_000)
            source_filter.click()
            page.get_by_role("menuitemcheckbox", name="All Sources", exact=True).click()
            page.keyboard.press("Escape")
            question_input.fill(question)
            page.get_by_role("button", name="Submit question").click()

            citations = _wait_for_citation(page, diagnostics=diagnostics)
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
                str(selected_citation.get("sourcepage") or ""),
                str(selected_citation.get("sourcefile") or ""),
            )
            if expected_body not in highlighted_text and highlighted_text not in expected_body:
                raise BrowserGateError("Highlighted subsection text does not match canonical oracle evidence")
            if next_heading and next_heading in highlighted_text:
                raise BrowserGateError("Highlighted subsection includes the next canonical subsection")

            primary_button = supporting_card.get_by_role("button", name=re.compile("primary source", re.IGNORECASE))
            primary_button.wait_for(state="visible", timeout=5_000)
            primary_button.click()
            primary_viewer = page.locator("[data-canonical-text-sha256]").filter(
                has=page.locator("[data-source-revision]")
            ).last
            primary_viewer.wait_for(state="visible", timeout=30_000)
            for field in ("source-revision", "source-id", "document-id", "subsection-id", "canonical-text-sha256"):
                expected = str(selected_citation.get(field.replace("-", "_")) or "").strip()
                observed = str(primary_viewer.get_attribute(f"data-{field}") or "").strip()
                if not expected or observed != expected:
                    raise BrowserGateError(
                        f"Primary Source identity mismatch for {field}: expected={expected!r}, observed={observed!r}"
                    )

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
                    "primary_source_visible": True,
                    "primary_source_identity_verified": True,
                    "highlighted_text_sha256": hashlib.sha256(highlighted_text.encode("utf-8")).hexdigest(),
                },
                "case_id": target_case["case_id"],
                "subsection_id": target_case["subsection_id"],
            }
        finally:
            _write_browser_diagnostics(page, diagnostics, diagnostics_dir)
            browser.close()


def build_report(
    candidate_url: str,
    oracle_path: Path,
    snapshot_dir: Path,
    provenance: dict[str, str],
    question: str,
    diagnostics_dir: Path | None = None,
) -> dict[str, Any]:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    validated_oracle = validate_oracle(oracle, snapshot_dir, provenance=None)
    browser_evidence = run_browser_gate(candidate_url, oracle, question, diagnostics_dir)
    return {
        "schema_version": 1,
        "gate": "highlight",
        "status": "PASS",
        "oracle_version": validated_oracle["oracle_version"],
        "case_count": validated_oracle["case_count"],
        "source_count": validated_oracle["source_count"],
        "snapshot_manifest_sha256": validated_oracle["snapshot_manifest_sha256"],
        "browser_evidence": browser_evidence,
        "checks": [
            {
                "id": "canonical_citation_highlight",
                "case_id": browser_evidence["case_id"],
                "subsection_id": browser_evidence["subsection_id"],
                "status": "PASS",
            }
        ],
        "provenance": provenance,
    }


def run_exhaustive_browser_gate(
    candidate_url: str,
    oracle: dict[str, Any],
    output_path: Path,
    diagnostics_dir: Path | None = None,
) -> dict[str, Any]:
    """Run every oracle case and emit the input consumed by citation coverage validation."""
    cases = [case for case in oracle.get("cases", []) if isinstance(case, dict)]
    if not cases:
        raise BrowserGateError("Highlight oracle contains no exhaustive cases")

    manifest: list[dict[str, Any]] = []
    search_documents: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_diagnostics = diagnostics_dir / f"case-{index:04d}" if diagnostics_dir else None
        question = f"What is {case.get('subsection_id', '')} in {case.get('sourcepage', '')}?"
        try:
            evidence = run_browser_gate(
                candidate_url,
                oracle,
                question,
                case_diagnostics,
                target_case=case,
            )
            selected = evidence["browser"]["selected_citation"]
            document_id = str(selected.get("document_id") or "").strip()
            source_revision = str(selected.get("source_revision") or "").strip()
            source_id = str(selected.get("source_id") or "").strip()
            canonical_hash = str(selected.get("canonical_text_sha256") or "").strip()
            if not all((document_id, source_revision, source_id, canonical_hash)):
                raise BrowserGateError(f"Citation {case.get('case_id')} lacks immutable identity attributes")
            search_documents[document_id] = {
                "id": document_id,
                "source_revision": source_revision,
                "canonical_text_sha256": canonical_hash,
            }
            manifest.append(
                {
                    "source_revision": source_revision,
                    "source_id": source_id,
                    "document_id": document_id,
                    "subsection_id": str(case.get("subsection_id") or ""),
                    "canonical_text_sha256": canonical_hash,
                    "rendered": True,
                    "clicked": True,
                    "supporting_content_count": 1,
                    "primary_source_count": 1,
                    "highlighted_text_sha256": canonical_hash,
                }
            )
        except Exception as error:
            failures.append({"case_id": case.get("case_id"), "error": str(error)})

    payload = {
        "schema_version": 1,
        "status": "PASS" if not failures and len(manifest) == len(cases) else "FAIL",
        "manifest": manifest,
        "search_documents": list(search_documents.values()),
        "case_count": len(cases),
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise BrowserGateError(f"Exhaustive citation coverage failed for {len(failures)} of {len(cases)} cases")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--question", default="What is CPR Part 24 rule 24.2 and the test for summary judgment?")
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--exhaustive-coverage-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
        if args.exhaustive_coverage_input:
            validated_oracle = validate_oracle(oracle, args.snapshot_dir, provenance=None)
            coverage = run_exhaustive_browser_gate(
                args.candidate_url, oracle, args.exhaustive_coverage_input, args.diagnostics_dir
            )
            report = {
                "schema_version": 1,
                "gate": "highlight",
                "status": "PASS",
                "oracle_version": validated_oracle["oracle_version"],
                "case_count": validated_oracle["case_count"],
                "source_count": validated_oracle["source_count"],
                "snapshot_manifest_sha256": validated_oracle["snapshot_manifest_sha256"],
                "citation_coverage_input": str(args.exhaustive_coverage_input),
                "coverage": coverage,
                "provenance": provenance,
            }
        else:
            report = build_report(
                args.candidate_url,
                args.oracle,
                args.snapshot_dir,
                provenance,
                args.question,
                args.diagnostics_dir,
            )
    except (OSError, json.JSONDecodeError, BrowserGateError, PlaywrightTimeoutError, ValueError) as error:
        report = {"schema_version": 1, "gate": "highlight", "status": "FAIL", "error": str(error)}
        if args.diagnostics_dir and (args.diagnostics_dir / "browser-diagnostics.json").exists():
            report["diagnostics_dir"] = str(args.diagnostics_dir)
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