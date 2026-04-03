"""
Computer Use Local Test Harness
================================
Uses Azure OpenAI GPT-5.4 (or computer-use-preview) with the Responses API
to drive a Playwright browser against the locally-running RAG app.

The model sees screenshots, decides on clicks/typing/scrolling, and this script
executes those actions in a real Chromium window pointed at localhost.

Prerequisites:
  1. App running locally:  ./app/start.sh  (serves at http://localhost:50505)
     OR  npm run dev (frontend at http://localhost:5173) + quart backend
  2. Azure OpenAI deployment of gpt-5.4 (or computer-use-preview)
  3. Azure CLI logged in:  az login
  4. Python packages:  pip install openai azure-identity playwright
  5. Playwright browsers:  playwright install chromium

Environment variables (set in .env or export):
  COMPUTER_USE_AZURE_OPENAI_ENDPOINT  – e.g. https://myresource.openai.azure.com
  COMPUTER_USE_MODEL                  – deployment name, default "gpt-5.4"
  COMPUTER_USE_TARGET_URL             – default http://localhost:50505
    COMPUTER_USE_MAX_ITERATIONS         – default 12
  COMPUTER_USE_DISPLAY_WIDTH          – default 1440
  COMPUTER_USE_DISPLAY_HEIGHT         – default 900

Usage:
  # Interactive mode (type tasks at the prompt):
  python scripts/computer_use_test.py

  # Single-task mode:
  python scripts/computer_use_test.py --task "Ask 'What is the dental plan?' and report the answer"

    # Built-in detailed citation suite:
    python scripts/computer_use_test.py --suite detailed-citations

  # Headless mode (no visible browser window):
  python scripts/computer_use_test.py --headless --task "Search for eye exam coverage"

Safety:
  - Navigation is restricted to the TARGET_URL origin.
  - A hard iteration cap prevents runaway loops.
  - Safety checks from the API are surfaced and require user confirmation.
  - Run in a test environment; do not expose to sensitive data.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import OpenAI
from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

def read_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


def is_computer_use_capable_model(model_name: str) -> bool:
    normalized = read_env(model_name) if model_name.isupper() else model_name.strip().strip('"').strip("'")
    normalized = normalized.lower()
    if not normalized:
        return False
    if "computer-use-preview" in normalized:
        return True
    if normalized.startswith("gpt-5.4") and "mini" not in normalized and "nano" not in normalized:
        return True
    return False


def resolve_model() -> str:
    explicit_model = read_env("COMPUTER_USE_MODEL")
    if explicit_model:
        return explicit_model

    deployment_candidates = [
        (read_env("AZURE_OPENAI_CHATGPT_DEPLOYMENT"), read_env("AZURE_OPENAI_CHATGPT_MODEL")),
        (read_env("AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT"), read_env("AZURE_OPENAI_SEARCHAGENT_MODEL")),
        (read_env("AZURE_OPENAI_EVAL_DEPLOYMENT"), read_env("AZURE_OPENAI_EVAL_MODEL")),
        (read_env("AZURE_OPENAI_GPT4V_DEPLOYMENT"), read_env("AZURE_OPENAI_GPT4V_MODEL")),
        (read_env("AZURE_OPENAI_KNOWLEDGEBASE_DEPLOYMENT"), read_env("AZURE_OPENAI_KNOWLEDGEBASE_MODEL")),
    ]
    for deployment_name, model_name in deployment_candidates:
        if deployment_name and is_computer_use_capable_model(model_name):
            return deployment_name

    return "gpt-5.4"


ENDPOINT = read_env("COMPUTER_USE_AZURE_OPENAI_ENDPOINT") or read_env("AZURE_OPENAI_ENDPOINT")
MODEL = resolve_model()
TARGET_URL = os.getenv("COMPUTER_USE_TARGET_URL", "http://localhost:50505")
MAX_ITERATIONS = int(os.getenv("COMPUTER_USE_MAX_ITERATIONS", "12"))
DISPLAY_WIDTH = int(os.getenv("COMPUTER_USE_DISPLAY_WIDTH", "1440"))
DISPLAY_HEIGHT = int(os.getenv("COMPUTER_USE_DISPLAY_HEIGHT", "900"))

# Allowed origin for navigation restriction
ALLOWED_ORIGIN = f"{urlparse(TARGET_URL).scheme}://{urlparse(TARGET_URL).netloc}"

# Log directory
LOG_DIR = Path("scripts/computer_use_logs")
SCENARIO_PARSE_RETRIES = 1


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    source_filter: str
    questions: list[str]
    focus: list[str] = field(default_factory=list)
    notes: str = ""
    excluded_heading: str = ""


SUITES: dict[str, list[Scenario]] = {
    "highlight-regression": [
        Scenario(
            key="all-sources-circuit-commercial-claim-form-subsection",
            title="All Sources: Circuit Commercial Claim Form Subsection",
            source_filter="All Sources",
            questions=[
                "If particulars of claim are not contained in or served with the claim form in a Circuit Commercial Court claim, what must the claim form say and what timing applies for serving particulars of claim?"
            ],
            focus=["supporting-content highlight", "full-section rendering", "source title fidelity"],
            notes="Use screenshots of the supporting-content panel to verify the cited view starts at rule 59.4 itself, excludes the adjacent heading above it, and does not bleed into rule 59.5.",
            excluded_heading="Claim form and particulars of claim",
        ),
        Scenario(
            key="cpr-circuit-commercial-claim-form-subsection",
            title="CPR: Circuit Commercial Claim Form Subsection",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=[
                "Under CPR Part 59, if particulars of claim are not contained in or served with the claim form, what must the claim form say and what timing applies for serving particulars of claim?"
            ],
            focus=["part-59 highlighting", "full-section rendering", "supporting panel scope"],
            notes="This is the direct regression check that the clicked panel starts at rule 59.4, excludes the adjacent heading above it, and stops before rule 59.5.",
            excluded_heading="Claim form and particulars of claim",
        ),
        Scenario(
            key="commercial-guide-default-judgment-subsection",
            title="Commercial Court Guide: Default Judgment Subsection",
            source_filter="Commercial Court Guide",
            questions=["What does the Commercial Court Guide say about default judgment?"],
            focus=["guide subsection targeting", "highlight screenshot review", "citation click targeting"],
            excluded_heading="Default judgment",
        ),
        Scenario(
            key="chancery-guide-applications-before-issue-subsection",
            title="Chancery Guide: Applications Before Issue Subsection",
            source_filter="Chancery Guide",
            questions=["What does the Chancery Guide say about applications made before issue of a claim form?"],
            focus=["hierarchical subsection targeting", "highlight screenshot review", "panel destination accuracy"],
            excluded_heading="Applications before issue",
        ),
        Scenario(
            key="senior-courts-costs-office-guide",
            title="Senior Courts Costs Office Guide",
            source_filter="Senior Courts Costs Office",
            questions=[
                "In the Senior Courts Costs Office Guide, what does paragraph 3.3 say about costs management orders and costs capping orders?"
            ],
            focus=["costs guide subsection targeting", "highlight screenshot review", "source filter coverage"],
            excluded_heading="Costs management orders and costs capping orders",
        ),
        Scenario(
            key="court-of-appeal-civil-division-guide",
            title="Court of Appeal Civil Division Guide",
            source_filter="Court of Appeal Civil Division",
            questions=[
                "In the Court of Appeal Civil Division Guide, what does paragraph 39.2 say about an application for disclosure?"
            ],
            focus=["appeal guide subsection targeting", "highlight screenshot review", "source filter coverage"],
            excluded_heading="Application for disclosure",
        ),
        Scenario(
            key="kbd-guide-enrolment-subsection",
            title="King's Bench Division Guide: Enrolment Subsection",
            source_filter="King's Bench Division Guide",
            questions=["What does the King's Bench Division Guide say about enrolment of deeds and other documents?"],
            focus=["guide subsection visibility", "highlight screenshot review", "source-title fidelity"],
            excluded_heading="Enrolment of deeds and other documents",
        ),
        Scenario(
            key="tcc-guide-enforcement-subsection",
            title="Technology And Construction Court Guide: Enforcement Subsection",
            source_filter="Technology and Construction Court Guide",
            questions=["How are TCC judgments and orders generally enforced according to the Technology and Construction Court Guide?"],
            focus=["section subsection targeting", "highlight screenshot review", "answer-to-panel correlation"],
            notes="This question typically produces a long answer with many citations. Wait extra time for streaming to finish — the answer is not complete until you see in-text citation markers like [1] and a citation list at the bottom.",
            excluded_heading="Enforcement",
        ),
        Scenario(
            key="patents-guide-handing-down-subsection",
            title="Patents Court Guide: Handing Down Subsection",
            source_filter="Patents Court Guide",
            questions=["After judgment is handed down, what does the Patents Court Guide say about agreed minutes of order?"],
            focus=["annex subsection targeting", "highlight screenshot review", "panel destination accuracy"],
            excluded_heading="Handing down order",
        ),
        Scenario(
            key="circuit-commercial-guide-triaging-subsection",
            title="Circuit Commercial Court Guide: Triaging Subsection",
            source_filter="Circuit Commercial Court Guide",
            questions=["What does the Circuit Commercial Court Guide say about London Circuit Commercial Court triaging?"],
            focus=["court-guide subsection targeting", "highlight screenshot review", "source filter coverage"],
            excluded_heading="London Circuit Commercial Court Triaging",
        ),
    ],
    "detailed-citations": [
        Scenario(
            key="all-sources-summary-judgment-and-costs",
            title="All Sources: Summary Judgment And Costs",
            source_filter="All Sources",
            questions=["What is the procedure for summary judgment and what costs rules apply?"],
            focus=["mixed CPR + practice direction citations", "cross-source in-text click correlation", "subsection highlighting"],
            notes="Based on tests/e2e_citations.py v3 scenario using Part 24 and Practice Direction 44 naming patterns.",
        ),
        Scenario(
            key="all-sources-dense-multi-citation",
            title="All Sources: Dense Multi-Citation Answer",
            source_filter="All Sources",
            questions=[
                "In short sentences, explain the overriding objective, case management powers, summary judgment, appeals, costs discretion, and derivative claims.",
            ],
            focus=["many-citation formatting", "citation list readability", "multiple in-text citation destinations"],
            notes="Derived from the v3 dense mixed-source edge case in tests/e2e_citations.py.",
        ),
        Scenario(
            key="all-sources-preaction-summary-judgment-pi",
            title="All Sources: Pre-Action + Summary Judgment + PI PD",
            source_filter="All Sources",
            questions=[
                "Before commencing construction proceedings, what pre-action steps apply, when can summary judgment be granted, and what Practice Direction 27B point is relevant?",
            ],
            focus=["three distinct citations", "source switching in supporting panel", "citation-to-sentence correlation"],
            notes="Grounded in the mixed pre-action / Part 24 / PD27B v3 scenario.",
        ),
        Scenario(
            key="cpr-overriding-objective",
            title="CPR: Overriding Objective",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["In one sentence, what is the overriding objective under CPR Part 1?"],
            focus=["source-filtered citations", "bottom citation labels", "highlight accuracy"],
            notes="Uses the Part 1 v3 sourcepage pattern 'Part 1 – Overriding Objective'.",
        ),
        Scenario(
            key="cpr-case-management-powers",
            title="CPR: Case Management Powers",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["What power does the court have under CPR Part 3 to extend or shorten time for compliance?"],
            focus=["apostrophe and punctuation handling", "single-rule highlighting", "clean citation labels"],
            notes="Derived from the Part 3 v3 citation parsing scenario.",
        ),
        Scenario(
            key="cpr-summary-judgment",
            title="CPR: Summary Judgment",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["When may the court give summary judgment under CPR Part 24?"],
            focus=["two-digit part citation labels", "citation targeting", "supporting content scope"],
            notes="Uses the canonical v3 summary judgment scenario from e2e and live tests.",
        ),
        Scenario(
            key="cpr-appeals-permission",
            title="CPR: Appeals Permission",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["When is permission to appeal generally required under CPR Part 52?"],
            focus=["short-title CPR part citations", "clicked citation correctness"],
            notes="Matches the v3 Part 52 citation scenario.",
        ),
        Scenario(
            key="cpr-part44-costs",
            title="CPR: Part 44 Costs",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["Summarize what CPR Part 44 says about the court's discretion on costs in two short sentences."],
            focus=["multi-citation labels", "supporting content scope", "citation-to-answer correlation"],
            notes="This should often pull Practice Direction 44 style v3 citations, not only Part 44 headings.",
        ),
        Scenario(
            key="cpr-part46-special-cases",
            title="CPR: Part 46 Special Cases",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["What does CPR Part 46 say about costs in special cases?"],
            focus=["special-case citation mapping", "subsection highlight correctness"],
            notes="Useful for finding label regressions where numeric subsection text can leak into citation labels.",
        ),
        Scenario(
            key="cpr-service-on-company",
            title="CPR: Service On A Company",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["How is a claim form served personally on a company under CPR Part 6?"],
            focus=["service-rule citation targeting", "single-rule highlighting", "supporting panel scope"],
            notes="Grounded in Part 6.5 and useful for checking concise rule extraction outside the costs-heavy CPR areas.",
        ),
        Scenario(
            key="cpr-transfer-criteria",
            title="CPR: Transfer Criteria",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["What criteria must the court consider under CPR Part 30 when deciding whether to transfer proceedings?"],
            focus=["mid-range CPR part citations", "multi-factor supporting content", "citation-to-answer correlation"],
            notes="Based on Part 30 transfer material and intended to widen coverage beyond Parts 1, 3, 24, 44, and 52.",
        ),
        Scenario(
            key="cpr-standard-disclosure",
            title="CPR: Standard Disclosure",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["What does CPR Part 31 say standard disclosure requires a party to disclose?"],
            focus=["list-style rule citations", "supporting-content completeness", "clean citation labels"],
            notes="Grounded in Part 31.6 and useful for testing disclosure content instead of case-management content.",
        ),
        Scenario(
            key="cpr-judicial-review-permission-and-time",
            title="CPR: Judicial Review Permission And Time",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["Under CPR Part 54, when is permission required for judicial review and what does the rule say about timing for filing the claim form?"],
            focus=["administrative-law CPR citations", "multi-point answer grounding", "supporting panel accuracy"],
            notes="Uses Part 54 judicial review content so the suite exercises administrative court material, not just mainstream civil procedure topics.",
        ),
        Scenario(
            key="cpr-circuit-commercial-default-judgment",
            title="CPR: Circuit Commercial Default Judgment",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["In a Circuit Commercial Court Part 7 claim where the claim form has been served but particulars have not, how is default judgment obtained if the defendant files no acknowledgment of service?"],
            focus=["specialist-list CPR citations", "less-common Part 59 retrieval", "bottom citation readability"],
            notes="Grounded in Part 59.7 and intended to push the harness into the Circuit Commercial portion of the index.",
        ),
        Scenario(
            key="cpr-practice-direction-44",
            title="CPR: Practice Direction 44",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["What does Practice Direction 44 say about the court's discretion over costs?"],
            focus=["practice-direction citation labels", "clicked panel title quality", "subsection highlighting"],
            notes="Derived from the PD_44 v3 fixture used in tests/e2e_citations.py.",
        ),
        Scenario(
            key="cpr-practice-direction-19a",
            title="CPR: Practice Direction 19A",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["What is a derivative claim under Practice Direction 19A?"],
            focus=["alpha-suffix practice direction labels", "citation correlation"],
            notes="Exercises the 19A naming pattern from the v3 fixtures.",
        ),
        Scenario(
            key="cpr-preaction-judicial-review",
            title="CPR: Pre-Action Protocol For Judicial Review",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["Before making a judicial review claim, what pre-action step should the claimant take?"],
            focus=["truncated sourcefile labels like 'Pre'", "panel destination accuracy", "highlighting despite generic labels"],
            notes="Based on the pre-action judicial review v3 scenario where citations may use a short 'Pre' label.",
        ),
        Scenario(
            key="cpr-charging-order-application",
            title="CPR: Charging Order Application",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["What information must an application notice for a charging order contain under Practice Direction 73?"],
            focus=["practice-direction list citations", "subsection highlighting", "supporting content depth"],
            notes="Grounded in Practice Direction 73 section 1.2 and broadens enforcement-related coverage.",
        ),
        Scenario(
            key="cpr-warrant-of-delivery-warning-notice",
            title="CPR: Warrant Of Delivery Warning Notice",
            source_filter="Civil Procedure Rules and Practice Directions",
            questions=["Under CPR Part 83, what warning notice and inventory steps apply before and after executing a warrant of delivery?"],
            focus=["enforcement-rule citations", "multiple related subsections", "supporting panel completeness"],
            notes="Based on Part 83.24 and 83.25, which gives the suite another enforcement-heavy area of the index.",
        ),
        Scenario(
            key="commercial-guide-default-judgment",
            title="Commercial Court Guide: Default Judgment",
            source_filter="Commercial Court Guide",
            questions=["What does the Commercial Court Guide say about default judgment?"],
            focus=["complex guide sourcepage labels", "panel header correctness", "citation click correlation"],
            notes="Uses the B.11.1 Commercial Court Guide v3 scenario rather than a generic guide-purpose question.",
        ),
        Scenario(
            key="chancery-guide-applications-before-issue",
            title="Chancery Guide: Applications Before Issue",
            source_filter="Chancery Guide",
            questions=["What does the Chancery Guide say about applications made before issue of a claim form?"],
            focus=["hierarchical sourcepage labels", "in-text click targeting", "highlight alignment"],
            notes="Matches the v3 Chancery guide scenario for applications before issue.",
        ),
        Scenario(
            key="kbd-guide-enrolment-of-deeds",
            title="King's Bench Division Guide: Enrolment Of Deeds",
            source_filter="King's Bench Division Guide",
            questions=["What does the King's Bench Division Guide say about enrolment of deeds and other documents?"],
            focus=["PDF-style sourcefile labels", "label readability", "citation destination accuracy"],
            notes="Based on the v3 King's Bench fixture where sourcefile resembles a PDF filename.",
        ),
        Scenario(
            key="tcc-guide-enforcement",
            title="Technology And Construction Court Guide: Enforcement",
            source_filter="Technology and Construction Court Guide",
            questions=["How are TCC judgments and orders generally enforced according to the Technology and Construction Court Guide?"],
            focus=["section-number guide citations", "supporting content highlight", "sentence-to-panel correlation"],
            notes="Uses the v3 TCC guide section 17.2 enforcement pattern.",
        ),
        Scenario(
            key="tcc-guide-outside-london-heading",
            title="Technology And Construction Court Guide: Outside London Heading",
            source_filter="Technology and Construction Court Guide",
            questions=["When TCC proceedings are commenced in a district registry Business and Property Court outside London, how should the heading on statements of case and applications be styled?"],
            focus=["guide heading citations", "quoted-text supporting content", "destination accuracy"],
            notes="Grounded in the TCC Guide section on the TCC outside London and useful for testing long quoted supporting content.",
        ),
        Scenario(
            key="patents-guide-handing-down-order",
            title="Patents Court Guide: Handing Down Order",
            source_filter="Patents Court Guide",
            questions=["After judgment is handed down, what does the Patents Court Guide say about agreed minutes of order?"],
            focus=["annex-style citation labels", "citation destination accuracy", "clicked subsection highlight"],
            notes="Based on the Annex F v3 Patents Court fixture.",
        ),
        Scenario(
            key="patents-guide-trial-listing-window",
            title="Patents Court Guide: Trial Listing Window",
            source_filter="Patents Court Guide",
            questions=["What does the Patents Court Guide say about the court's objective for bringing patent cases on for trial and how trial windows are used?"],
            focus=["annex and practice-statement citations", "supporting panel scope", "answer-to-citation correlation"],
            notes="Grounded in the Patents Court Guide trial-listing practice statement and meant to exercise a different section from the handing-down-order scenario.",
        ),
    ]
}

# ---------------------------------------------------------------------------
# Key mapping (model key names → Playwright key names)
# ---------------------------------------------------------------------------

KEY_MAPPING: dict[str, str] = {
    "/": "Slash",
    "\\": "Backslash",
    "alt": "Alt",
    "option": "Alt",
    "arrowdown": "ArrowDown",
    "down": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "left": "ArrowLeft",
    "arrowright": "ArrowRight",
    "right": "ArrowRight",
    "arrowup": "ArrowUp",
    "up": "ArrowUp",
    "backspace": "Backspace",
    "ctrl": "Control",
    "control": "Control",
    "cmd": "Meta",
    "command": "Meta",
    "meta": "Meta",
    "win": "Meta",
    "super": "Meta",
    "delete": "Delete",
    "enter": "Enter",
    "return": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "shift": "Shift",
    "space": " ",
    "tab": "Tab",
    "pagedown": "PageDown",
    "pageup": "PageUp",
    "home": "Home",
    "end": "End",
    "insert": "Insert",
    **{f"f{i}": f"F{i}" for i in range(1, 13)},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

last_successful_screenshot: str | None = None


def build_single_task(task: str) -> str:
    return task


def build_scenario_task(scenario: Scenario) -> str:
    questions = "\n".join(f"{index}. Ask exactly: \"{question}\"" for index, question in enumerate(scenario.questions, start=1))
    focus = ", ".join(scenario.focus) if scenario.focus else "citation accuracy and supporting content correctness"
    notes = f" Extra notes: {scenario.notes}" if scenario.notes else ""
    heading_hint = f" Adjacent heading text that should stay outside the highlighted block when available: '{scenario.excluded_heading}'." if scenario.excluded_heading else ""
    return textwrap.dedent(
    f"""
        You are testing the legal RAG UI for citation quality.

                Scenario name: {scenario.title}
                Source filter to select before asking questions: {scenario.source_filter}
        Focus areas: {focus}.{notes}{heading_hint}

                Steps:
                1. Open the chat page if needed.
                2. Select the source filter exactly as '{scenario.source_filter}'. If it is already selected, keep it. If the dropdown shows multiple selected sources such as '3 selected', deselect the extras and do not continue until the control visibly shows only the exact required source label, or 'All Sources' when that is the intended filter.
                3. Run the following question flow in order:
                {questions}
                4. For each answer:
                     - Wait until the assistant has fully finished responding before inspecting citations. If text is still streaming or changing, wait. Before treating the answer as truncated, check whether the visible text is merely clipped by the bottom of the viewport: if the answer card or page can scroll further, scroll down to confirm whether the answer continues below. An answer is NOT finished if it ends mid-sentence, ends mid-bullet, has dangling markdown such as an unmatched '**', has no visible citation labels like [1] in the body, or shows a loading indicator. If any of those conditions appear, you MUST wait at least 8 more seconds, take another screenshot, and re-check. Complete up to 3 full wait-and-recheck cycles before concluding the answer is truncated or missing citations.
                     - Record the answer text verbatim.
                     - Record every in-text citation shown in the answer body.
                     - Record every citation label shown in the citation list at the bottom. If the number of visible citation labels is fewer than the number of in-text citations, scroll down to check for additional labels below the viewport before reporting a mismatch.
                     - State whether any label is malformed, ambiguous, duplicated, or looks like the attached bug pattern such as '1. 1' or '1. 1.1'. Do not treat shortened but still valid labels as an issue by themselves.
                     - Click only bracketed in-text citations inside the answer body such as '[1]' or '[2]'. Do not click citation labels in the bottom citation list, source links, or any external links. If there is a second distinct in-text citation, click that too.
                     - After each click, verify whether the opened supporting-content panel matches the claim in the sentence, whether the destination looks like the correct source, and whether the cited subsection is clearly highlighted.
                     - Use the screenshot after the click to inspect the highlighted block itself. Verify that the full supporting section remains visible while the highlighted block itself starts at the cited subsection rather than including the adjacent heading/title above it.
                     - If this scenario provides adjacent heading text, explicitly check that the heading is absent from the highlighted block.
                     - Verify the highlighted block does not spill into the next subsection when a clear next subsection or heading is visible in the source.
                     - Record whether the panel shows only a tiny snippet or a substantial full section.
                     - Record whether an Updated date is visible.
                     - Only report issues for citations or supporting-content behavior you actually verified. If the answer contains more than two distinct in-text citations, it is acceptable not to verify the additional ones.
                5. Treat a missing answer as an issue only if the UI still shows misleading citations or mismatched supporting content.
                6. At the end, return exactly one JSON object inside a ```json fenced block using this shape:
                     {{
                         "scenario": "{scenario.key}",
                         "title": "{scenario.title}",
                         "sourceFilter": "{scenario.source_filter}",
                         "answers": [
                             {{
                                 "question": "...",
                                 "answerText": "...",
                                 "inTextCitations": ["[1]", "[2]"],
                                 "citationLabels": ["1. ...", "2. ..."],
                                 "clickedCitations": [
                                     {{
                                         "citation": "[1]",
                                         "sentenceClaim": "...",
                                         "panelTitle": "...",
                                         "panelSourceLooksCorrect": true,
                                         "answerCorrelatesToCitation": true,
                                         "subsectionHighlighted": true,
                                         "highlightExcludesAdjacentTitle": true,
                                         "excludedHeadingAbsent": true,
                                         "supportingContentSubstantial": true,
                                         "updatedVisible": false,
                                         "notes": "..."
                                     }}
                                 ]
                             }}
                         ],
                         "issues": [
                             {{
                                 "severity": "high|medium|low",
                                 "title": "...",
                                 "evidence": "...",
                                 "suggestedArea": "frontend|backend|data|unknown"
                             }}
                         ],
                         "summary": {{
                             "citationFormattingPass": true,
                             "citationTargetingPass": true,
                             "highlightingPass": true,
                             "subsectionOnlyHighlightPass": true,
                             "notes": "..."
                         }}
                     }}
                7. If there are no issues, return an empty issues array.
                """
    ).strip()


def extract_json_block(text: str) -> dict[str, Any] | None:
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced_match.group(1) if fenced_match else None
    if not candidate:
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        candidate = brace_match.group(1) if brace_match else None
    if not candidate:
        return None

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_structured_result(raw_result: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[str] = []

    final_message = raw_result.get("final_message", "")
    if isinstance(final_message, str) and final_message.strip():
        candidates.append(final_message)

    messages = raw_result.get("messages", [])
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, str) and message.strip():
                candidates.append(message)

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        parsed = extract_json_block(candidate)
        if parsed:
            return parsed

    return None


def build_parse_failure_issue(raw_result: dict[str, Any], scenario: Scenario, attempts_used: int) -> dict[str, Any]:
    final_message = str(raw_result.get("final_message", "")).strip()
    if final_message:
        evidence = (
            f"Scenario did not return the required JSON block after {attempts_used} attempt(s). "
            f"Last model message: {final_message[:240]}"
        )
    else:
        evidence = (
            f"Scenario did not return the required JSON block after {attempts_used} attempt(s). "
            f"The browser agent exhausted its run without any final structured response."
        )

    return {
        "scenario": scenario.key,
        "title": "Scenario did not return parseable JSON",
        "severity": "high",
        "evidence": evidence,
        "suggestedArea": "unknown",
    }


def normalize_issue(issue: dict[str, Any], scenario: Scenario) -> dict[str, Any]:
    return {
        "scenario": scenario.key,
        "title": str(issue.get("title", "Untitled issue")),
        "severity": str(issue.get("severity", "medium")),
        "evidence": str(issue.get("evidence", "")),
        "suggestedArea": str(issue.get("suggestedArea", "unknown")),
    }


def aggregate_suite_results(suite_name: str, scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    scenarios_with_parse_failures: list[str] = []
    for result in scenario_results:
        parsed = result.get("parsed_result")
        scenario_key = result.get("scenario", {}).get("key", "unknown")
        if not parsed:
            scenarios_with_parse_failures.append(scenario_key)
        for issue in result.get("issues", []):
            if isinstance(issue, dict):
                issues.append(issue)

    severity_counts = {
        "high": sum(1 for issue in issues if issue.get("severity") == "high"),
        "medium": sum(1 for issue in issues if issue.get("severity") == "medium"),
        "low": sum(1 for issue in issues if issue.get("severity") == "low"),
    }

    return {
        "suite": suite_name,
        "scenarioCount": len(scenario_results),
        "issueCount": len(issues),
        "severityCounts": severity_counts,
        "parseFailures": scenarios_with_parse_failures,
        "issues": issues,
    }


def validate_coordinates(x: int | float, y: int | float) -> tuple[int, int]:
    return max(0, min(int(x), DISPLAY_WIDTH)), max(0, min(int(y), DISPLAY_HEIGHT))


def is_allowed_url(url: str) -> bool:
    """Return True if url belongs to the allowed origin."""
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return origin == ALLOWED_ORIGIN or url in ("about:blank", "")


async def take_screenshot(page) -> str:
    global last_successful_screenshot
    try:
        screenshot_bytes = await page.screenshot(full_page=False)
        last_successful_screenshot = base64.b64encode(screenshot_bytes).decode("utf-8")
        return last_successful_screenshot
    except Exception as exc:
        print(f"  [warn] screenshot failed: {exc}")
        if last_successful_screenshot:
            return last_successful_screenshot
        raise


async def reset_app_state(page) -> None:
    """Reset browser-side state so each task starts from a clean chat session."""
    try:
        await page.context.clear_cookies()
    except Exception as exc:
        print(f"  [warn] clear_cookies failed: {exc}")

    try:
        await page.goto(TARGET_URL, wait_until="domcontentloaded")
        await page.evaluate(
            """
            async () => {
                try { window.localStorage.clear(); } catch {}
                try { window.sessionStorage.clear(); } catch {}

                try {
                    if ('caches' in window) {
                        const cacheKeys = await window.caches.keys();
                        await Promise.all(cacheKeys.map(key => window.caches.delete(key)));
                    }
                } catch {}

                try {
                    if ('indexedDB' in window && typeof indexedDB.databases === 'function') {
                        const databases = await indexedDB.databases();
                        await Promise.all(
                            (databases || []).map(database => {
                                if (!database || !database.name) {
                                    return Promise.resolve();
                                }
                                return new Promise(resolve => {
                                    const request = indexedDB.deleteDatabase(database.name);
                                    request.onsuccess = () => resolve(null);
                                    request.onerror = () => resolve(null);
                                    request.onblocked = () => resolve(null);
                                });
                            })
                        );
                    }
                } catch {}
            }
            """
        )
        await page.reload(wait_until="domcontentloaded")
        await asyncio.sleep(1.0)
    except Exception as exc:
        print(f"  [warn] app-state reset failed: {exc}")

    try:
        clear_chat_button = page.get_by_role("button", name=re.compile(r"clear chat", re.I))
        if await clear_chat_button.count() > 0 and await clear_chat_button.first.is_visible():
            await clear_chat_button.first.click()
            await asyncio.sleep(0.5)
    except Exception:
        pass


async def handle_action(page, action: dict) -> None:
    """Execute a single action dict returned by the model."""
    action_type = action.get("type")

    if action_type == "click":
        button = action.get("button", "left")
        x, y = validate_coordinates(action.get("x", 0), action.get("y", 0))
        print(f"    click ({x}, {y}) button={button}")
        if button == "back":
            await page.go_back()
        elif button == "forward":
            await page.go_forward()
        elif button == "wheel":
            await page.mouse.wheel(x, y)
        else:
            btn = {"left": "left", "right": "right", "middle": "middle"}.get(button, "left")
            await page.mouse.click(x, y, button=btn)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=3000)
            except PlaywrightTimeout:
                pass

    elif action_type == "double_click":
        x, y = validate_coordinates(action.get("x", 0), action.get("y", 0))
        print(f"    double_click ({x}, {y})")
        await page.mouse.dblclick(x, y)

    elif action_type == "drag":
        path = action.get("path", [])
        if len(path) < 2:
            print("    drag: need ≥2 points, skipping")
            return
        start = path[0]
        sx, sy = validate_coordinates(start.get("x", 0), start.get("y", 0))
        print(f"    drag from ({sx}, {sy}) through {len(path)-1} points")
        await page.mouse.move(sx, sy)
        await page.mouse.down()
        for point in path[1:]:
            px, py = validate_coordinates(point.get("x", 0), point.get("y", 0))
            await page.mouse.move(px, py)
        await page.mouse.up()

    elif action_type == "move":
        x, y = validate_coordinates(action.get("x", 0), action.get("y", 0))
        print(f"    move ({x}, {y})")
        await page.mouse.move(x, y)

    elif action_type == "scroll":
        scroll_x = action.get("scroll_x", 0)
        scroll_y = action.get("scroll_y", 0)
        x, y = validate_coordinates(action.get("x", 0), action.get("y", 0))
        print(f"    scroll at ({x}, {y}) dx={scroll_x} dy={scroll_y}")
        await page.mouse.move(x, y)
        await page.evaluate(
            f"window.scrollBy({{left: {scroll_x}, top: {scroll_y}, behavior: 'smooth'}});"
        )

    elif action_type == "keypress":
        keys = action.get("keys", [])
        print(f"    keypress {keys}")
        mapped = [KEY_MAPPING.get(k.lower(), k) for k in keys]
        if len(mapped) > 1:
            for k in mapped:
                await page.keyboard.down(k)
            await asyncio.sleep(0.1)
            for k in reversed(mapped):
                await page.keyboard.up(k)
        else:
            for k in mapped:
                await page.keyboard.press(k)

    elif action_type == "type":
        text = action.get("text", "")
        print(f"    type: {text!r}")
        await page.keyboard.type(text, delay=20)

    elif action_type == "wait":
        ms = action.get("ms", 1000)
        print(f"    wait {ms}ms")
        await asyncio.sleep(ms / 1000)

    elif action_type == "screenshot":
        print("    screenshot (model requested)")

    else:
        print(f"    [warn] unrecognised action: {action_type}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def process_response(client: OpenAI, response, page, log: list) -> tuple[Any, list[str]]:
    """Run the screenshot→action→screenshot loop until done or capped."""
    model_messages: list[str] = []
    for iteration in range(MAX_ITERATIONS):
        if not response.output:
            print("  No output from model.")
            break

        response_id = response.id
        print(f"\n── Iteration {iteration + 1} ── response={response_id}")

        # Surface text & reasoning
        for item in response.output:
            if hasattr(item, "type"):
                if item.type == "message" and hasattr(item, "content"):
                    for block in item.content:
                        if hasattr(block, "text"):
                            print(f"  Model: {block.text}")
                            model_messages.append(block.text)
                            log.append({"iteration": iteration + 1, "model_text": block.text})
                if item.type == "reasoning" and hasattr(item, "summary") and item.summary:
                    for s in item.summary:
                        if hasattr(s, "text") and s.text.strip():
                            print(f"  Reasoning: {s.text}")

        # Find computer_call
        computer_calls = [i for i in response.output if hasattr(i, "type") and i.type == "computer_call"]
        if not computer_calls:
            print("  No computer_call → task complete or model chose to stop.")
            break

        computer_call = computer_calls[0]
        call_id = computer_call.call_id
        actions = computer_call.actions

        # Safety checks
        acknowledged_checks: list = []
        if hasattr(computer_call, "pending_safety_checks") and computer_call.pending_safety_checks:
            pending = computer_call.pending_safety_checks
            print("\n  ⚠ Safety checks required:")
            for chk in pending:
                print(f"    - {chk.code}: {chk.message}")
            confirm = input("  Proceed? (y/n): ").strip().lower()
            if confirm != "y":
                print("  Cancelled by user.")
                break
            acknowledged_checks = pending

        # Execute actions
        print(f"  Executing {len(actions)} action(s):")
        try:
            await page.bring_to_front()
            for action in actions:
                await handle_action(page, action)

                # Check for new tabs after clicks
                if action.get("type") == "click":
                    await asyncio.sleep(1.5)
                    all_pages = page.context.pages
                    if len(all_pages) > 1:
                        newest = all_pages[-1]
                        if newest != page and is_allowed_url(newest.url):
                            print(f"    Switching to new tab: {newest.url}")
                            page = newest
                        elif newest != page:
                            print(f"    [blocked] new tab navigated to disallowed URL: {newest.url}")
                            await newest.close()
                elif action.get("type") != "wait":
                    await asyncio.sleep(0.5)
        except Exception as exc:
            print(f"  [error] executing action: {exc}")

        # Block if page navigated outside allowed origin
        if not is_allowed_url(page.url):
            print(f"  [blocked] page navigated to {page.url} — outside allowed origin")
            await page.goto(TARGET_URL, wait_until="domcontentloaded")

        # Screenshot
        screenshot_b64 = await take_screenshot(page)
        log.append({"iteration": iteration + 1, "actions": [a if isinstance(a, dict) else str(a) for a in actions]})

        # Next turn
        input_content: list[dict] = [
            {
                "type": "computer_call_output",
                "call_id": call_id,
                "output": {
                    "type": "computer_screenshot",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                    "detail": "original",
                },
            }
        ]
        if acknowledged_checks:
            input_content[0]["acknowledged_safety_checks"] = [
                {"id": c.id, "code": c.code, "message": c.message} for c in acknowledged_checks
            ]

        try:
            response = client.responses.create(
                model=MODEL,
                previous_response_id=response_id,
                tools=[{"type": "computer"}],
                input=input_content,
            )
        except Exception as exc:
            print(f"  [error] API call failed: {exc}")
            break

    else:
        print(f"\n  Reached max iterations ({MAX_ITERATIONS}). Stopping.")

    return page, model_messages


async def run_task(client: OpenAI, page, task: str, label: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Run a single natural-language task and return the action log."""
    log: list = [{"task": task, "label": label, "started": datetime.now(timezone.utc).isoformat()}]

    screenshot_b64 = await take_screenshot(page)
    print(f"\nSending task to {MODEL}: {task}")

    response = client.responses.create(
        model=MODEL,
        tools=[{"type": "computer"}],
        instructions=(
            "You are an AI agent testing a legal RAG web application running on localhost. "
            "You can control a browser via keyboard and mouse. "
            "After each action you will receive a screenshot to evaluate the result. "
            "When the task is complete, stop and describe the outcome. "
            f"Stay within {ALLOWED_ORIGIN} — do not navigate to external sites."
        ),
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": task},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot_b64}",
                        "detail": "original",
                    },
                ],
            }
        ],
        reasoning={"summary": "concise"},
    )

    page, model_messages = await process_response(client, response, page, log)
    log.append({"finished": datetime.now(timezone.utc).isoformat()})
    result = {
        "label": label,
        "task": task,
        "messages": model_messages,
        "final_message": model_messages[-1] if model_messages else "",
        "log": log,
    }
    return page, result


async def run_suite(client: OpenAI, page, suite_name: str) -> tuple[Any, dict[str, Any]]:
    scenarios = SUITES[suite_name]
    scenario_results: list[dict[str, Any]] = []

    for index, scenario in enumerate(scenarios, start=1):
        print(f"\n{'=' * 72}\nScenario {index}/{len(scenarios)}: {scenario.title}\n{'=' * 72}")
        task = build_scenario_task(scenario)
        result: dict[str, Any] | None = None
        parsed_result: dict[str, Any] | None = None
        attempts_used = 0

        for attempt_index in range(SCENARIO_PARSE_RETRIES + 1):
            attempts_used = attempt_index + 1
            if attempt_index > 0:
                print(f"  [retry] Scenario did not return parseable JSON; retrying attempt {attempts_used}/{SCENARIO_PARSE_RETRIES + 1}.")

            await reset_app_state(page)
            attempt_task = task
            if attempt_index > 0:
                attempt_task = (
                    f"{task}\n\nIMPORTANT RETRY INSTRUCTION: This is retry attempt {attempts_used}. "
                    "As soon as you have enough evidence, stop and return the required JSON fenced block immediately. "
                    "Do not spend extra turns reformatting or rechecking once the requested verifications are complete."
                )

            page, result = await run_task(client, page, attempt_task, label=scenario.title)
            parsed_result = extract_structured_result(result)
            if parsed_result:
                break

        if result is None:
            raise RuntimeError(f"Scenario {scenario.key} produced no result")

        normalized_issues = [normalize_issue(issue, scenario) for issue in parsed_result.get("issues", [])] if parsed_result else []
        if not parsed_result:
            normalized_issues.append(build_parse_failure_issue(result, scenario, attempts_used))

        scenario_results.append(
            {
                "scenario": {
                    "key": scenario.key,
                    "title": scenario.title,
                    "source_filter": scenario.source_filter,
                    "questions": scenario.questions,
                },
                "attemptCount": attempts_used,
                "raw_result": result,
                "parsed_result": parsed_result,
                "issues": normalized_issues,
            }
        )

    summary = aggregate_suite_results(suite_name, scenario_results)
    return page, {"suite": suite_name, "summary": summary, "scenarios": scenario_results}


async def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-5.4 Computer Use local test harness")
    parser.add_argument("--task", type=str, default=None, help="Single task to perform (otherwise interactive)")
    parser.add_argument("--suite", choices=sorted(SUITES.keys()), default=None, help="Run a built-in multi-scenario suite")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    args = parser.parse_args()

    if args.task and args.suite:
        print("ERROR: Use either --task or --suite, not both.")
        sys.exit(1)

    if not ENDPOINT:
        print(
            "ERROR: Set COMPUTER_USE_AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_ENDPOINT to your Azure OpenAI resource endpoint.\n"
            "  Example: export COMPUTER_USE_AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com"
        )
        sys.exit(1)

    # Azure credential → bearer token
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")

    client = OpenAI(
        base_url=f"{ENDPOINT.rstrip('/')}/openai/v1/",
        api_key=token_provider,
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=args.headless,
            args=[f"--window-size={DISPLAY_WIDTH},{DISPLAY_HEIGHT}", "--disable-extensions"],
        )
        context = await browser.new_context(
            viewport={"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
            accept_downloads=False,
        )
        page = await context.new_page()

        await page.goto(TARGET_URL, wait_until="domcontentloaded")
        print(f"Browser opened → {TARGET_URL}")

        all_logs: list = []

        try:
            if args.task:
                # Single-task mode
                await reset_app_state(page)
                page, result = await run_task(client, page, build_single_task(args.task))
                all_logs.append(result)
            elif args.suite:
                page, suite_result = await run_suite(client, page, args.suite)
                all_logs.append(suite_result)
                summary = suite_result["summary"]
                print("\nSuite summary:")
                print(json.dumps(summary, indent=2))
            else:
                # Interactive mode
                print(
                    textwrap.dedent(
                        """
                    ═══════════════════════════════════════════════════
                    Computer Use Test Harness — Interactive Mode
                    Type a task for the model, or 'exit' to quit.

                    Example tasks:
                      Ask 'What is the dental plan?' and report the answer
                      Open developer settings and change retrieval mode to vectors
                      Search for annual eye exam coverage
                    ═══════════════════════════════════════════════════
                    """
                    )
                )
                while True:
                    task = input("\nTask> ").strip()
                    if task.lower() in ("exit", "quit", "q"):
                        break
                    if not task:
                        continue

                    page, result = await run_task(client, page, build_single_task(task))
                    all_logs.append(result)

                    # Reset to home for next task
                    await page.goto(TARGET_URL, wait_until="domcontentloaded")

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            # Save logs
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path = LOG_DIR / f"run_{ts}.json"
            with open(log_path, "w") as f:
                json.dump(all_logs, f, indent=2, default=str)
            print(f"\nSession log saved to {log_path}")

            await context.close()
            await browser.close()
            print("Browser closed.")


if __name__ == "__main__":
    asyncio.run(main())
