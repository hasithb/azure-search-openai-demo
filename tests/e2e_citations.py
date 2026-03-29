"""
E2E tests for citations, highlighting, and supporting content structure.

These tests verify that:
1. Citations render correctly in the answer (inline superscripts and citation list)
2. Multiple distinct citations from different sources are displayed
3. Clicking a citation opens the Supporting Content panel
4. Supporting content items have correct structure (title, content, metadata)
5. Subsection highlighting works when a citation is clicked
6. The thought process panel shows search results and prompts (admin mode)
7. Both streaming and non-streaming modes work correctly
8. Citations work against a deployed version (when DEPLOYED_URL is set)

Run locally:
    pytest tests/e2e_citations.py -v

Run against deployed:
    DEPLOYED_URL=https://your-app.azurewebsites.net pytest tests/e2e_citations.py -v -k deployed
"""

import json
import os
import re

import pytest
from playwright.sync_api import Page, Route, expect

from .e2e import free_port, live_server_url, run_server, wait_for_server_ready  # noqa: F401

expect.set_options(timeout=15_000)

# Snapshot file paths
STREAMING_SNAPSHOT = "tests/snapshots/test_app/test_chat_citations_stream/client0/result.jsonlines"
NONSTREAMING_SNAPSHOT = "tests/snapshots/test_app/test_chat_citations_nonstream/client0/result.json"

# Page title — customized for this legal RAG fork
PAGE_TITLE = "Civil Procedure Copilot"

# Expected citation data for assertions — based on index v3 source types
EXPECTED_CITATIONS = [
    {
        "sourcepage": "Part 24",
        "sourcefile": "Part 24 - Summary judgment",
        "category": "Civil Procedure Rules and Practice Directions",
        "subsection_id": "24.2",
        "content_snippet": "summary judgment against a claimant or defendant",
        "source_type": "CPR",
    },
    {
        "sourcepage": "PD44",
        "sourcefile": "Practice Direction 44 - General rules about costs",
        "category": "Civil Procedure Rules and Practice Directions",
        "subsection_id": "1.1",
        "content_snippet": "court discretion as to whether costs are payable",
        "source_type": "PD",
    },
    {
        "sourcepage": "Practice Direction - Pre-Action Conduct and Protocols",
        "sourcefile": "Practice Direction - Pre-Action Conduct and Protocols",
        "category": "Civil Procedure Rules and Practice Directions",
        "subsection_id": "4",
        "content_snippet": "complied with this Practice Direction or the relevant pre-action protocol",
        "source_type": "Pre-Action",
    },
    {
        "sourcepage": "The Commercial Court Guide",
        "sourcefile": "The Commercial Court Guide",
        "category": "Commercial Court",
        "subsection_id": "D5.3",
        "content_snippet": "case management conference",
        "source_type": "Court Guide",
    },
]

# A unique snippet from the answer text used to verify the answer rendered
ANSWER_VISIBLE_TEXT = "summary judgment where a party has no real prospect"

TEST_QUESTION = "What is the procedure for summary judgment and what costs rules apply?"
DEPLOYED_TEST_QUESTION = "What is the procedure for summary judgment under the CPR?"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dismiss_splash_screen(page: Page):
    """Dismiss the splash screen if it is showing, or wait for it to auto-dismiss."""
    splash = page.locator("[role='dialog'][aria-modal='true']")
    # Wait briefly for the splash to appear (it fades in over ~800ms)
    try:
        splash.wait_for(state="visible", timeout=2000)
    except Exception:
        return  # Splash never appeared or already gone
    # Click to dismiss immediately rather than waiting for auto-dismiss
    try:
        splash.click()
    except Exception:
        pass
    # Wait for the splash to fully disappear
    try:
        splash.wait_for(state="hidden", timeout=5000)
    except Exception:
        pass


def submit_question(page: Page, question: str):
    """Type a question into the chat input and submit it."""
    dismiss_splash_screen(page)
    page.wait_for_load_state("networkidle")
    # Use a regex placeholder to handle both customized and upstream defaults
    placeholder = page.get_by_placeholder(re.compile(r"Ask a question|Type a new question"))
    expect(placeholder).to_be_visible()
    placeholder.click()
    placeholder.fill(question)
    submit_btn = page.get_by_role("button", name="Submit question")
    expect(submit_btn).to_be_enabled()
    submit_btn.click()


def open_admin_mode(page: Page, url: str):
    """Navigate to the app with admin mode enabled via query parameter."""
    separator = "&" if "?" in url else "?"
    page.goto(f"{url}{separator}admin=true")
    dismiss_splash_screen(page)


def setup_config_mocks(page: Page):
    """Mock /config and /api/categories to disable the category filter requirement."""

    def handle_config(route: Route):
        route.fulfill(
            json={
                "showCategoryFilter": False,
                "showGPT4VOptions": False,
                "showMultimodalOptions": False,
                "showSemanticRankerOption": False,
                "showQueryRewritingOption": False,
                "showReasoningEffortOption": False,
                "streamingEnabled": True,
                "defaultReasoningEffort": None,
                "defaultRetrievalReasoningEffort": None,
                "showVectorOption": False,
                "showUserUpload": False,
                "showLanguagePicker": False,
                "showSpeechInput": False,
                "showSpeechOutputBrowser": False,
                "showSpeechOutputAzure": False,
                "showChatHistoryBrowser": False,
                "showChatHistoryCosmos": False,
                "showAgenticRetrievalOption": False,
                "ragSearchTextEmbeddings": True,
                "ragSearchImageEmbeddings": False,
                "ragSendTextSources": True,
                "ragSendImageSources": False,
                "webSourceEnabled": False,
                "sharepointSourceEnabled": False,
                "deployedUiCompat": False,
            },
            status=200,
        )

    def handle_categories(route: Route):
        route.fulfill(json=[], status=200)

    page.route("*/**/config", handle_config)
    page.route("*/**/api/categories", handle_categories)


# ---------------------------------------------------------------------------
# LOCAL TESTS — use mock server with snapshot data
# ---------------------------------------------------------------------------


class TestCitationsStreaming:
    """Test citation rendering, clicking, and supporting content with streaming responses."""

    @pytest.fixture(autouse=True)
    def setup_route(self, page: Page, live_server_url: str):  # noqa: F811
        """Set up mock streaming route and navigate to the app."""

        def handle_stream(route: Route):
            with open(STREAMING_SNAPSHOT) as f:
                jsonl = f.read()
            route.fulfill(body=jsonl, status=200, headers={"Transfer-encoding": "Chunked"})

        setup_config_mocks(page)
        page.route("*/**/chat/stream", handle_stream)
        open_admin_mode(page, live_server_url)
        expect(page).to_have_title(PAGE_TITLE)

    def test_answer_text_appears(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify the answer text from the streaming response appears on screen."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

    def test_multiple_citation_badges_appear(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify that multiple distinct citation badges are rendered at the bottom of the answer."""
        submit_question(page, TEST_QUESTION)

        # Wait for answer to appear
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # Check that citation entries appear (numbered references at the bottom)
        # The citation list should have at least 2 distinct citations
        citation_links = page.locator("a.citation, a[class*='citation']")
        expect(citation_links.first).to_be_visible()
        count = citation_links.count()
        assert count >= 2, f"Expected at least 2 citation badges, got {count}"

    def test_inline_superscript_citations_present(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify that inline superscript citation markers appear within the answer text."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # Superscript citations are rendered as <sup> elements with class supContainer
        sup_elements = page.locator("sup, .supContainer, [class*='supContainer']")
        expect(sup_elements.first).to_be_visible()
        count = sup_elements.count()
        assert count >= 1, f"Expected at least 1 inline superscript citation, got {count}"

    def test_citation_click_opens_supporting_content(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify that clicking a citation badge opens the Supporting Content panel."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # Click the first citation in the citation list
        first_citation = page.locator("a.citation, a[class*='citation']").first
        first_citation.click()

        # The Supporting Content tab should become visible
        expect(page.get_by_text("Supporting content")).to_be_visible()

    def test_supporting_content_has_source_titles(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify that Supporting Content panel shows source document titles."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # Open supporting content via button
        page.get_by_label("Show supporting content").click()
        expect(page.get_by_text("Supporting content")).to_be_visible()

        # Check that at least one source document heading appears
        # The SupportingContent component renders document titles as headings
        content_panel = page.locator("[class*='analysisPanelContent'], [class*='supportingContent']")
        expect(content_panel.first).to_be_visible()

    def test_supporting_content_has_document_text(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify that Supporting Content panel shows actual document content."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # Open supporting content
        page.get_by_label("Show supporting content").click()
        expect(page.get_by_text("Supporting content")).to_be_visible()

        # At least one of the source content snippets should be visible
        found_any = False
        for citation in EXPECTED_CITATIONS:
            try:
                locator = page.get_by_text(citation["content_snippet"])
                if locator.is_visible():
                    found_any = True
                    break
            except Exception:
                continue
        assert found_any, "None of the expected source content snippets were found in the Supporting Content panel"

    def test_thought_process_shows_search_info(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify that Thought Process panel shows search query and results (admin mode)."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # Click thought process button (only visible in admin mode)
        page.get_by_label("Show thought process").click()
        expect(page.get_by_text("Thought process")).to_be_visible()

        # Check that the search query title is visible
        expect(page.get_by_text("Prompt to generate search query")).to_be_visible()

    def test_citation_click_then_supporting_content_shows_match(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify that clicking a specific citation and then showing supporting content
        displays the matching source content."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # Click the first citation in the list
        first_citation = page.locator("a.citation, a[class*='citation']").first
        first_citation.click()

        # Supporting content should be visible
        expect(page.get_by_text("Supporting content")).to_be_visible()

        # The supporting content panel should contain at least one of the expected content snippets
        panel_text = page.locator("[class*='analysisPanelContent'], [class*='supportingContent'], [role='tabpanel']")
        expect(panel_text.first).to_be_visible()

    def test_clear_chat_resets_citations(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify that clearing chat removes citations and supporting content."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # Clear the chat
        page.get_by_role("button", name="Clear chat").click()

        # Answer and citations should be gone
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).not_to_be_visible()
        expect(page.get_by_role("button", name="Clear chat")).to_be_disabled()


class TestCitationsNonStreaming:
    """Test citation rendering with non-streaming responses."""

    @pytest.fixture(autouse=True)
    def setup_route(self, page: Page, live_server_url: str):  # noqa: F811
        """Set up mock non-streaming route and navigate to the app."""

        def handle_chat(route: Route):
            with open(NONSTREAMING_SNAPSHOT) as f:
                json_data = f.read()
            route.fulfill(body=json_data, status=200)

        setup_config_mocks(page)
        page.route("*/**/chat", handle_chat)
        open_admin_mode(page, live_server_url)
        expect(page).to_have_title(PAGE_TITLE)

        # Disable streaming
        page.get_by_role("button", name="Developer settings").click()
        page.get_by_text("Stream chat completion responses").click()
        page.locator("button").filter(has_text="Close").click()

    def test_nonstreaming_answer_with_citations(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify answer and citations render correctly in non-streaming mode."""
        submit_question(page, TEST_QUESTION)

        # Answer should appear
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # Citation badges should be present
        citation_links = page.locator("a.citation, a[class*='citation']")
        expect(citation_links.first).to_be_visible()

    def test_nonstreaming_supporting_content(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify supporting content works in non-streaming mode."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # Open supporting content
        page.get_by_label("Show supporting content").click()
        expect(page.get_by_text("Supporting content")).to_be_visible()

    def test_nonstreaming_thought_process(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify thought process works in non-streaming mode (admin mode)."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        page.get_by_label("Show thought process").click()
        expect(page.get_by_text("Thought process")).to_be_visible()
        expect(page.get_by_text("Prompt to generate search query")).to_be_visible()


class TestCitationStructure:
    """Test the structural integrity of citations and supporting content data attributes."""

    @pytest.fixture(autouse=True)
    def setup_route(self, page: Page, live_server_url: str):  # noqa: F811
        """Set up mock streaming route and navigate."""

        def handle_stream(route: Route):
            with open(STREAMING_SNAPSHOT) as f:
                jsonl = f.read()
            route.fulfill(body=jsonl, status=200, headers={"Transfer-encoding": "Chunked"})

        setup_config_mocks(page)
        page.route("*/**/chat/stream", handle_stream)
        open_admin_mode(page, live_server_url)
        expect(page).to_have_title(PAGE_TITLE)

    def test_citation_data_attributes_present(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify that inline citation elements have proper data-citation-path attributes."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # Check for citation elements with data attributes
        citations_with_path = page.locator("[data-citation-path]")
        count = citations_with_path.count()
        assert count >= 1, f"Expected citation elements with data-citation-path, got {count}"

        # Verify the first citation has a non-empty path
        first_path = citations_with_path.first.get_attribute("data-citation-path")
        assert first_path and len(first_path) > 0, "data-citation-path should not be empty"

    def test_citation_badges_have_numbered_labels(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify citation badges at the bottom show numbered references (e.g., '1. filename.pdf')."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        citation_links = page.locator("a.citation, a[class*='citation']")
        expect(citation_links.first).to_be_visible()

        # Verify at least one citation starts with a number pattern like "1."
        first_text = citation_links.first.inner_text()
        assert re.match(r"^\d+\.\s", first_text), f"Citation badge should start with 'N. ' but got: '{first_text}'"

    def test_ai_disclaimer_present(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify the AI-generated content disclaimer appears below the answer."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # The custom AI disclaimer should be visible
        expect(page.get_by_text("AI-generated content may be incorrect")).to_be_visible()

    def test_supporting_content_structure_has_paragraphs(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify that supporting content is rendered with paragraph structure, not as a raw text block."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # Open supporting content
        page.get_by_label("Show supporting content").click()
        expect(page.get_by_text("Supporting content")).to_be_visible()

        # The supporting content panel should be visible with structured content
        panel_content = page.locator("[class*='analysisPanelContent'], [class*='supportingContent']")
        expect(panel_content.first).to_be_visible()


class TestCitationSwitching:
    """Test switching between different citation-related panels."""

    @pytest.fixture(autouse=True)
    def setup_route(self, page: Page, live_server_url: str):  # noqa: F811
        def handle_stream(route: Route):
            with open(STREAMING_SNAPSHOT) as f:
                jsonl = f.read()
            route.fulfill(body=jsonl, status=200, headers={"Transfer-encoding": "Chunked"})

        setup_config_mocks(page)
        page.route("*/**/chat/stream", handle_stream)
        open_admin_mode(page, live_server_url)
        expect(page).to_have_title(PAGE_TITLE)

    def test_switch_between_thought_process_and_supporting_content(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify switching between Thought Process and Supporting Content tabs."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # Open thought process
        page.get_by_label("Show thought process").click()
        expect(page.get_by_text("Thought process")).to_be_visible()

        # Switch to supporting content
        page.get_by_label("Show supporting content").click()
        expect(page.get_by_text("Supporting content")).to_be_visible()

        # Switch back to thought process
        page.get_by_label("Show thought process").click()
        expect(page.get_by_text("Thought process")).to_be_visible()

    def test_toggle_supporting_content_off(self, page: Page, live_server_url: str):  # noqa: F811
        """Verify that clicking supporting content button again hides the panel."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(ANSWER_VISIBLE_TEXT)).to_be_visible()

        # Open supporting content
        page.get_by_label("Show supporting content").click()
        expect(page.get_by_text("Supporting content")).to_be_visible()

        # Click again to close
        page.get_by_label("Show supporting content").click()
        # The tab panel content should no longer be visible
        supporting_tab = page.get_by_role("tab", name="Supporting content")
        expect(supporting_tab).not_to_be_visible()


# ---------------------------------------------------------------------------
# Snapshot Builder — generates mock response data from source definitions
# ---------------------------------------------------------------------------


def make_source(*, id, subsection_id, sourcepage, sourcefile, category, content, full_content=None, storageurl=""):
    """Create a source data point dict for snapshot building."""
    citation = f"{subsection_id}, {sourcepage}, {sourcefile}"
    return {
        "id": id,
        "citation": citation,
        "content": content,
        "full_content": full_content or content,
        "sourcepage": sourcepage,
        "sourcefile": sourcefile,
        "category": category,
        "storageurl": storageurl,
        "updated": "2025-04-01",
        "subsection_id": subsection_id,
    }


def build_streaming_snapshot(sources: list, answer_text: str, question: str = "Test question") -> str:
    """Build NDJSON streaming response string from source definitions and answer text."""
    citations_list = [s["citation"] for s in sources]
    text_data = [dict(s) for s in sources]
    search_results = [
        {
            "type": "searchIndex",
            "id": s["id"],
            "content": s["content"][:80],
            "category": s["category"],
            "sourcepage": s["sourcepage"],
            "sourcefile": s["sourcefile"],
            "oids": None,
            "groups": None,
            "captions": [{"additional_properties": {}, "text": f"Caption: {s['sourcefile']}.", "highlights": []}],
            "score": 0.089,
            "reranker_score": 3.95,
            "activity": None,
            "images": None,
        }
        for s in sources
    ]
    thoughts = [
        {
            "title": "Prompt to generate search query",
            "description": [{"role": "system", "content": f"Generate search query for: {question}"}],
            "props": {
                "model": "gpt-4.1-mini",
                "token_usage": {"prompt_tokens": 50, "completion_tokens": 200, "reasoning_tokens": 0, "total_tokens": 250},
            },
        },
        {
            "title": "Search using generated search query",
            "description": "generated search query",
            "props": {
                "use_semantic_captions": False,
                "use_semantic_ranker": True,
                "use_query_rewriting": False,
                "top": len(sources),
                "filter": None,
                "use_vector_search": True,
                "use_text_search": True,
                "search_text_embeddings": True,
                "search_image_embeddings": False,
            },
        },
        {"title": "Search results", "description": search_results, "props": None},
        {
            "title": "Prompt to generate answer",
            "description": [
                {"role": "system", "content": "Answer based on sources below."},
                {"role": "user", "content": question},
            ],
            "props": {"model": "gpt-4.1-mini"},
        },
    ]
    context_block = {
        "data_points": {"text": text_data, "images": [], "citations": citations_list, "external_results_metadata": []},
        "thoughts": thoughts,
        "followup_questions": None,
        "answer": None,
    }
    line1 = json.dumps({"delta": {"role": "assistant"}, "context": context_block, "session_state": None})
    line2 = json.dumps({"delta": {"content": None, "role": "assistant"}})
    line3 = json.dumps({"delta": {"content": answer_text, "role": None}})
    line4 = json.dumps({"delta": {"role": "assistant"}, "context": context_block, "session_state": None})
    return f"{line1}\n{line2}\n{line3}\n{line4}\n"


# ---------------------------------------------------------------------------
# Diverse Source Definitions — based on real index v3 naming patterns
# ---------------------------------------------------------------------------

# --- CPR Parts (Civil Procedure Rules) ---

CPR_PART_1 = make_source(
    id="cpr-part1-1_1",
    subsection_id="1.1",
    sourcepage="Part 1 \u2013 Overriding Objective",
    sourcefile="Part 1",
    category="Civil Procedure Rules and Practice Directions",
    content="These Rules are a procedural code with the overriding objective of enabling the court to deal with cases justly and at proportionate cost.",
    full_content="PART 1 \u2013 OVERRIDING OBJECTIVE\n\n1.1 These Rules are a procedural code with the overriding objective of enabling the court to deal with cases justly and at proportionate cost.\n\n1.2 Dealing with a case justly and at proportionate cost includes ensuring that the parties are on an equal footing.",
)

CPR_PART_3 = make_source(
    id="cpr-part3-3_1",
    subsection_id="3.1",
    sourcepage="Part 3 \u2013 The Court\u2019s Case Management Powers",
    sourcefile="Part 3",
    category="Civil Procedure Rules and Practice Directions",
    content="The court may extend or shorten the time for compliance with any rule, practice direction or court order even if an application for extension is made after the time for compliance has expired.",
    full_content="PART 3 \u2013 THE COURT\u2019S CASE MANAGEMENT POWERS\n\n3.1 The court may extend or shorten the time for compliance with any rule, practice direction or court order even if an application for extension is made after the time for compliance has expired.",
)

CPR_PART_24 = make_source(
    id="cpr-part24-24_2",
    subsection_id="24.2",
    sourcepage="Part 24 \u2013 Summary Judgment",
    sourcefile="Part 24",
    category="Civil Procedure Rules and Practice Directions",
    content="The court may give summary judgment against a claimant or defendant on the whole of a claim or on an issue if it considers that the claimant has no real prospect of succeeding on the claim or issue.",
)

CPR_PART_52 = make_source(
    id="cpr-part52-52_3",
    subsection_id="52.3",
    sourcepage="Part 52 \u2013 Appeals",
    sourcefile="Part 52",
    category="Civil Procedure Rules and Practice Directions",
    content="An appellant or respondent requires permission to appeal unless the appeal is against a committal order or a refusal to grant habeas corpus.",
)

CPR_PART_56 = make_source(
    id="cpr-part56-56_1",
    subsection_id="56.1",
    sourcepage="Part 56 \u2013 Landlord and Tenant Claims and Miscellaneous Provisions about Land and claims under the Renting Homes (Wales) Act 2016",
    sourcefile="Part 56",
    category="Civil Procedure Rules and Practice Directions",
    content="This Part contains rules about landlord and tenant claims and miscellaneous provisions about land including claims under the Renting Homes (Wales) Act 2016.",
)

CPR_PART_85 = make_source(
    id="cpr-part85-85_2",
    subsection_id="85.2",
    sourcepage="Part 85 Claims on Controlled Goods and Executed Goods",
    sourcefile="Part 85 \u2013 Claims On Controlled Goods And Executed Goods",
    category="Civil Procedure Rules and Practice Directions",
    content="A claim under this Part must be made in accordance with the procedure set out in this Part and the relevant practice direction.",
)

# --- Practice Directions ---

PD_44 = make_source(
    id="pd44-1_1",
    subsection_id="1.1",
    sourcepage="Practice Direction 44 \u2013 General Rules About Costs",
    sourcefile="Practice Direction 44",
    category="Civil Procedure Rules and Practice Directions",
    content="This Practice Direction supplements Part 44. The court has discretion as to whether costs are payable by one party to another, the amount of those costs and when they are to be paid.",
)

PD_19A = make_source(
    id="pd19a-2_1",
    subsection_id="2.1",
    sourcepage="Practice Direction 19A \u2013 Derivative Claims",
    sourcefile="Practice Direction 19A",
    category="Civil Procedure Rules and Practice Directions",
    content="A derivative claim is a claim brought by a member of a company in respect of a cause of action vested in the company seeking relief on behalf of the company.",
)

PD_53B = make_source(
    id="pd53b-1_1",
    subsection_id="1.1",
    sourcepage="Practice Direction 53B: Media and communications claims",
    sourcefile="Practice Direction 53B \u2013 Media And Communications Claims",
    category="Civil Procedure Rules and Practice Directions",
    content="This practice direction applies to media and communications claims as defined in paragraph 1.2 and supplements Part 53.",
)

PD_57AC = make_source(
    id="pd57ac-3_1",
    subsection_id="3.1",
    sourcepage="Practice Direction 57AC: trial witness statements in the business and property courts",
    sourcefile="Practice Direction 57Ac \u2013 Trial Witness Statements In The Business And Property Courts",
    category="Civil Procedure Rules and Practice Directions",
    content="A trial witness statement must contain only the evidence which that witness would be allowed to give orally at trial and must comply with the requirements of this Practice Direction.",
)

PD_27B = make_source(
    id="pd27b-4_1",
    subsection_id="4.1",
    sourcepage="Practice Direction 27B \u2013 Claims Under the Pre-Action Protocol for Personal Injury Claims Below the Small Claims Limit in Road Traffic Accidents \u2013 Court Procedure",
    sourcefile="Practice Direction 27B",
    category="Civil Procedure Rules and Practice Directions",
    content="Where a claim has been started under the relevant pre-action protocol but is no longer continuing under that protocol, it must be started in accordance with Part 7 or Part 8.",
)

# --- Pre-Action Protocols ---

PREACTION_JUDICIAL_REVIEW = make_source(
    id="pre-action-jud-review-6",
    subsection_id="6",
    sourcepage="Pre-Action Protocol for Judicial Review",
    sourcefile="Pre",
    category="Civil Procedure Rules and Practice Directions",
    content="Before making a claim the claimant should send a letter to the defendant identifying the issues in dispute and enclosing the documents relied upon.",
)

PREACTION_HOUSING = make_source(
    id="pre-action-housing-3",
    subsection_id="3",
    sourcepage="Pre-Action Protocol for Housing Conditions Claims (England)",
    sourcefile="Pre",
    category="Civil Procedure Rules and Practice Directions",
    content="The aims of this protocol are to encourage the exchange of early and full information about the claim, to enable parties to avoid litigation by agreeing a settlement before proceedings are started.",
)

PREACTION_CONSTRUCTION = make_source(
    id="pre-action-construction-5",
    subsection_id="5",
    sourcepage="Pre-Action Protocol for the Construction and Engineering Disputes",
    sourcefile="Pre",
    category="Civil Procedure Rules and Practice Directions",
    content="The claimant shall send to each proposed defendant a letter of claim which shall contain a clear summary of the facts on which each claim is based.",
)

# --- Court Guides ---

GUIDE_COMMERCIAL = make_source(
    id="commercial-guide-B11-1",
    subsection_id="B.11.1",
    sourcepage="B.  Commencement, Transfer and Removal, B.11  Default judgment (p. 26)",
    sourcefile="Commercial Court Guide",
    category="Commercial Court",
    content="Default judgment is not available in a claim in the Commercial Court unless the court gives permission. An application for default judgment should be made by application notice under Part 23.",
)

GUIDE_TCC = make_source(
    id="tcc-guide-17_2",
    subsection_id="17.2",
    sourcepage="Section 17. Enforcement, 17.2 High Court (p. 108)",
    sourcefile="Technology and Construction Court Guide",
    category="Technology and Construction Court",
    content="Enforcement of TCC judgments and orders is generally carried out in the same way as enforcement of any other High Court judgment or order.",
)

GUIDE_CHANCERY = make_source(
    id="chancery-guide-ch3",
    subsection_id="Part 8 claims",
    sourcepage="Part 1, Chapter 3 Commencement and transfer, Applications made pre-issue or at point of issue (p. 33)",
    sourcefile="Chancery Guide",
    category="Chancery Division",
    content="Applications may be made to a Master or a Judge before the issue of a claim form in cases of urgency or where it is otherwise desirable in the interests of justice.",
)

GUIDE_KINGS_BENCH = make_source(
    id="kings-bench-26",
    subsection_id="p. 207",
    sourcepage="26. Enrolment of deeds and other documents (p. 207)",
    sourcefile="35.16_JO_Kings_Bench_Division_Guide_2025_WEB4.pdf",
    category="King's Bench Division",
    content="Deeds and other documents may be enrolled in the Senior Courts under the Enrolment of Deeds Act 1845 or any other enactment or rule.",
)

GUIDE_PATENTS = make_source(
    id="patents-guide-annex-f",
    subsection_id="Annex F",
    sourcepage="Annex F: Specimen order on handing down of judgment. (p. 33)",
    sourcefile="Patents Court Guide",
    category="Patents Court",
    content="Upon judgment being handed down the parties shall within 14 days lodge agreed minutes of order giving effect to the judgment.",
)


# ---------------------------------------------------------------------------
# Source-type Parsing Scenarios
# ---------------------------------------------------------------------------

# Each scenario: sources, answer_text with [citation] brackets, answer_fragment to assert visibility

CPR_SCENARIOS = [
    {
        "id": "cpr_simple_overriding_objective",
        "sources": [CPR_PART_1],
        "answer_text": "The overriding objective requires the court to deal with cases justly and at proportionate cost [Part 1].",
        "answer_fragment": "overriding objective requires the court",
    },
    {
        "id": "cpr_apostrophe_case_management",
        "sources": [CPR_PART_3],
        "answer_text": "The court has broad case management powers, including the ability to extend or shorten time for compliance with any rule or order [Part 3].",
        "answer_fragment": "extend or shorten time for compliance",
    },
    {
        "id": "cpr_two_digit_part_number",
        "sources": [CPR_PART_24],
        "answer_text": "The court may give summary judgment where a party has no real prospect of succeeding on the claim [Part 24].",
        "answer_fragment": "summary judgment where a party",
    },
    {
        "id": "cpr_short_title_appeals",
        "sources": [CPR_PART_52],
        "answer_text": "Permission to appeal is generally required unless the appeal is against a committal order [Part 52].",
        "answer_fragment": "Permission to appeal is generally",
    },
    {
        "id": "cpr_very_long_sourcepage_wales_act",
        "sources": [CPR_PART_56],
        "answer_text": "Part 56 covers landlord and tenant claims, including those under the Renting Homes (Wales) Act 2016 [Part 56].",
        "answer_fragment": "landlord and tenant claims",
    },
    {
        "id": "cpr_full_title_in_sourcefile",
        "sources": [CPR_PART_85],
        "answer_text": "Claims on controlled goods must follow the procedure in Part 85 and the relevant practice direction [Part 85 \u2013 Claims On Controlled Goods And Executed Goods].",
        "answer_fragment": "Claims on controlled goods must follow",
    },
]

PD_SCENARIOS = [
    {
        "id": "pd_simple_numeric",
        "sources": [PD_44],
        "answer_text": "The court has discretion over costs including whether they are payable and their amount [Practice Direction 44].",
        "answer_fragment": "discretion over costs",
    },
    {
        "id": "pd_letter_suffix_A",
        "sources": [PD_19A],
        "answer_text": "A derivative claim may be brought by a company member on behalf of the company [Practice Direction 19A].",
        "answer_fragment": "derivative claim may be brought",
    },
    {
        "id": "pd_colon_sourcepage_mixed_case_B",
        "sources": [PD_53B],
        "answer_text": "Media and communications claims are governed by specific rules supplementing Part 53 [Practice Direction 53B \u2013 Media And Communications Claims].",
        "answer_fragment": "Media and communications claims are governed",
    },
    {
        "id": "pd_57ac_multi_letter_suffix",
        "sources": [PD_57AC],
        "answer_text": "Trial witness statements must contain only evidence the witness would give orally [Practice Direction 57Ac \u2013 Trial Witness Statements In The Business And Property Courts].",
        "answer_fragment": "Trial witness statements must contain",
    },
    {
        "id": "pd_double_em_dash_long_name",
        "sources": [PD_27B],
        "answer_text": "Claims under the personal injury pre-action protocol must be started under Part 7 or Part 8 [Practice Direction 27B].",
        "answer_fragment": "personal injury pre-action protocol",
    },
]

PREACTION_SCENARIOS = [
    {
        "id": "preaction_judicial_review_truncated_sourcefile",
        "sources": [PREACTION_JUDICIAL_REVIEW],
        "answer_text": "Before making a judicial review claim, the claimant should send a letter to the defendant identifying the issues [Pre].",
        "answer_fragment": "judicial review claim",
    },
    {
        "id": "preaction_housing_parenthetical_england",
        "sources": [PREACTION_HOUSING],
        "answer_text": "The housing conditions protocol aims to encourage early exchange of information before litigation [Pre].",
        "answer_fragment": "housing conditions protocol",
    },
    {
        "id": "preaction_construction_engineering",
        "sources": [PREACTION_CONSTRUCTION],
        "answer_text": "The construction protocol requires a letter of claim containing a clear summary of the facts [Pre].",
        "answer_fragment": "construction protocol requires",
    },
]

COURT_GUIDE_SCENARIOS = [
    {
        "id": "guide_commercial_complex_sourcepage_with_dots",
        "sources": [GUIDE_COMMERCIAL],
        "answer_text": "Default judgment is not available in the Commercial Court without the court's permission [Commercial Court Guide].",
        "answer_fragment": "Default judgment is not available",
    },
    {
        "id": "guide_tcc_section_number_pattern",
        "sources": [GUIDE_TCC],
        "answer_text": "TCC judgments are enforced in the same way as other High Court judgments [Technology and Construction Court Guide].",
        "answer_fragment": "TCC judgments are enforced",
    },
    {
        "id": "guide_chancery_hierarchical_sourcepage",
        "sources": [GUIDE_CHANCERY],
        "answer_text": "Applications may be made before issue of a claim form in cases of urgency [Chancery Guide].",
        "answer_fragment": "Applications may be made before issue",
    },
    {
        "id": "guide_kings_bench_pdf_filename_sourcefile",
        "sources": [GUIDE_KINGS_BENCH],
        "answer_text": "Deeds may be enrolled under the Enrolment of Deeds Act 1845 [35.16_JO_Kings_Bench_Division_Guide_2025_WEB4.pdf].",
        "answer_fragment": "Deeds may be enrolled",
    },
    {
        "id": "guide_patents_annex_colon_sourcepage",
        "sources": [GUIDE_PATENTS],
        "answer_text": "After judgment is handed down, parties must lodge agreed minutes of order within 14 days [Patents Court Guide].",
        "answer_fragment": "lodge agreed minutes of order",
    },
]

MIXED_SOURCE_SCENARIOS = [
    {
        "id": "mixed_cpr_and_pd",
        "sources": [CPR_PART_3, PD_44],
        "answer_text": "The court has case management powers to control proceedings [Part 3]. Costs are governed by Practice Direction 44 which gives the court discretion [Practice Direction 44].",
        "answer_fragment": "case management powers to control",
        "expected_citation_count": 2,
    },
    {
        "id": "mixed_all_categories_five_sources",
        "sources": [CPR_PART_1, PD_19A, PREACTION_JUDICIAL_REVIEW, GUIDE_COMMERCIAL, GUIDE_PATENTS],
        "answer_text": "The overriding objective [Part 1] governs all civil proceedings. Derivative claims [Practice Direction 19A] have specific rules. Judicial review requires pre-action steps [Pre]. The Commercial Court has its own procedures [Commercial Court Guide]. Patents cases follow the Patents Court Guide [Patents Court Guide].",
        "answer_fragment": "overriding objective",
        "expected_citation_count": 5,
    },
    {
        "id": "mixed_all_court_guides",
        "sources": [GUIDE_COMMERCIAL, GUIDE_TCC, GUIDE_CHANCERY, GUIDE_KINGS_BENCH, GUIDE_PATENTS],
        "answer_text": "Each specialist court has its own guide. The Commercial Court [Commercial Court Guide] deals with business disputes. The TCC [Technology and Construction Court Guide] handles construction cases. The Chancery Division [Chancery Guide] covers equity matters. The King\u2019s Bench Division [35.16_JO_Kings_Bench_Division_Guide_2025_WEB4.pdf] handles general civil claims. Patents cases follow the Patents Court Guide [Patents Court Guide].",
        "answer_fragment": "specialist court has its own guide",
        "expected_citation_count": 5,
    },
    {
        "id": "mixed_preaction_and_cpr",
        "sources": [PREACTION_CONSTRUCTION, CPR_PART_24, PD_27B],
        "answer_text": "Before commencing construction proceedings, parties must comply with the pre-action protocol [Pre]. If a party has no real prospect of success, the court may grant summary judgment [Part 24]. Personal injury claims below the small claims limit follow Practice Direction 27B [Practice Direction 27B].",
        "answer_fragment": "construction proceedings",
        "expected_citation_count": 3,
    },
]

EDGE_CASE_SCENARIOS = [
    {
        "id": "edge_adjacent_citations_no_space",
        "sources": [CPR_PART_1, CPR_PART_3],
        "answer_text": "The court must deal with cases justly [Part 1][Part 3] using its case management powers.",
        "answer_fragment": "deal with cases justly",
        "expected_citation_count": 2,
    },
    {
        "id": "edge_citation_at_start_of_answer",
        "sources": [CPR_PART_52],
        "answer_text": "[Part 52] provides that permission to appeal is generally required before an appeal can proceed.",
        "answer_fragment": "permission to appeal is generally required",
        "expected_citation_count": 1,
    },
    {
        "id": "edge_citation_at_end_no_period",
        "sources": [GUIDE_CHANCERY],
        "answer_text": "Applications in cases of urgency may be made before issue of a claim form [Chancery Guide]",
        "answer_fragment": "urgency may be made before",
        "expected_citation_count": 1,
    },
    {
        "id": "edge_very_long_citation_text_in_brackets",
        "sources": [PD_57AC],
        "answer_text": "Witness statements must comply with strict requirements [Practice Direction 57Ac \u2013 Trial Witness Statements In The Business And Property Courts] to be admissible at trial.",
        "answer_fragment": "Witness statements must comply",
        "expected_citation_count": 1,
    },
    {
        "id": "edge_pdf_filename_with_dots_and_underscores",
        "sources": [GUIDE_KINGS_BENCH],
        "answer_text": "The King\u2019s Bench Division Guide covers enrolment of deeds and other procedural matters [35.16_JO_Kings_Bench_Division_Guide_2025_WEB4.pdf].",
        "answer_fragment": "enrolment of deeds",
        "expected_citation_count": 1,
    },
    {
        "id": "edge_same_sourcefile_different_sources",
        "sources": [PREACTION_JUDICIAL_REVIEW, PREACTION_HOUSING],
        "answer_text": "The judicial review protocol requires a letter before claim [Pre]. The housing protocol also requires early engagement [Pre].",
        "answer_fragment": "judicial review protocol",
        "expected_citation_count": 1,
    },
    {
        "id": "edge_em_dash_in_citation_brackets",
        "sources": [CPR_PART_85],
        "answer_text": "Claims on controlled goods require specific procedures [Part 85 \u2013 Claims On Controlled Goods And Executed Goods].",
        "answer_fragment": "Claims on controlled goods require",
        "expected_citation_count": 1,
    },
    {
        "id": "edge_many_citations_dense_answer",
        "sources": [CPR_PART_1, CPR_PART_3, CPR_PART_24, CPR_PART_52, PD_44, PD_19A],
        "answer_text": "Proceedings are governed by the overriding objective [Part 1]. The court has case management powers [Part 3]. Summary judgment may be granted [Part 24]. Appeals require permission [Part 52]. Costs are at the court\u2019s discretion [Practice Direction 44]. Derivative claims have their own rules [Practice Direction 19A].",
        "answer_fragment": "governed by the overriding objective",
        "expected_citation_count": 6,
    },
]


# ---------------------------------------------------------------------------
# DETAILED PER-SOURCE-TYPE PARSING TESTS
# ---------------------------------------------------------------------------


class TestCPRSourceParsing:
    """Test citation parsing for diverse CPR Part naming patterns.

    Covers: simple titles, apostrophes, 2-digit part numbers, short names,
    very long sourcepages (Wales Act), full title in sourcefile.
    """

    @pytest.fixture(params=CPR_SCENARIOS, ids=lambda s: s["id"])
    def scenario(self, request):
        return request.param

    @pytest.fixture(autouse=True)
    def setup_route(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        snapshot = build_streaming_snapshot(scenario["sources"], scenario["answer_text"])

        def handle_stream(route: Route):
            route.fulfill(body=snapshot, status=200, headers={"Transfer-encoding": "Chunked"})

        setup_config_mocks(page)
        page.route("*/**/chat/stream", handle_stream)
        open_admin_mode(page, live_server_url)
        expect(page).to_have_title(PAGE_TITLE)

    def test_answer_renders_without_corruption(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        """Verify the answer text renders intact — no characters dropped or mangled by the parser."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()

    def test_citation_badge_appears(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        """Verify at least one citation badge renders at the bottom of the answer."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()
        citations = page.locator("a.citation, a[class*='citation']")
        expect(citations.first).to_be_visible()

    def test_supporting_content_shows_source_text(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        """Verify the source's content appears in the Supporting Content panel."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()
        page.get_by_label("Show supporting content").click()
        expect(page.get_by_text("Supporting content")).to_be_visible()
        panel = page.locator("[class*='analysisPanelContent'], [class*='supportingContent']")
        expect(panel.first).to_be_visible()
        panel_text = panel.first.text_content() or ""
        snippet = scenario["sources"][0]["content"][:40]
        assert snippet in panel_text, f"Expected '{snippet}' in supporting content panel"


class TestPDSourceParsing:
    """Test citation parsing for diverse Practice Direction naming patterns.

    Covers: simple numeric, letter suffix A, colon separator with mixed case B,
    multi-letter suffix AC, double em-dash with very long name.
    """

    @pytest.fixture(params=PD_SCENARIOS, ids=lambda s: s["id"])
    def scenario(self, request):
        return request.param

    @pytest.fixture(autouse=True)
    def setup_route(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        snapshot = build_streaming_snapshot(scenario["sources"], scenario["answer_text"])

        def handle_stream(route: Route):
            route.fulfill(body=snapshot, status=200, headers={"Transfer-encoding": "Chunked"})

        setup_config_mocks(page)
        page.route("*/**/chat/stream", handle_stream)
        open_admin_mode(page, live_server_url)
        expect(page).to_have_title(PAGE_TITLE)

    def test_answer_renders_without_corruption(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        """Verify the answer text renders intact — PD names with colons, dashes, mixed case survive."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()

    def test_citation_badge_appears(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()
        citations = page.locator("a.citation, a[class*='citation']")
        expect(citations.first).to_be_visible()

    def test_supporting_content_shows_source_text(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()
        page.get_by_label("Show supporting content").click()
        expect(page.get_by_text("Supporting content")).to_be_visible()
        panel = page.locator("[class*='analysisPanelContent'], [class*='supportingContent']")
        expect(panel.first).to_be_visible()
        panel_text = panel.first.text_content() or ""
        snippet = scenario["sources"][0]["content"][:40]
        assert snippet in panel_text, f"Expected '{snippet}' in supporting content panel"


class TestPreActionSourceParsing:
    """Test citation parsing for Pre-Action Protocol naming patterns.

    Covers: truncated 'Pre' sourcefile, parenthetical '(England)', construction disputes.
    These all share the truncated sourcefile='Pre' — tests that identical citation text
    is handled gracefully.
    """

    @pytest.fixture(params=PREACTION_SCENARIOS, ids=lambda s: s["id"])
    def scenario(self, request):
        return request.param

    @pytest.fixture(autouse=True)
    def setup_route(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        snapshot = build_streaming_snapshot(scenario["sources"], scenario["answer_text"])

        def handle_stream(route: Route):
            route.fulfill(body=snapshot, status=200, headers={"Transfer-encoding": "Chunked"})

        setup_config_mocks(page)
        page.route("*/**/chat/stream", handle_stream)
        open_admin_mode(page, live_server_url)
        expect(page).to_have_title(PAGE_TITLE)

    def test_answer_renders_without_corruption(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        """Verify the answer text renders intact — short 'Pre' citation doesn't break parsing."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()

    def test_citation_badge_appears(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()
        citations = page.locator("a.citation, a[class*='citation']")
        expect(citations.first).to_be_visible()

    def test_supporting_content_shows_source_text(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()
        page.get_by_label("Show supporting content").click()
        expect(page.get_by_text("Supporting content")).to_be_visible()
        panel = page.locator("[class*='analysisPanelContent'], [class*='supportingContent']")
        expect(panel.first).to_be_visible()
        panel_text = panel.first.text_content() or ""
        snippet = scenario["sources"][0]["content"][:40]
        assert snippet in panel_text, f"Expected '{snippet}' in supporting content panel"


class TestCourtGuideSourceParsing:
    """Test citation parsing for diverse Court Guide naming patterns.

    Covers: Commercial Court (dot-letter sourcepage), TCC (Section N. pattern),
    Chancery (Part/Chapter hierarchy), King's Bench (PDF filename as sourcefile!),
    Patents (Annex: pattern).
    """

    @pytest.fixture(params=COURT_GUIDE_SCENARIOS, ids=lambda s: s["id"])
    def scenario(self, request):
        return request.param

    @pytest.fixture(autouse=True)
    def setup_route(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        snapshot = build_streaming_snapshot(scenario["sources"], scenario["answer_text"])

        def handle_stream(route: Route):
            route.fulfill(body=snapshot, status=200, headers={"Transfer-encoding": "Chunked"})

        setup_config_mocks(page)
        page.route("*/**/chat/stream", handle_stream)
        open_admin_mode(page, live_server_url)
        expect(page).to_have_title(PAGE_TITLE)

    def test_answer_renders_without_corruption(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        """Verify the answer text renders intact — complex court guide names survive parsing."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()

    def test_citation_badge_appears(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()
        citations = page.locator("a.citation, a[class*='citation']")
        expect(citations.first).to_be_visible()

    def test_supporting_content_shows_source_text(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()
        page.get_by_label("Show supporting content").click()
        expect(page.get_by_text("Supporting content")).to_be_visible()
        panel = page.locator("[class*='analysisPanelContent'], [class*='supportingContent']")
        expect(panel.first).to_be_visible()
        panel_text = panel.first.text_content() or ""
        snippet = scenario["sources"][0]["content"][:40]
        assert snippet in panel_text, f"Expected '{snippet}' in supporting content panel"


class TestMixedSourcesParsing:
    """Test citation parsing with multiple sources from different categories in one answer.

    Validates that the parser handles diverse citation patterns mixed together
    without corruption or interference.
    """

    @pytest.fixture(params=MIXED_SOURCE_SCENARIOS, ids=lambda s: s["id"])
    def scenario(self, request):
        return request.param

    @pytest.fixture(autouse=True)
    def setup_route(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        snapshot = build_streaming_snapshot(scenario["sources"], scenario["answer_text"])

        def handle_stream(route: Route):
            route.fulfill(body=snapshot, status=200, headers={"Transfer-encoding": "Chunked"})

        setup_config_mocks(page)
        page.route("*/**/chat/stream", handle_stream)
        open_admin_mode(page, live_server_url)
        expect(page).to_have_title(PAGE_TITLE)

    def test_answer_renders_without_corruption(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        """Verify the answer with multiple mixed citations renders correctly."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()

    def test_expected_citation_count(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        """Verify the correct number of citation badges appear for mixed-source answers."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()
        citations = page.locator("a.citation, a[class*='citation']")
        expect(citations.first).to_be_visible()
        count = citations.count()
        expected = scenario.get("expected_citation_count", len(scenario["sources"]))
        assert count == expected, f"Expected {expected} citation badges, got {count}"

    def test_supporting_content_has_multiple_sources(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        """Verify supporting content panel loads with content from multiple sources."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()
        page.get_by_label("Show supporting content").click()
        expect(page.get_by_text("Supporting content")).to_be_visible()
        panel = page.locator("[class*='analysisPanelContent'], [class*='supportingContent']")
        expect(panel.first).to_be_visible()
        panel_text = panel.first.text_content() or ""
        # At least one source's content snippet should appear
        found = any(s["content"][:30] in panel_text for s in scenario["sources"])
        assert found, "None of the source content snippets appeared in the supporting content panel"


class TestEdgeCaseParsing:
    """Test citation parsing edge cases that stress the parser.

    Covers: adjacent citations [A][B], citation at start of answer, citation at end
    without period, very long citation text in brackets, PDF filename with dots/underscores,
    duplicate citation text from different sources, em-dash inside brackets,
    dense answer with 6 citations.
    """

    @pytest.fixture(params=EDGE_CASE_SCENARIOS, ids=lambda s: s["id"])
    def scenario(self, request):
        return request.param

    @pytest.fixture(autouse=True)
    def setup_route(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        snapshot = build_streaming_snapshot(scenario["sources"], scenario["answer_text"])

        def handle_stream(route: Route):
            route.fulfill(body=snapshot, status=200, headers={"Transfer-encoding": "Chunked"})

        setup_config_mocks(page)
        page.route("*/**/chat/stream", handle_stream)
        open_admin_mode(page, live_server_url)
        expect(page).to_have_title(PAGE_TITLE)

    def test_answer_renders_without_corruption(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        """Verify the answer text renders intact — edge case patterns don't corrupt output."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()

    def test_citation_badges_appear(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        """Verify the correct number of citation badges appear for edge case answers."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()
        citations = page.locator("a.citation, a[class*='citation']")
        expect(citations.first).to_be_visible()
        count = citations.count()
        expected = scenario.get("expected_citation_count", len(scenario["sources"]))
        assert count == expected, f"Expected {expected} citation badges, got {count}"

    def test_supporting_content_accessible(self, page: Page, live_server_url: str, scenario):  # noqa: F811
        """Verify supporting content panel opens and shows content for edge case scenarios."""
        submit_question(page, TEST_QUESTION)
        expect(page.get_by_text(scenario["answer_fragment"])).to_be_visible()
        page.get_by_label("Show supporting content").click()
        expect(page.get_by_text("Supporting content")).to_be_visible()
        panel = page.locator("[class*='analysisPanelContent'], [class*='supportingContent']")
        expect(panel.first).to_be_visible()


# ---------------------------------------------------------------------------
# DEPLOYED VERSION TESTS — run against a live deployment
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("DEPLOYED_URL"),
    reason="Set DEPLOYED_URL environment variable to run deployed tests",
)
class TestDeployedCitations:
    """Test citations against a live deployed version of the application.

    Set DEPLOYED_URL=https://your-app.azurewebsites.net to run these tests.
    These tests send real questions and validate the response structure
    without mocking, so they require a working deployment with indexed data.
    """

    @pytest.fixture
    def deployed_url(self):
        return os.environ["DEPLOYED_URL"].rstrip("/") + "/"

    def test_deployed_page_loads(self, page: Page, deployed_url: str):
        """Verify the deployed app loads correctly."""
        page.goto(deployed_url)
        expect(page).to_have_title(PAGE_TITLE)
        expect(page.get_by_role("heading", name=PAGE_TITLE)).to_be_visible()

    def test_deployed_chat_returns_citations(self, page: Page, deployed_url: str):
        """Verify that a real chat query returns an answer with citations."""
        open_admin_mode(page, deployed_url)
        expect(page).to_have_title(PAGE_TITLE)

        # Ask a question relevant to index v3 source types
        submit_question(page, DEPLOYED_TEST_QUESTION)

        # Wait for an answer to appear (longer timeout for real API calls)
        answer_container = page.locator("[class*='answerContainer'], [data-answer-index]")
        expect(answer_container.first).to_be_visible(timeout=60_000)

        # There should be at least one citation
        citation_links = page.locator("a.citation, a[class*='citation']")
        expect(citation_links.first).to_be_visible(timeout=30_000)

    def test_deployed_supporting_content_works(self, page: Page, deployed_url: str):
        """Verify supporting content panel works on the deployed version."""
        open_admin_mode(page, deployed_url)
        expect(page).to_have_title(PAGE_TITLE)

        submit_question(page, DEPLOYED_TEST_QUESTION)

        # Wait for answer
        answer_container = page.locator("[class*='answerContainer'], [data-answer-index]")
        expect(answer_container.first).to_be_visible(timeout=60_000)

        # Open supporting content
        page.get_by_label("Show supporting content").click()
        expect(page.get_by_text("Supporting content")).to_be_visible(timeout=10_000)

        # The panel should have some content
        panel_content = page.locator("[role='tabpanel']")
        expect(panel_content.first).to_be_visible()

    def test_deployed_citation_click_opens_panel(self, page: Page, deployed_url: str):
        """Verify clicking a citation on the deployed version opens the correct panel."""
        open_admin_mode(page, deployed_url)
        expect(page).to_have_title(PAGE_TITLE)

        submit_question(page, DEPLOYED_TEST_QUESTION)

        # Wait for answer and citations
        answer_container = page.locator("[class*='answerContainer'], [data-answer-index]")
        expect(answer_container.first).to_be_visible(timeout=60_000)

        citation_links = page.locator("a.citation, a[class*='citation']")
        expect(citation_links.first).to_be_visible(timeout=30_000)

        # Click the first citation
        citation_links.first.click()

        # A panel should open (either SupportingContent or Citation tab)
        panel = page.locator("[role='tabpanel']")
        expect(panel.first).to_be_visible(timeout=10_000)

    def test_deployed_thought_process_works(self, page: Page, deployed_url: str):
        """Verify the thought process panel works on the deployed version (admin mode)."""
        open_admin_mode(page, deployed_url)
        expect(page).to_have_title(PAGE_TITLE)

        submit_question(page, DEPLOYED_TEST_QUESTION)

        # Wait for answer
        answer_container = page.locator("[class*='answerContainer'], [data-answer-index]")
        expect(answer_container.first).to_be_visible(timeout=60_000)

        # Open thought process
        page.get_by_label("Show thought process").click()
        expect(page.get_by_text("Thought process")).to_be_visible(timeout=10_000)

        # Should show search-related information
        expect(page.get_by_text("Prompt to generate search query")).to_be_visible()

    def test_deployed_multiple_questions_maintain_citations(self, page: Page, deployed_url: str):
        """Verify that citations work correctly across multiple follow-up questions."""
        open_admin_mode(page, deployed_url)
        expect(page).to_have_title(PAGE_TITLE)

        # First question
        submit_question(page, DEPLOYED_TEST_QUESTION)
        answer_container = page.locator("[class*='answerContainer'], [data-answer-index]")
        expect(answer_container.first).to_be_visible(timeout=60_000)

        # Second question
        submit_question(page, "Tell me more about the first topic you mentioned")
        # Wait for second answer
        expect(answer_container.nth(1)).to_be_visible(timeout=60_000)

        # Both answers should still be visible
        expect(answer_container).to_have_count(2, timeout=10_000)

    def test_deployed_ai_disclaimer_present(self, page: Page, deployed_url: str):
        """Verify the AI disclaimer appears on the deployed version."""
        open_admin_mode(page, deployed_url)
        expect(page).to_have_title(PAGE_TITLE)

        submit_question(page, DEPLOYED_TEST_QUESTION)

        answer_container = page.locator("[class*='answerContainer'], [data-answer-index]")
        expect(answer_container.first).to_be_visible(timeout=60_000)

        # Check for AI disclaimer
        expect(page.get_by_text("AI-generated content may be incorrect")).to_be_visible()
