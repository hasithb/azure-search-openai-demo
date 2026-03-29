"""
Tests for the multi-citation pipeline: verifies that the backend generates
multiple numbered citations ([1], [2], [3]) and that the frontend can resolve
them back to the correct enhanced citation strings.

This covers:
1. format_text_sources_for_prompt() - numbering multiple sources
2. get_sources_content() - building DataPoints with multiple citations
3. Jinja2 prompt template - listing multiple [1] [2] [3] placeholders
4. AnswerParser (frontend tested separately) - resolving [n] to data_points.text[n-1].citation
"""

import pytest
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient

from approaches.approach import Document
from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach
from approaches.promptmanager import PromptManager

from .mocks import MOCK_EMBEDDING_DIMENSIONS, MOCK_EMBEDDING_MODEL_NAME


@pytest.fixture
def chat_approach_minimal():
    """Minimal chat approach for unit testing (no blob managers needed)."""
    return ChatReadRetrieveReadApproach(
        search_client=SearchClient(endpoint="", index_name="", credential=AzureKeyCredential("")),
        search_index_name=None,
        knowledgebase_model=None,
        knowledgebase_deployment=None,
        knowledgebase_client=None,
        openai_client=None,
        chatgpt_model="gpt-4.1-mini",
        chatgpt_deployment="chat",
        embedding_deployment="embeddings",
        embedding_model=MOCK_EMBEDDING_MODEL_NAME,
        embedding_dimensions=MOCK_EMBEDDING_DIMENSIONS,
        embedding_field="embedding3",
        sourcepage_field="",
        content_field="",
        query_language="en-us",
        query_speller="lexicon",
        prompt_manager=PromptManager(),
    )


# ──────────────────────────────────────────────────────────
# 1. format_text_sources_for_prompt: numbered formatting
# ──────────────────────────────────────────────────────────


class TestFormatTextSourcesForPrompt:
    """Verify that format_text_sources_for_prompt produces [1]:, [2]:, [3]: prefixed strings."""

    def test_single_source_dict(self):
        sources = [{"content": "Rule 1.1 governs filing deadlines."}]
        result = ChatReadRetrieveReadApproach.format_text_sources_for_prompt(sources)
        assert len(result) == 1
        assert result[0] == "[1]: Rule 1.1 governs filing deadlines."

    def test_multiple_sources_dict(self):
        sources = [
            {"content": "Rule 1.1 governs filing deadlines."},
            {"content": "Practice Direction 3E covers costs budgets."},
            {"content": "Part 35 deals with expert evidence."},
        ]
        result = ChatReadRetrieveReadApproach.format_text_sources_for_prompt(sources)
        assert len(result) == 3
        assert result[0] == "[1]: Rule 1.1 governs filing deadlines."
        assert result[1] == "[2]: Practice Direction 3E covers costs budgets."
        assert result[2] == "[3]: Part 35 deals with expert evidence."

    def test_multiple_sources_strings(self):
        """Legacy string format still works."""
        sources = ["First source text", "Second source text"]
        result = ChatReadRetrieveReadApproach.format_text_sources_for_prompt(sources)
        assert len(result) == 2
        assert result[0] == "[1]: First source text"
        assert result[1] == "[2]: Second source text"

    def test_empty_sources(self):
        result = ChatReadRetrieveReadApproach.format_text_sources_for_prompt([])
        assert result == []

    def test_none_sources(self):
        result = ChatReadRetrieveReadApproach.format_text_sources_for_prompt(None)
        assert result == []

    def test_five_sources(self):
        """Verify numbering works for more than a handful of sources."""
        sources = [{"content": f"Source {i} content"} for i in range(1, 6)]
        result = ChatReadRetrieveReadApproach.format_text_sources_for_prompt(sources)
        assert len(result) == 5
        for i, line in enumerate(result, 1):
            assert line.startswith(f"[{i}]: ")
            assert f"Source {i} content" in line


# ──────────────────────────────────────────────────────────
# 2. get_sources_content: builds multiple citations
# ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_sources_content_multiple_documents_produces_multiple_citations(chat_approach_minimal):
    """Each Document gets a unique citation in data_points.citations."""
    docs = [
        Document(
            id="doc1",
            content="1.1 Filing requirements for claims must be met within 14 days.",
            sourcepage="Part 1",
            sourcefile="CPR_Part1.pdf",
            subsection_id="1.1",
        ),
        Document(
            id="doc2",
            content="3.1 The court may strike out a statement of case.",
            sourcepage="Part 3",
            sourcefile="CPR_Part3.pdf",
            subsection_id="3.1",
        ),
        Document(
            id="doc3",
            content="A4.1 Documents should be paginated and indexed.",
            sourcepage="D5",
            sourcefile="Commercial_Court_Guide.pdf",
            subsection_id="A4.1",
        ),
    ]

    data_points = await chat_approach_minimal.get_sources_content(
        docs,
        use_semantic_captions=False,
        include_text_sources=True,
        download_image_sources=False,
        user_oid=None,
    )

    # Should have 3 distinct citations
    assert len(data_points.citations) == 3
    # Each citation should be unique
    assert len(set(data_points.citations)) == 3

    # Should have 3 text sources
    assert len(data_points.text) == 3

    # Each text source dict should have a citation field matching data_points.citations
    for i, text_source in enumerate(data_points.text):
        assert "citation" in text_source
        assert text_source["citation"] == data_points.citations[i]

    # Verify content preserved
    assert "Filing requirements" in data_points.text[0]["content"]
    assert "strike out" in data_points.text[1]["content"]
    assert "paginated and indexed" in data_points.text[2]["content"]


@pytest.mark.asyncio
async def test_get_sources_content_custom_fields_populated(chat_approach_minimal):
    """Verify that updated, storageurl, and subsection_id flow through to text sources."""
    docs = [
        Document(
            id="doc1",
            content="Rule 31.1 applies to disclosure.",
            sourcepage="Part 31",
            sourcefile="CPR_Part31.pdf",
            subsection_id="31.1",
            storage_url="https://example.blob.core.windows.net/docs/CPR_Part31.pdf",
            updated="2025-01-15",
        ),
        Document(
            id="doc2",
            content="Practice Direction 3E: costs management.",
            sourcepage="PD3E",
            sourcefile="PD3E.pdf",
            subsection_id="3E.1",
            storage_url="https://example.blob.core.windows.net/docs/PD3E.pdf",
            updated="2025-06-20",
        ),
    ]

    data_points = await chat_approach_minimal.get_sources_content(
        docs,
        use_semantic_captions=False,
        include_text_sources=True,
        download_image_sources=False,
        user_oid=None,
    )

    # Verify custom fields are present and populated
    assert data_points.text[0]["updated"] == "2025-01-15"
    assert data_points.text[0]["storageurl"] == "https://example.blob.core.windows.net/docs/CPR_Part31.pdf"
    assert data_points.text[0]["subsection_id"] == "31.1"

    assert data_points.text[1]["updated"] == "2025-06-20"
    assert data_points.text[1]["storageurl"] == "https://example.blob.core.windows.net/docs/PD3E.pdf"
    assert data_points.text[1]["subsection_id"] == "3E.1"


@pytest.mark.asyncio
async def test_get_sources_content_deduplicates_citations(chat_approach_minimal):
    """If two documents produce the same citation string, it should only appear once."""
    docs = [
        Document(
            id="doc1",
            content="1.1 First chunk of Part 1.",
            sourcepage="Part 1",
            sourcefile="CPR_Part1.pdf",
            subsection_id="1.1",
        ),
        Document(
            id="doc2",
            content="1.1 Second chunk of Part 1 (same subsection).",
            sourcepage="Part 1",
            sourcefile="CPR_Part1.pdf",
            subsection_id="1.1",
        ),
    ]

    data_points = await chat_approach_minimal.get_sources_content(
        docs,
        use_semantic_captions=False,
        include_text_sources=True,
        download_image_sources=False,
        user_oid=None,
    )

    # Citations should be deduplicated
    assert len(data_points.citations) == 1
    # But both text sources are still included (content differs)
    assert len(data_points.text) == 2


# ──────────────────────────────────────────────────────────
# 3. Jinja2 prompt: citations list passed to template
# ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_renders_multiple_numbered_citations(chat_approach_minimal):
    """Verify chat_answer.system.jinja2 renders [1] [2] [3] when given 3 text sources."""
    docs = [
        Document(
            id="doc1",
            content="1.1 Filing deadlines for claims.",
            sourcepage="Part 1",
            sourcefile="CPR_Part1.pdf",
            subsection_id="1.1",
        ),
        Document(
            id="doc2",
            content="3.1 Strike out provisions.",
            sourcepage="Part 3",
            sourcefile="CPR_Part3.pdf",
            subsection_id="3.1",
        ),
        Document(
            id="doc3",
            content="A4.1 Bundle preparation guidelines.",
            sourcepage="D5",
            sourcefile="Commercial_Court_Guide.pdf",
            subsection_id="A4.1",
        ),
    ]

    data_points = await chat_approach_minimal.get_sources_content(
        docs,
        use_semantic_captions=False,
        include_text_sources=True,
        download_image_sources=False,
        user_oid=None,
    )

    # Build the numbered citations list as chatreadretrieveread.py does
    numbered_citations = [str(i) for i in range(1, len(data_points.text) + 1)]
    assert numbered_citations == ["1", "2", "3"]

    # Render the prompt
    messages = chat_approach_minimal.prompt_manager.build_conversation(
        system_template_path="chat_answer.system.jinja2",
        system_template_variables={
            "include_follow_up_questions": False,
            "image_sources": data_points.images,
            "citations": numbered_citations,
        },
        user_template_path="chat_answer.user.jinja2",
        user_template_variables={
            "user_query": "What are the filing deadlines?",
            "text_sources": ChatReadRetrieveReadApproach.format_text_sources_for_prompt(data_points.text),
        },
        user_image_sources=data_points.images,
        past_messages=[],
    )

    # System prompt should contain all three citation placeholders
    system_msg = next(m["content"] for m in messages if m["role"] == "system")
    assert "[1]" in system_msg
    assert "[2]" in system_msg
    assert "[3]" in system_msg

    # User message should contain numbered sources
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert "[1]:" in user_msg
    assert "[2]:" in user_msg
    assert "[3]:" in user_msg

    # User message should contain the actual content
    assert "Filing deadlines" in user_msg
    assert "Strike out" in user_msg
    assert "Bundle preparation" in user_msg


# ──────────────────────────────────────────────────────────
# 4. Data points structure: frontend resolution test data
# ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_data_points_text_has_citation_field_for_frontend_resolution(chat_approach_minimal):
    """
    The frontend resolves [1] → data_points.text[0].citation, [2] → data_points.text[1].citation, etc.
    Verify that each text source dict has a non-empty 'citation' field that the frontend can use.
    """
    docs = [
        Document(id="d1", content="Filing rules.", sourcepage="Part 1", sourcefile="CPR_Part1.pdf", subsection_id="1.1"),
        Document(id="d2", content="Strike out rules.", sourcepage="Part 3", sourcefile="CPR_Part3.pdf", subsection_id="3.1"),
        Document(id="d3", content="Expert evidence.", sourcepage="Part 35", sourcefile="CPR_Part35.pdf", subsection_id="35.1"),
        Document(id="d4", content="Costs management.", sourcepage="PD3E", sourcefile="PD3E.pdf", subsection_id="3E.1"),
        Document(id="d5", content="Bundle prep.", sourcepage="D5", sourcefile="Commercial_Court_Guide.pdf", subsection_id="A4.1"),
    ]

    data_points = await chat_approach_minimal.get_sources_content(
        docs,
        use_semantic_captions=False,
        include_text_sources=True,
        download_image_sources=False,
        user_oid=None,
    )

    # Should have 5 text sources
    assert len(data_points.text) == 5

    # Each text source should have a non-empty citation that is a string
    for i, text_source in enumerate(data_points.text):
        assert isinstance(text_source["citation"], str), f"text_source[{i}].citation is not a string"
        assert len(text_source["citation"]) > 0, f"text_source[{i}].citation is empty"

    # The citations should all be different (since all documents are different)
    all_citations = [ts["citation"] for ts in data_points.text]
    assert len(set(all_citations)) == 5, f"Expected 5 unique citations but got: {all_citations}"

    # Verify the frontend mapping:
    # LLM outputs [1] → frontend reads data_points.text[0].citation → gets enhanced citation
    # LLM outputs [2] → frontend reads data_points.text[1].citation → gets enhanced citation
    # etc.
    for i in range(5):
        citation = data_points.text[i]["citation"]
        # Each enhanced citation should contain the sourcefile name
        expected_file = docs[i].sourcefile
        assert expected_file in citation, (
            f"Citation for doc {i+1} should contain '{expected_file}' but got '{citation}'"
        )
