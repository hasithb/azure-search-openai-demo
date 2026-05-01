import base64
import json

import pytest
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseUsage,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from approaches.approach import (
    ActivityDetail,
    DataPoints,
    Document,
    ExtraInfo,
    RewriteQueryResult,
    SharePointResult,
    ThoughtStep,
    WebResult,
)
from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach
from approaches.promptmanager import PromptManager
from prepdocslib.embeddings import ImageEmbeddings

from .mocks import (
    MOCK_EMBEDDING_DIMENSIONS,
    MOCK_EMBEDDING_MODEL_NAME,
    MockAsyncSearchResultsIterator,
    mock_retrieval_response,
)


async def mock_search(*args, **kwargs):
    return MockAsyncSearchResultsIterator(kwargs.get("search_text"), kwargs.get("vector_queries"))


async def mock_retrieval(*args, **kwargs):
    return mock_retrieval_response()


def test_get_search_query(chat_approach):
    response = Response(
        id="resp-81JkxYqYppUkPtOAia40gki2vJ9QM",
        object="response",
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        created_at=1695324963,
        model="gpt-4.1-mini",
        output=[
            ResponseFunctionToolCall(
                id="fc_search_sources1235",
                type="function_call",
                call_id="call_search_sources1235",
                name="search_sources",
                arguments='{\n"search_query":"accesstelemedicineservices"\n}',
                status="completed",
            ),
            ResponseOutputMessage(
                id="msg-1",
                type="message",
                role="assistant",
                status="completed",
                content=[{"type": "output_text", "text": "this is the query", "annotations": []}],
            ),
        ],
        status="completed",
    )
    default_query = "hello"
    query = chat_approach.get_search_query(response, default_query)

    assert query == "accesstelemedicineservices"


def test_get_search_query_returns_default(chat_approach):
    response = Response(
        id="resp-81JkxYqYppUkPtOAia40gki2vJ9QM",
        object="response",
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        created_at=1695324963,
        model="gpt-4.1-mini",
        output=[
            ResponseOutputMessage(
                id="msg-1",
                type="message",
                role="assistant",
                status="completed",
                content=[{"type": "output_text", "text": "", "annotations": []}],
            ),
        ],
        status="completed",
    )
    default_query = "hello"
    query = chat_approach.get_search_query(response, default_query)

    assert query == default_query


def test_get_search_query_returns_default_on_error(chat_approach, monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(chat_approach, "extract_rewritten_query", explode)

    response = Response(
        id="resp-1",
        object="response",
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        created_at=0,
        model="gpt-4.1-mini",
        output=[
            ResponseOutputMessage(
                id="msg-1",
                type="message",
                role="assistant",
                status="completed",
                content=[{"type": "output_text", "text": "anything", "annotations": []}],
            )
        ],
        status="completed",
    )

    assert chat_approach.get_search_query(response, "default") == "default"


def test_extract_rewritten_query_invalid_json(chat_approach):
    response = Response(
        id="resp-2",
        object="response",
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        created_at=0,
        model="gpt-4.1-mini",
        output=[
            ResponseFunctionToolCall(
                id="fc_tool-1",
                type="function_call",
                call_id="call_tool-1",
                name="search_sources",
                arguments="{not-json",
                status="completed",
            ),
            ResponseOutputMessage(
                id="msg-1",
                type="message",
                role="assistant",
                status="completed",
                content=[{"type": "output_text", "text": "fallback query", "annotations": []}],
            ),
        ],
        status="completed",
    )

    result = chat_approach.extract_rewritten_query(response, "original", no_response_token=chat_approach.NO_RESPONSE)

    assert result == "fallback query"


@pytest.mark.asyncio
async def test_rewrite_query_sends_live_query_as_user_message_and_forces_tool(chat_approach, monkeypatch):
    captured: dict[str, object] = {}

    async def fake_create_response(*_args, **kwargs):
        captured["input"] = kwargs["input"]
        captured["tools"] = kwargs.get("tools")
        return Response.model_validate(
            {
                "id": "rewrite-structured",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4.1-mini",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc-1",
                        "call_id": "call-1",
                        "name": "search_sources",
                        "arguments": json.dumps(
                            {"search_query": "CPR Part 3 extend shorten time compliance", "subsection_hint": ""}
                        ),
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
                "status": "completed",
            },
            strict=False,
        )

    monkeypatch.setattr(chat_approach, "create_response", fake_create_response)

    result = await chat_approach.rewrite_query(
        prompt_template="query_rewrite.system.jinja2",
        prompt_variables={
            "user_query": "What power does the court have under CPR Part 3 to extend or shorten time for compliance?",
            "past_messages": [{"role": "assistant", "content": "Earlier answer"}],
        },
        overrides={},
        chatgpt_model=chat_approach.chatgpt_model,
        chatgpt_deployment=chat_approach.chatgpt_deployment,
        user_query="What power does the court have under CPR Part 3 to extend or shorten time for compliance?",
        response_token_limit=100,
        tools=chat_approach.query_rewrite_tools,
        temperature=0.0,
        no_response_token=chat_approach.NO_RESPONSE,
    )

    messages = captured["input"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert "What power does the court have under CPR Part 3" not in messages[0]["content"]
    assert messages[1] == {"role": "assistant", "content": "Earlier answer"}
    assert messages[2] == {
        "role": "user",
        "content": "Generate search query for: What power does the court have under CPR Part 3 to extend or shorten time for compliance?",
    }
    assert captured["tools"] == chat_approach.query_rewrite_tools
    assert result.query == "CPR Part 3 extend shorten time compliance"
    assert result.subsection_hint is None


def test_merge_rewritten_query_preserves_explicit_legal_references(chat_approach):
    merged = chat_approach._merge_rewritten_query_with_explicit_references(
        "pre action steps before construction proceedings summary judgment relevant rules",
        "Before commencing construction proceedings, what pre-action steps apply, when can summary judgment be granted, and what Practice Direction 27B point is relevant under Part 24?",
    )

    assert "Part 24" in merged
    assert "Practice Direction 27B" in merged


def test_should_retry_when_named_legal_reference_missing(chat_approach):
    documents = [
        Document(
            id="pre",
            content="The construction protocol contains an exception for claims that will be the subject of summary judgment pursuant to Part 24.",
            sourcepage="Pre-Action Protocol for the Construction and Engineering Disputes",
            sourcefile="Pre",
            category="Civil Procedure Rules and Practice Directions",
        )
    ]

    should_retry = chat_approach._should_retry_for_query_intent(
        documents,
        "Before commencing construction proceedings, what pre-action steps apply, when can summary judgment be granted, and what Practice Direction 27B point is relevant under Part 24?",
    )

    assert should_retry is True


def test_query_intent_scoring_prefers_focused_section_over_annex(chat_approach):
    query = "What does the King's Bench Division Guide say about enrolment of deeds and other documents?"
    focused_doc = Document(
        id="kbd-26-1",
        content="26.1 Any deed or document which by virtue of any enactment is required or authorised to be enrolled in the Senior Courts may be enrolled in the Central Office.",
        sourcepage="26. Enrolment of deeds and other documents (p. 206)",
        sourcefile="King's Bench Division Guide",
        category="King's Bench Division",
        subsection_id="26.1",
    )
    annex_doc = Document(
        id="kbd-annex-13",
        content="Annex 13 gives London Gazette contact details for enrolled deed poll enquiries.",
        sourcepage="Contact Details, The London Gazette enquires: (p. 265) [Part 1]",
        sourcefile="King's Bench Division Guide",
        category="King's Bench Division",
        subsection_id="Annex 13",
    )
    adjacent_doc = Document(
        id="kbd-27-1",
        content="27.1 Under Part III of the Representation of the People Act 1983, the result of an election may be questioned.",
        sourcepage="26. Enrolment of deeds and other documents, 27. Election Petitions (p. 208)",
        sourcefile="King's Bench Division Guide",
        category="King's Bench Division",
        subsection_id="27.1",
    )

    focused_score = chat_approach._score_document_query_intent(focused_doc, query)
    annex_score = chat_approach._score_document_query_intent(annex_doc, query)
    adjacent_score = chat_approach._score_document_query_intent(adjacent_doc, query)

    assert focused_score > annex_score
    assert focused_score > adjacent_score


@pytest.mark.asyncio
async def test_run_search_approach_filters_tangential_guide_annexes_from_initial_results(chat_approach, monkeypatch):
    completion = Response.model_validate(
        {
            "id": "rewrite-kbd-1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-4.1-mini",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [{"id": "msg-1", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "enrolment of deeds and other documents", "annotations": []}], "status": "completed"}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
            "status": "completed",
        },
        strict=False,
    )

    async def fake_rewrite_query(**_kwargs):
        return RewriteQueryResult(
            query="What does the King's Bench Division Guide say about enrolment of deeds and other documents?",
            messages=[{"role": "user", "content": "original"}],
            completion=completion,
            reasoning_effort="minimal",
            subsection_hint=None,
        )

    async def fake_search(*_args, **_kwargs):
        return [
            Document(
                id="kbd-26-1",
                content="26.1 Any deed or document which by virtue of any enactment is required or authorised to be enrolled in the Senior Courts may be enrolled in the Central Office.",
                sourcepage="26. Enrolment of deeds and other documents (p. 206)",
                sourcefile="King's Bench Division Guide",
                category="King's Bench Division",
                subsection_id="26.1",
            ),
            Document(
                id="kbd-annex-13",
                content="Annex 13 gives London Gazette contact details for enrolled deed poll enquiries.",
                sourcepage="Contact Details, The London Gazette enquires: (p. 265) [Part 1]",
                sourcefile="King's Bench Division Guide",
                category="King's Bench Division",
                subsection_id="Annex 13",
            ),
            Document(
                id="kbd-annex-12",
                content="Annex 12 explains the adult change of name enrolment procedure.",
                sourcepage="Annex 12 - Guidance for Procedure for Enrolling a Change of Name for Adults (p. 258)",
                sourcefile="King's Bench Division Guide",
                category="King's Bench Division",
                subsection_id="10.00",
            ),
            Document(
                id="kbd-27-1",
                content="27.1 Under Part III of the Representation of the People Act 1983, the result of an election may be questioned.",
                sourcepage="26. Enrolment of deeds and other documents, 27. Election Petitions (p. 208)",
                sourcefile="King's Bench Division Guide",
                category="King's Bench Division",
                subsection_id="27.1",
            ),
        ]

    async def fake_get_sources_content(results, *_args, **_kwargs):
        return DataPoints(
            text=[
                {
                    "citation": result.sourcepage or result.id or "",
                    "content": result.content or "",
                    "sourcepage": result.sourcepage or "",
                    "sourcefile": result.sourcefile or "",
                    "category": result.category or "",
                }
                for result in results
            ],
            citations=[],
        )

    monkeypatch.setattr(chat_approach, "rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(chat_approach, "search", fake_search)
    monkeypatch.setattr(chat_approach, "get_sources_content", fake_get_sources_content)

    extra_info = await chat_approach.run_search_approach(
        messages=[{"role": "user", "content": "What does the King's Bench Division Guide say about enrolment of deeds and other documents?"}],
        overrides={
            "retrieval_mode": "text",
            "semantic_ranker": True,
            "semantic_captions": False,
            "query_rewriting": True,
            "top": 7,
            "send_text_sources": True,
        },
        auth_claims={},
    )

    assert extra_info.data_points.text is not None
    kept_sourcepages = [source["sourcepage"] for source in extra_info.data_points.text]
    assert kept_sourcepages == ["26. Enrolment of deeds and other documents (p. 206)"]


def test_canonical_concept_queries_pad_acronym(chat_approach):
    results = chat_approach._extract_canonical_legal_concept_queries("what are the requirements for PAD")
    assert len(results) == 1
    assert "31.16" in results[0][0]
    assert results[0][1] == "Part 31"


def test_canonical_concept_queries_pre_action_disclosure(chat_approach):
    results = chat_approach._extract_canonical_legal_concept_queries("what is pre-action disclosure")
    assert len(results) == 1
    assert "31.16" in results[0][0]
    assert results[0][1] == "Part 31"


def test_canonical_concept_queries_pad_no_false_positive(chat_approach):
    results = chat_approach._extract_canonical_legal_concept_queries("what is the padding requirement for documents")
    assert len(results) == 0


def test_canonical_concept_queries_summary_judgment(chat_approach):
    results = chat_approach._extract_canonical_legal_concept_queries("when can I get summary judgment")
    assert len(results) == 1
    assert "24.3" in results[0][0]
    assert results[0][1] == "Part 24"


def test_canonical_concept_queries_no_match(chat_approach):
    results = chat_approach._extract_canonical_legal_concept_queries("what is the court fee for filing a claim")
    assert len(results) == 0


def test_query_rewrite_prompt_contains_legal_disambiguation(chat_approach):
    """The query rewrite system prompt must teach the LLM to disambiguate legal terms."""
    system_msg = chat_approach.prompt_manager.build_system_prompt(
        "query_rewrite.system.jinja2",
        {"user_query": "test", "past_messages": []},
    )
    prompt_content = system_msg["content"]
    # Core legal domain context
    assert "Civil Procedure Rules" in prompt_content or "CPR" in prompt_content
    # PAD disambiguation — the key scenario
    assert "31.16" in prompt_content
    assert "pre-action disclosure" in prompt_content.lower()
    # Should teach about common confusions
    assert "Pre-Action Protocol" in prompt_content
    # Should include few-shot examples for legal queries
    assert "summary judgment" in prompt_content.lower()


def test_query_rewrite_tool_description_references_legal_domain(chat_approach):
    """The search tool description should guide the LLM toward legal-aware queries."""
    tool_def = chat_approach.query_rewrite_tools[0]
    description = tool_def["description"]
    assert "CPR" in description or "Civil Procedure" in description
    search_query_desc = tool_def["parameters"]["properties"]["search_query"]["description"]
    assert "CPR" in search_query_desc or "rule" in search_query_desc.lower()


SAMPLE_AVAILABLE_SOURCES = [
    "Chancery Guide",
    "Circuit Commercial Court Guide",
    "Civil Procedure Rules and Practice Directions",
    "Commercial Court Guide",
    "Court of Appeal Civil Division Guide",
    "King's Bench Division Guide",
    "Patents Court Guide",
    "Pre-Action Protocols",
    "Senior Courts Costs Office Guide",
    "Technology and Construction Court Guide",
]


def test_chat_answer_prompt_lists_available_sources(chat_approach):
    """The chat answer system prompt must list all available document sources when provided."""
    system_msg = chat_approach.prompt_manager.build_system_prompt(
        "chat_answer.system.jinja2",
        {
            "include_follow_up_questions": False,
            "image_sources": None,
            "citations": ["1", "2"],
            "available_sources": SAMPLE_AVAILABLE_SOURCES,
        },
    )
    prompt_content = system_msg["content"]
    # Must list all major source categories dynamically
    for source in SAMPLE_AVAILABLE_SOURCES:
        assert source in prompt_content, f"Missing source: {source}"


def test_chat_answer_prompt_fallback_when_no_sources(chat_approach):
    """When available_sources is empty, the prompt should use a fallback description."""
    system_msg = chat_approach.prompt_manager.build_system_prompt(
        "chat_answer.system.jinja2",
        {
            "include_follow_up_questions": False,
            "image_sources": None,
            "citations": ["1"],
            "available_sources": [],
        },
    )
    prompt_content = system_msg["content"]
    assert "English civil court procedure documents" in prompt_content
    assert "Court Guides" in prompt_content


def test_chat_answer_prompt_contains_mismatch_detection(chat_approach):
    """The chat answer prompt must instruct the LLM to detect source-question mismatches."""
    system_msg = chat_approach.prompt_manager.build_system_prompt(
        "chat_answer.system.jinja2",
        {
            "include_follow_up_questions": False,
            "image_sources": None,
            "citations": ["1"],
        },
    )
    prompt_content = system_msg["content"]
    # Should contain mismatch detection guidance
    assert "mismatch" in prompt_content.lower() or "Source mismatch" in prompt_content
    # Should contain disambiguation examples
    assert "standard" in prompt_content.lower() and "disclosure" in prompt_content.lower()
    # Should contain query refinement guidance
    assert "refinement" in prompt_content.lower() or "more targeted query" in prompt_content.lower()
    # Should contain source recommendation guidance
    assert "recommend" in prompt_content.lower() or "suggest" in prompt_content.lower()


def test_chat_answer_prompt_contains_ambiguous_term_examples(chat_approach):
    """The chat answer prompt must include examples of ambiguous legal terms."""
    system_msg = chat_approach.prompt_manager.build_system_prompt(
        "chat_answer.system.jinja2",
        {
            "include_follow_up_questions": False,
            "image_sources": None,
            "citations": ["1"],
        },
    )
    prompt_content = system_msg["content"]
    # Key ambiguous terms that the prompt should help with
    assert "standard" in prompt_content and "extended" in prompt_content  # disclosure types
    assert "costs" in prompt_content.lower()
    assert "appeal" in prompt_content.lower()
    assert "injunction" in prompt_content.lower()
    assert "service" in prompt_content.lower()


def test_query_rewrite_prompt_contains_disclosure_disambiguation(chat_approach):
    """The query rewrite prompt must distinguish standard vs extended disclosure."""
    system_msg = chat_approach.prompt_manager.build_system_prompt(
        "query_rewrite.system.jinja2",
        {"user_query": "test", "past_messages": []},
    )
    prompt_content = system_msg["content"]
    # Should distinguish standard disclosure from extended disclosure
    assert "standard disclosure" in prompt_content.lower()
    assert "extended disclosure" in prompt_content.lower() or "57AD" in prompt_content
    assert "CPR 31.6" in prompt_content or "CPR 31" in prompt_content


def test_query_rewrite_prompt_contains_broad_query_guidance(chat_approach):
    """The query rewrite prompt must guide the LLM on handling broad/ambiguous queries."""
    system_msg = chat_approach.prompt_manager.build_system_prompt(
        "query_rewrite.system.jinja2",
        {"user_query": "test", "past_messages": []},
    )
    prompt_content = system_msg["content"]
    # Should contain guidance on broad queries
    assert "broad" in prompt_content.lower() or "ambiguous" in prompt_content.lower()
    # Should contain the disclosure few-shot example
    assert "standard disclosure" in prompt_content.lower()


def test_query_rewrite_prompt_lists_all_court_guides(chat_approach):
    """The query rewrite prompt must list available sources when provided."""
    system_msg = chat_approach.prompt_manager.build_system_prompt(
        "query_rewrite.system.jinja2",
        {"user_query": "test", "past_messages": [], "available_sources": SAMPLE_AVAILABLE_SOURCES},
    )
    prompt_content = system_msg["content"]
    assert "Court of Appeal Civil Division Guide" in prompt_content
    assert "Senior Courts Costs Office Guide" in prompt_content
    assert "Circuit Commercial Court Guide" in prompt_content


def test_query_rewrite_prompt_fallback_when_no_sources(chat_approach):
    """The query rewrite prompt should use fallback text when no sources provided."""
    system_msg = chat_approach.prompt_manager.build_system_prompt(
        "query_rewrite.system.jinja2",
        {"user_query": "test", "past_messages": [], "available_sources": []},
    )
    prompt_content = system_msg["content"]
    assert "English civil court procedure documents" in prompt_content
    assert "Court Guides" in prompt_content


def test_chat_answer_prompt_contains_search_depth_guidance(chat_approach):
    """The chat answer prompt must include search depth recommendation guidance."""
    system_msg = chat_approach.prompt_manager.build_system_prompt(
        "chat_answer.system.jinja2",
        {
            "include_follow_up_questions": False,
            "image_sources": None,
            "citations": ["1"],
            "search_depth": "Standard",
        },
    )
    prompt_content = system_msg["content"]
    # Should describe the three search depth levels
    assert "Quick" in prompt_content
    assert "Standard" in prompt_content
    assert "Thorough" in prompt_content
    # Should mention the user's current depth
    assert "currently using **Standard**" in prompt_content
    # Should suggest increasing depth for complex questions
    assert "suggest" in prompt_content.lower() or "recommend" in prompt_content.lower()


def test_chat_answer_prompt_search_depth_omitted_when_empty(chat_approach):
    """When search_depth is empty, the current-depth line should not appear."""
    system_msg = chat_approach.prompt_manager.build_system_prompt(
        "chat_answer.system.jinja2",
        {
            "include_follow_up_questions": False,
            "image_sources": None,
            "citations": ["1"],
            "search_depth": "",
        },
    )
    prompt_content = system_msg["content"]
    # The three levels should still be described
    assert "Quick" in prompt_content
    assert "Thorough" in prompt_content
    # But the "currently using" line should NOT appear
    assert "currently using" not in prompt_content


def test_format_text_sources_with_metadata():
    """format_text_sources_for_prompt should include category and sourcepage metadata."""
    from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach

    sources = [
        {"content": "Rule 31.6 content", "category": "Civil Procedure Rules and Practice Directions", "sourcepage": "Part 31"},
        {"content": "Commercial Court content", "category": "Commercial Court", "sourcepage": "Case management (p. 141)"},
        {"content": "Plain string source"},
    ]
    result = ChatReadRetrieveReadApproach.format_text_sources_for_prompt(sources)
    assert result[0] == "[1] (Category: Civil Procedure Rules and Practice Directions | Source: Part 31): Rule 31.6 content"
    assert result[1] == "[2] (Category: Commercial Court | Source: Case management (p. 141)): Commercial Court content"
    assert result[2] == "[3]: Plain string source"


def test_format_text_sources_without_metadata():
    """format_text_sources_for_prompt should handle sources without metadata gracefully."""
    from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach

    sources = [
        {"content": "Just content"},
        "Plain string",
    ]
    result = ChatReadRetrieveReadApproach.format_text_sources_for_prompt(sources)
    assert result[0] == "[1]: Just content"
    assert result[1] == "[2]: Plain string"


def test_format_text_sources_empty():
    """format_text_sources_for_prompt should handle empty/None input."""
    from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach

    assert ChatReadRetrieveReadApproach.format_text_sources_for_prompt([]) == []
    assert ChatReadRetrieveReadApproach.format_text_sources_for_prompt(None) == []


def test_extract_followup_questions(chat_approach):
    content = "Here is answer to your question.<<What is the dress code?>>"
    pre_content, followup_questions = chat_approach.extract_followup_questions(content)
    assert pre_content == "Here is answer to your question."
    assert followup_questions == ["What is the dress code?"]


def test_extract_followup_questions_three(chat_approach):
    content = """Here is answer to your question.

<<What are some examples of successful product launches they should have experience with?>>
<<Are there any specific technical skills or certifications required for the role?>>
<<Is there a preference for candidates with experience in a specific industry or sector?>>"""
    pre_content, followup_questions = chat_approach.extract_followup_questions(content)
    assert pre_content == "Here is answer to your question.\n\n"
    assert followup_questions == [
        "What are some examples of successful product launches they should have experience with?",
        "Are there any specific technical skills or certifications required for the role?",
        "Is there a preference for candidates with experience in a specific industry or sector?",
    ]


def test_extract_followup_questions_no_followup(chat_approach):
    content = "Here is answer to your question."
    pre_content, followup_questions = chat_approach.extract_followup_questions(content)
    assert pre_content == "Here is answer to your question."
    assert followup_questions == []


def test_extract_followup_questions_no_pre_content(chat_approach):
    content = "<<What is the dress code?>>"
    pre_content, followup_questions = chat_approach.extract_followup_questions(content)
    assert pre_content == ""
    assert followup_questions == ["What is the dress code?"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "minimum_search_score,minimum_reranker_score,expected_result_count",
    [
        (0, 0, 1),
        (0, 2, 1),
        (0.03, 0, 1),
        (0.03, 2, 1),
        (1, 0, 0),
        (0, 4, 0),
        (1, 4, 0),
    ],
)
async def test_search_results_filtering_by_scores(
    chat_approach, monkeypatch, minimum_search_score, minimum_reranker_score, expected_result_count
):
    monkeypatch.setattr(SearchClient, "search", mock_search)

    filtered_results = await chat_approach.search(
        top=10,
        query_text="test query",
        filter=None,
        vectors=[],
        use_text_search=True,
        use_vector_search=True,
        use_semantic_ranker=True,
        use_semantic_captions=True,
        minimum_search_score=minimum_search_score,
        minimum_reranker_score=minimum_reranker_score,
    )

    assert (
        len(filtered_results) == expected_result_count
    ), f"Expected {expected_result_count} results with minimum_search_score={minimum_search_score} and minimum_reranker_score={minimum_reranker_score}"


@pytest.mark.asyncio
async def test_search_results_query_rewriting(chat_approach, monkeypatch):

    query_rewrites = None
    semantic_query = None

    async def validate_qr_and_mock_search(*args, **kwargs):
        nonlocal query_rewrites
        nonlocal semantic_query
        query_rewrites = kwargs.get("query_rewrites")
        semantic_query = kwargs.get("semantic_query")
        return await mock_search(*args, **kwargs)

    monkeypatch.setattr(SearchClient, "search", validate_qr_and_mock_search)

    results = await chat_approach.search(
        top=10,
        query_text="test query",
        filter=None,
        vectors=[],
        use_text_search=True,
        use_vector_search=True,
        use_semantic_ranker=True,
        use_semantic_captions=True,
        use_query_rewriting=True,
    )
    assert len(results) == 1
    assert query_rewrites == "generative"
    assert semantic_query == "test query"


@pytest.mark.asyncio
async def test_run_search_approach_retries_when_rewrite_results_are_weak(chat_approach, monkeypatch):
    completion = Response.model_validate(
        {
            "id": "rewrite-1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-4.1-mini",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [{"id": "msg-1", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Patents Court guidance", "annotations": []}], "status": "completed"}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
            "status": "completed",
        },
        strict=False,
    )

    async def fake_rewrite_query(**_kwargs):
        return RewriteQueryResult(
            query="Patents Court guidance",
            messages=[{"role": "user", "content": "original"}],
            completion=completion,
            reasoning_effort="minimal",
        )

    search_calls: list[str] = []

    async def fake_search(*args, **kwargs):
        query_text = args[1]
        search_calls.append(query_text)
        if query_text == "Patents Court guidance":
            return [
                Document(
                    id="weak-doc",
                    content="General introduction to the Patents Court Guide.",
                    sourcepage="The Patents Court Guide",
                    sourcefile="The Patents Court Guide",
                    category="Patents Court",
                )
            ]
        return [
            Document(
                id="strong-doc",
                content="Urgent applications must be raised promptly and supported by evidence.",
                sourcepage="Urgent applications",
                sourcefile="The Patents Court Guide",
                category="Patents Court",
                subsection_id="3.4",
            )
        ]

    async def fake_get_sources_content(results, *_args, **_kwargs):
        return DataPoints(
            text=[
                {
                    "citation": result.sourcepage or result.id or "",
                    "content": result.content or "",
                    "sourcepage": result.sourcepage or "",
                    "sourcefile": result.sourcefile or "",
                    "category": result.category or "",
                }
                for result in results
            ],
            citations=[],
        )

    monkeypatch.setattr(chat_approach, "rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(chat_approach, "search", fake_search)
    monkeypatch.setattr(chat_approach, "get_sources_content", fake_get_sources_content)

    extra_info = await chat_approach.run_search_approach(
        messages=[{"role": "user", "content": "What does the Patents Court Guide say about urgent applications?"}],
        overrides={
            "retrieval_mode": "text",
            "semantic_ranker": True,
            "semantic_captions": False,
            "query_rewriting": True,
            "top": 3,
        },
        auth_claims={},
    )

    assert search_calls == [
        "Patents Court guidance",
        "What does the Patents Court Guide say about urgent applications?",
    ]
    assert extra_info.data_points.text is not None
    assert extra_info.data_points.text[0]["content"] == "Urgent applications must be raised promptly and supported by evidence."


@pytest.mark.asyncio
async def test_run_search_approach_preserves_explicit_legal_references_in_rewritten_query(chat_approach, monkeypatch):
    completion = Response.model_validate(
        {
            "id": "rewrite-2",
            "object": "response",
            "created_at": 0,
            "model": "gpt-4.1-mini",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [{"id": "msg-1", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "construction pre action summary judgment", "annotations": []}], "status": "completed"}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
            "status": "completed",
        },
        strict=False,
    )

    async def fake_rewrite_query(**_kwargs):
        return RewriteQueryResult(
            query="construction pre action summary judgment",
            messages=[{"role": "user", "content": "original"}],
            completion=completion,
            reasoning_effort="minimal",
            subsection_hint=None,
        )

    search_calls: list[str] = []

    async def fake_search(*args, **kwargs):
        search_calls.append(args[1])
        return []

    async def fake_get_sources_content(*_args, **_kwargs):
        return DataPoints(text=[], citations=[])

    monkeypatch.setattr(chat_approach, "rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(chat_approach, "search", fake_search)
    monkeypatch.setattr(chat_approach, "get_sources_content", fake_get_sources_content)

    await chat_approach.run_search_approach(
        messages=[
            {
                "role": "user",
                "content": "Before commencing construction proceedings, what pre-action steps apply, when can summary judgment be granted, and what Practice Direction 27B point is relevant under Part 24?",
            }
        ],
        overrides={
            "retrieval_mode": "text",
            "semantic_ranker": True,
            "semantic_captions": False,
            "query_rewriting": True,
            "top": 5,
        },
        auth_claims={},
    )

    assert "Part 24" in search_calls[0]
    assert "Practice Direction 27B" in search_calls[0]


@pytest.mark.asyncio
async def test_run_search_approach_targets_missing_explicit_legal_reference(chat_approach, monkeypatch):
    completion = Response.model_validate(
        {
            "id": "rewrite-3",
            "object": "response",
            "created_at": 0,
            "model": "gpt-4.1-mini",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [{"id": "msg-1", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "CPR 24.3 construction pre action summary judgment no real prospect of succeeding", "annotations": []}], "status": "completed"}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
            "status": "completed",
        },
        strict=False,
    )

    async def fake_rewrite_query(**_kwargs):
        return RewriteQueryResult(
            query="CPR 24.3 construction pre action summary judgment no real prospect of succeeding",
            messages=[{"role": "user", "content": "original"}],
            completion=completion,
            reasoning_effort="minimal",
            subsection_hint=None,
        )

    search_calls: list[str] = []

    async def fake_search(*args, **kwargs):
        query_text = args[1]
        filter_text = args[2] if len(args) > 2 else kwargs.get("filter", "")
        search_calls.append(query_text)
        if query_text == "Practice Direction 27B":
            return [
                Document(
                    id="pd27b",
                    content="Claims under the personal injury pre-action protocol must be started under Part 7 or Part 8.",
                    sourcepage="Practice Direction 27B",
                    sourcefile="Practice Direction 27B",
                    category="Civil Procedure Rules and Practice Directions",
                )
            ]
        # Dynamic CPR Part supplemental search: the rewrite contains "CPR 24.3"
        # so the pipeline extracts Part 24 and does a category-filtered search
        # using the full rewrite text.
        if "Civil Procedure Rules" in (filter_text or "") and "24.3" in query_text:
            return [
                Document(
                    id="part24",
                    content="The court may give summary judgment where a party has no real prospect of succeeding.",
                    sourcepage="Part 24",
                    sourcefile="Part 24 - Summary judgment",
                    category="Civil Procedure Rules and Practice Directions",
                    subsection_id="24.3",
                )
            ]
        return [
            Document(
                id="pre",
                content="Construction disputes are subject to the pre-action protocol.",
                sourcepage="Pre-Action Protocol for the Construction and Engineering Disputes",
                sourcefile="Pre",
                category="Civil Procedure Rules and Practice Directions",
            )
        ]

    async def fake_get_sources_content(results, *_args, **_kwargs):
        return DataPoints(
            text=[
                {
                    "citation": result.sourcepage or result.id or "",
                    "content": result.content or "",
                    "sourcepage": result.sourcepage or "",
                    "sourcefile": result.sourcefile or "",
                    "category": result.category or "",
                }
                for result in results
            ],
            citations=[],
        )

    monkeypatch.setattr(chat_approach, "rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(chat_approach, "search", fake_search)
    monkeypatch.setattr(chat_approach, "get_sources_content", fake_get_sources_content)

    extra_info = await chat_approach.run_search_approach(
        messages=[
            {
                "role": "user",
                "content": "Before commencing construction proceedings, what pre-action steps apply, when can summary judgment be granted, and what Practice Direction 27B point is relevant?",
            }
        ],
        overrides={
            "retrieval_mode": "text",
            "semantic_ranker": True,
            "semantic_captions": False,
            "query_rewriting": True,
            "top": 5,
        },
        auth_claims={},
    )

    assert search_calls[0]
    assert "Practice Direction 27B" in search_calls
    # The dynamic CPR Part supplemental search uses the full rewrite text
    # (which contains "CPR 24.3") with a CPR category filter.
    assert any("24.3" in call for call in search_calls)
    assert extra_info.data_points.text is not None
    assert any(source["sourcefile"] == "Practice Direction 27B" for source in extra_info.data_points.text)
    assert any(source["sourcefile"] == "Part 24 - Summary judgment" for source in extra_info.data_points.text)


@pytest.mark.asyncio
async def test_compute_multimodal_embedding(monkeypatch, chat_approach):
    # Create a mock for the ImageEmbeddings.create_embedding_for_text method
    async def mock_create_embedding_for_text(self, q: str):
        # Return a mock vector
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    monkeypatch.setattr(ImageEmbeddings, "create_embedding_for_text", mock_create_embedding_for_text)

    # Create a mock ImageEmbeddings instance and set it on the chat_approach
    mock_image_embeddings = ImageEmbeddings(endpoint="https://mock-endpoint", token_provider=lambda: None)
    chat_approach.image_embeddings_client = mock_image_embeddings

    # Test the compute_multimodal_embedding method
    query = "What's in this image?"
    result = await chat_approach.compute_multimodal_embedding(query)

    # Verify the result is a VectorizedQuery with the expected properties
    assert isinstance(result, VectorizedQuery)
    assert result.vector == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert result.k == 50
    assert result.fields == "images/embedding"


@pytest.mark.asyncio
async def test_compute_multimodal_embedding_no_client():
    """Test that compute_multimodal_embedding raises ValueError when image_embeddings_client is not set."""
    # Create a chat approach without an image_embeddings_client
    chat_approach = ChatReadRetrieveReadApproach(
        search_client=SearchClient(endpoint="", index_name="", credential=AzureKeyCredential("")),
        search_index_name=None,
        knowledgebase_model=None,
        knowledgebase_deployment=None,
        knowledgebase_client=None,
        openai_client=None,
        chatgpt_model="gpt-35-turbo",
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
        # Explicitly set image_embeddings_client to None
        image_embeddings_client=None,
    )

    # Test that calling compute_multimodal_embedding raises a ValueError
    with pytest.raises(ValueError, match="Approach is missing an image embeddings client for multimodal queries"):
        await chat_approach.compute_multimodal_embedding("What's in this image?")


@pytest.mark.asyncio
async def test_chat_prompt_render_with_image_directive(chat_approach):
    """Verify DocFX style :::image directive is sanitized (replaced with [image]) during prompt rendering."""
    image_directive = (
        "activator-introduction.md#page=1: Intro text before image. "
        ':::image type="content" source="./media/activator-introduction/activator.png" '
        'alt-text="Diagram that shows the architecture of Fabric Activator."::: More text after image.'
    )

    async def build_sources():
        return await chat_approach.get_sources_content(
            [
                Document(
                    id="doc1",
                    content=image_directive.split(": ", 1)[1],
                    sourcepage="activator-introduction.md#page=1",
                    sourcefile="activator-introduction.md",
                )
            ],
            use_semantic_captions=False,
            include_text_sources=True,
            download_image_sources=False,
            user_oid=None,
        )

    data_points = await build_sources()

    messages = chat_approach.prompt_manager.build_conversation(
        system_template_path="chat_answer.system.jinja2",
        system_template_variables={
            "include_follow_up_questions": False,
            "image_sources": data_points.images,
            "citations": data_points.citations,
        },
        user_template_path="chat_answer.user.jinja2",
        user_template_variables={
            "user_query": "What is Fabric Activator?",
            "text_sources": data_points.text,
        },
        user_image_sources=data_points.images,
        past_messages=[],
    )
    assert messages
    # Find the user message containing Sources and verify placeholder
    combined = "\n".join([m["content"] for m in messages if m["role"] == "user"])
    # Expect triple colons escaped
    assert "&#58;&#58;&#58;image" in combined
    assert "activator-introduction/activator.png" in combined
    assert "Diagram that shows the architecture of Fabric Activator." in combined
    # Original unescaped sequence should be gone
    assert ":::image" not in combined


@pytest.mark.asyncio
async def test_get_sources_content_preserves_structured_full_content_for_highlighting(chat_approach):
    """full_content should retain subsection boundaries even when content is flattened for prompts."""

    doc = Document(
        id="part59-59_9",
        content=(
            "PART 59 - CIRCUIT COMMERCIAL COURT\n\n"
            "59.9 If particulars of claim are not served with the claim form, the claimant must serve them within 28 days.\n\n"
            "The court may extend time where appropriate."
        ),
        sourcepage="Part 59",
        sourcefile="Part 59 - Circuit Commercial Court",
        subsection_id="59.9",
    )

    data_points = await chat_approach.get_sources_content(
        [doc],
        use_semantic_captions=False,
        include_text_sources=True,
        download_image_sources=False,
        user_oid=None,
    )

    assert len(data_points.text) == 1
    source = data_points.text[0]
    assert isinstance(source, dict)
    assert source["content"] == (
        "PART 59 - CIRCUIT COMMERCIAL COURT 59.9 If particulars of claim are not served with the claim form, "
        "the claimant must serve them within 28 days. The court may extend time where appropriate."
    )
    assert source["full_content"] == (
        "PART 59 - CIRCUIT COMMERCIAL COURT\n\n"
        "59.9 If particulars of claim are not served with the claim form, the claimant must serve them within 28 days.\n\n"
        "The court may extend time where appropriate."
    )
    assert source["subsection_id"] == "59.9"


@pytest.mark.asyncio
async def test_get_sources_content_derives_subsection_metadata_when_field_missing(chat_approach):
    """subsection_id metadata should be derived from content when the index result omits the field."""

    doc = Document(
        id="part59-derived",
        content=(
            "PART 59 - CIRCUIT COMMERCIAL COURT\n\n"
            "59.9 If particulars of claim are not served with the claim form, the claimant must serve them within 28 days."
        ),
        sourcepage="Part 59",
        sourcefile="Part 59 - Circuit Commercial Court",
        subsection_id=None,
    )

    data_points = await chat_approach.get_sources_content(
        [doc],
        use_semantic_captions=False,
        include_text_sources=True,
        download_image_sources=False,
        user_oid=None,
    )

    assert len(data_points.text) == 1
    source = data_points.text[0]
    assert isinstance(source, dict)
    assert source["citation"].startswith("59.9, Part 59")
    assert source["subsection_id"] == "59.9"


@pytest.mark.asyncio
async def test_get_sources_content_splits_multi_subsection_documents_for_prompting(chat_approach):
    """Multi-subsection legal documents should become multiple text sources for accurate citations."""

    doc = Document(
        id="part59-multi",
        content=(
            "PART 59 - CIRCUIT COMMERCIAL COURT\n\n"
            "59.9 If particulars of claim are not served with the claim form, the claimant must serve them within 28 days.\n\n"
            "59.10 The defendant must file an acknowledgement of service within 14 days."
        ),
        sourcepage="Part 59",
        sourcefile="Part 59 - Circuit Commercial Court",
        subsection_id="59.9",
    )

    data_points = await chat_approach.get_sources_content(
        [doc],
        use_semantic_captions=False,
        include_text_sources=True,
        download_image_sources=False,
        user_oid=None,
    )

    assert len(data_points.text) == 2
    first = data_points.text[0]
    second = data_points.text[1]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    assert first["subsection_id"] == "59.9"
    assert second["subsection_id"] == "59.10"
    assert first["citation"].startswith("59.9, Part 59")
    assert second["citation"].startswith("59.10, Part 59")
    assert "59.9 If particulars of claim" in first["content"]
    assert "59.10 The defendant must file" in second["content"]
    assert data_points.citations == ["59.9, Part 59, Part 59 - Circuit Commercial Court"]


@pytest.mark.asyncio
async def test_get_sources_content_limits_subsection_expansion_to_adjacent_window(chat_approach):
    """Prompt sources should stay centered on the retrieved subsection instead of expanding an entire legal part."""

    doc = Document(
        id="part59-windowed",
        content=(
            "PART 59 - CIRCUIT COMMERCIAL COURT\n\n"
            "59.7 Case management directions.\n\n"
            "59.8 Pre-trial review requirements.\n\n"
            "59.9 If particulars of claim are not served with the claim form, the claimant must serve them within 28 days.\n\n"
            "59.10 The defendant must file an acknowledgement of service within 14 days.\n\n"
            "59.11 Service out of the jurisdiction requires permission in some cases."
        ),
        sourcepage="Part 59",
        sourcefile="Part 59 - Circuit Commercial Court",
        subsection_id="59.9",
    )

    data_points = await chat_approach.get_sources_content(
        [doc],
        use_semantic_captions=False,
        include_text_sources=True,
        download_image_sources=False,
        user_oid=None,
    )

    assert [source["subsection_id"] for source in data_points.text if isinstance(source, dict)] == ["59.8", "59.9", "59.10"]
    assert all("59.7" not in source["content"] for source in data_points.text if isinstance(source, dict))
    assert all("59.11" not in source["content"] for source in data_points.text if isinstance(source, dict))


@pytest.mark.asyncio
async def test_get_sources_content_caps_unfocused_multi_subsection_expansion(chat_approach):
    """Broad legal chunks without a usable subsection anchor should be capped to avoid flooding the prompt."""

    intro_lines = "\n".join([f"Intro line {index}" for index in range(1, 22)])

    doc = Document(
        id="pd31a-broad",
        content=(
            f"{intro_lines}\n\n"
            "1.1 First subsection.\n\n"
            "1.2 Second subsection.\n\n"
            "1.3 Third subsection.\n\n"
            "1.4 Fourth subsection.\n\n"
            "1.5 Fifth subsection.\n\n"
            "1.6 Sixth subsection."
        ),
        sourcepage="PD31A",
        sourcefile="Practice Direction 31A – Disclosure and Inspection",
        subsection_id="",
    )

    data_points = await chat_approach.get_sources_content(
        [doc],
        use_semantic_captions=False,
        include_text_sources=True,
        download_image_sources=False,
        user_oid=None,
    )

    assert [source["subsection_id"] for source in data_points.text if isinstance(source, dict)] == ["1.1", "1.2", "1.3", "1.4"]


@pytest.mark.asyncio
async def test_get_sources_content_deduplicates_repeated_subsections(chat_approach):
    """Repeated subsection labels from one document should collapse to a single prompt source."""

    doc = Document(
        id="commercial-dup",
        content=(
            "D.5.4 List of Common Ground and Issues\n\n"
            "Short heading stub.\n\n"
            "D.5.4 The List of Common Ground and Issues will be used as a case management tool at hearings.\n\n"
            "Longer explanatory text for the same subsection that should win over the stub."
        ),
        sourcepage="Commercial Court Guide",
        sourcefile="Commercial Court Guide",
        subsection_id="D.5.4",
    )

    data_points = await chat_approach.get_sources_content(
        [doc],
        use_semantic_captions=False,
        include_text_sources=True,
        download_image_sources=False,
        user_oid=None,
    )

    prompt_sources = [source for source in data_points.text if isinstance(source, dict)]
    assert len(prompt_sources) == 1
    assert prompt_sources[0]["subsection_id"] == "D.5.4"
    assert "case management tool at hearings" in prompt_sources[0]["content"]


@pytest.mark.asyncio
async def test_get_sources_content_downloads_images_from_images_container(chat_approach, monkeypatch):
    """Regression test: ensure image URLs in a non-default container download from that container."""

    called: dict[str, str] = {}

    async def fake_download_blob(blob_path: str, user_oid=None, container=None):
        called["blob_path"] = blob_path
        called["container"] = container
        assert user_oid is None
        return b"abc", {"content_settings": {"content_type": "image/png"}}

    monkeypatch.setattr(chat_approach.global_blob_manager, "download_blob", fake_download_blob)

    image_url = "https://examplestorage.blob.core.windows.net/images/doc1/page0/figure1.png"
    doc = Document(
        id="doc1",
        content="",
        sourcepage="doc1.pdf#page=1",
        sourcefile="doc1.pdf",
        images=[{"url": image_url}],
    )

    data_points = await chat_approach.get_sources_content(
        [doc],
        use_semantic_captions=False,
        include_text_sources=False,
        download_image_sources=True,
        user_oid=None,
    )

    assert called["container"] == "images"
    assert called["blob_path"] == "doc1/page0/figure1.png"
    assert data_points.images == [f"data:image/png;base64,{base64.b64encode(b'abc').decode('utf-8')}"]


def test_replace_all_ref_ids_unknown_fallback(chat_approach):
    """Test that unknown ref_ids remain unchanged (fallback case)."""
    answer = "This is an answer with [ref_id:999] that doesn't match any document or web result."
    documents = [
        Document(
            id="doc1",
            ref_id="1",
            content="Some content",
            sourcepage="page1.pdf",
            sourcefile="page1.pdf",
        )
    ]
    web_results = [
        WebResult(
            id="5",
            title="Web Result",
            url="https://example.com",
        )
    ]

    result = chat_approach.replace_all_ref_ids(answer, documents, web_results)

    # ref_id:999 doesn't exist in either documents or web_results, so it should remain unchanged
    assert "[ref_id:999]" in result
    assert result == "This is an answer with [ref_id:999] that doesn't match any document or web result."


def test_replace_all_ref_ids_mixed(chat_approach):
    """Test that ref_ids are replaced correctly for web, documents, and unknown refs."""
    answer = "Check [ref_id:1] and [ref_id:5] and also [ref_id:999]."
    documents = [
        Document(
            id="doc1",
            ref_id="1",
            content="Some content",
            sourcepage="page1.pdf",
            sourcefile="page1.pdf",
        )
    ]
    web_results = [
        WebResult(
            id="5",
            title="Web Result",
            url="https://example.com",
        )
    ]

    result = chat_approach.replace_all_ref_ids(answer, documents, web_results)

    # ref_id:1 should be replaced with document sourcepage
    assert "[page1.pdf]" in result
    # ref_id:5 should be replaced with web URL (web has priority)
    assert "[https://example.com]" in result
    # ref_id:999 doesn't exist, should remain unchanged
    assert "[ref_id:999]" in result
    assert result == "Check [page1.pdf] and [https://example.com] and also [ref_id:999]."


def test_replace_all_ref_ids_sharepoint_priority(chat_approach):
    """SharePoint URLs should be used when present."""

    answer = "See [ref_id:7] for the site link."
    documents = [
        Document(id="doc1", ref_id="7", sourcepage="page1.pdf", sourcefile="page1.pdf"),
    ]
    sharepoint_results = [
        SharePointResult(id="7", web_url="https://sharepoint.example.com/documents/7"),
    ]

    result = chat_approach.replace_all_ref_ids(answer, documents, [], sharepoint_results)

    # SharePoint extracts filename from URL (last part after /)
    assert result == "See [7] for the site link."


@pytest.mark.asyncio
async def test_get_sources_content_includes_sharepoint(chat_approach):

    documents = [
        Document(id="doc1", ref_id="1", sourcepage="page1.pdf", content="Doc content"),
    ]
    sharepoint_results = [
        SharePointResult(
            id="10",
            web_url="https://contoso.sharepoint.com/doc",
            content="SharePoint body",
            title="SharePoint Title",
            activity=ActivityDetail(id=3, number=1, type="remoteSharePoint", source="sharepoint", query="sp query"),
        )
    ]

    data_points = await chat_approach.get_sources_content(
        documents,
        use_semantic_captions=False,
        include_text_sources=True,
        download_image_sources=False,
        sharepoint_results=sharepoint_results,
    )

    # SharePoint extracts filename from URL (last part after /)
    assert "doc" in data_points.citations
    assert (
        data_points.external_results_metadata
        and data_points.external_results_metadata[0]["title"] == "SharePoint Title"
    )


def test_select_knowledgebase_client_priorities(chat_approach):
    primary = object()
    web = object()
    sharepoint = object()
    both = object()

    chat_approach.knowledgebase_client = primary
    chat_approach.knowledgebase_client_with_web = web
    chat_approach.knowledgebase_client_with_sharepoint = sharepoint
    chat_approach.knowledgebase_client_with_web_and_sharepoint = both

    selected, uses_web, uses_sp = chat_approach._select_knowledgebase_client(True, True)
    assert selected is both
    assert uses_web is True and uses_sp is True

    selected, uses_web, uses_sp = chat_approach._select_knowledgebase_client(True, False)
    assert selected is web and uses_web is True and uses_sp is False

    selected, uses_web, uses_sp = chat_approach._select_knowledgebase_client(False, True)
    assert selected is sharepoint and uses_web is False and uses_sp is True

    chat_approach.knowledgebase_client_with_web_and_sharepoint = None
    chat_approach.knowledgebase_client_with_sharepoint = None
    selected, uses_web, uses_sp = chat_approach._select_knowledgebase_client(True, True)
    assert selected is web and uses_web is True and uses_sp is False


def test_select_knowledgebase_client_requires_configuration(chat_approach):
    chat_approach.knowledgebase_client = None
    chat_approach.knowledgebase_client_with_web = None
    chat_approach.knowledgebase_client_with_sharepoint = None

    with pytest.raises(ValueError, match="Agentic retrieval requested but no knowledge base is configured"):
        chat_approach._select_knowledgebase_client(True, False)


@pytest.mark.asyncio
async def test_run_with_streaming_handles_non_stream_response(chat_approach, monkeypatch):
    extra_info = ExtraInfo(
        data_points=DataPoints(text=[], images=[], citations=[]),
        thoughts=[ThoughtStep("Final", None, props={})],
    )

    async def fake_completion():
        return Response(
            id="resp-stream",
            object="response",
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            created_at=0,
            model="gpt-4.1-mini",
            output=[
                ResponseOutputMessage(
                    id="msg-1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[{"type": "output_text", "text": "Answer text<<Follow up?>>", "annotations": []}],
                )
            ],
            status="completed",
            usage=ResponseUsage(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            ),
        )

    async def fake_run_until_final_call(messages, overrides, auth_claims, should_stream):
        assert should_stream is True
        return extra_info, fake_completion()

    monkeypatch.setattr(chat_approach, "run_until_final_call", fake_run_until_final_call)

    events = []
    async for event in chat_approach.run_with_streaming(
        messages=[{"role": "user", "content": "Hello"}],
        overrides={"suggest_followup_questions": True},
        auth_claims={},
        session_state="state",
    ):
        events.append(event)

    assert events[0]["context"] is extra_info
    assert events[1]["delta"]["content"] == "Answer text"
    assert events[2]["context"] is extra_info
    assert events[3]["context"]["followup_questions"] == ["Follow up?"]


@pytest.mark.asyncio
async def test_run_until_final_call_rejects_web_streaming(chat_approach):
    with pytest.raises(Exception, match="web source is enabled"):
        await chat_approach.run_until_final_call(
            messages=[{"role": "user", "content": "Hello"}],
            overrides={"use_agentic_knowledgebase": True, "use_web_source": True},
            auth_claims={},
            should_stream=True,
        )


# ---- Tests for related_aspects extraction in rewrite_query ----


@pytest.mark.asyncio
async def test_rewrite_query_extracts_related_aspects(chat_approach, monkeypatch):
    """When the LLM returns related_aspects in the tool call, they should be parsed into a list."""

    async def fake_create_response(*_args, **kwargs):
        return Response.model_validate(
            {
                "id": "rewrite-aspects",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4.1-mini",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc-1",
                        "call_id": "call-1",
                        "name": "search_sources",
                        "arguments": json.dumps(
                            {
                                "search_query": "CPR Part 31 standard disclosure",
                                "related_aspects": "CPR 31.3 right to inspect disclosed documents | CPR 31.12 specific disclosure order",
                            }
                        ),
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
                "status": "completed",
            },
            strict=False,
        )

    monkeypatch.setattr(chat_approach, "create_response", fake_create_response)

    result = await chat_approach.rewrite_query(
        prompt_template="query_rewrite.system.jinja2",
        prompt_variables={
            "user_query": "What documents do I have to share with the other side?",
            "past_messages": [],
        },
        overrides={},
        chatgpt_model=chat_approach.chatgpt_model,
        chatgpt_deployment=chat_approach.chatgpt_deployment,
        user_query="What documents do I have to share with the other side?",
        response_token_limit=100,
        tools=chat_approach.query_rewrite_tools,
        temperature=0.0,
        no_response_token=chat_approach.NO_RESPONSE,
    )

    assert result.related_aspects is not None
    assert len(result.related_aspects) == 2
    assert result.related_aspects[0] == "CPR 31.3 right to inspect disclosed documents"
    assert result.related_aspects[1] == "CPR 31.12 specific disclosure order"


@pytest.mark.asyncio
async def test_rewrite_query_returns_none_when_no_related_aspects(chat_approach, monkeypatch):
    """When the LLM omits related_aspects, the field should be None."""

    async def fake_create_response(*_args, **kwargs):
        return Response.model_validate(
            {
                "id": "rewrite-no-aspects",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4.1-mini",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc-1",
                        "call_id": "call-1",
                        "name": "search_sources",
                        "arguments": json.dumps({"search_query": "CPR Part 3 extend time"}),
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
                "status": "completed",
            },
            strict=False,
        )

    monkeypatch.setattr(chat_approach, "create_response", fake_create_response)

    result = await chat_approach.rewrite_query(
        prompt_template="query_rewrite.system.jinja2",
        prompt_variables={
            "user_query": "How can CPR Part 3 extend time?",
            "past_messages": [],
        },
        overrides={},
        chatgpt_model=chat_approach.chatgpt_model,
        chatgpt_deployment=chat_approach.chatgpt_deployment,
        user_query="How can CPR Part 3 extend time?",
        response_token_limit=100,
        tools=chat_approach.query_rewrite_tools,
        temperature=0.0,
        no_response_token=chat_approach.NO_RESPONSE,
    )

    assert result.related_aspects is None


# ---- Tests for adaptive merge threshold ----


def test_merge_threshold_is_stricter_for_focused_queries(chat_approach):
    """Queries with specific legal references (Part N, Rule N) should use the higher 50% threshold,
    dropping more marginal results."""
    query = "What does CPR Part 31 say about standard disclosure?"
    # Create 4 docs: 1 strong match, 1 moderate match, 2 weak matches
    docs = [
        Document(
            id="d1",
            content="31.6 Standard disclosure. A party's duty of standard disclosure is to disclose documents under CPR Part 31.",
            sourcepage="Part 31 – Disclosure and Inspection (p. 300)",
            sourcefile="Civil Procedure Rules",
            category="Civil Procedure Rules and Practice Directions",
            subsection_id="31.6",
        ),
        Document(
            id="d2",
            content="31.3 Right of inspection. Where a document has been disclosed, a party has a right to inspect it.",
            sourcepage="Part 31 – Disclosure and Inspection (p. 298)",
            sourcefile="Civil Procedure Rules",
            category="Civil Procedure Rules and Practice Directions",
            subsection_id="31.3",
        ),
        Document(
            id="d3",
            content="Overview of the court system and general rules for civil proceedings.",
            sourcepage="Introduction (p. 1)",
            sourcefile="Civil Procedure Rules",
            category="Civil Procedure Rules and Practice Directions",
            subsection_id="1.0",
        ),
        Document(
            id="d4",
            content="Annex B: Contact details for court offices.",
            sourcepage="Annex B (p. 500)",
            sourcefile="Civil Procedure Rules",
            category="Civil Procedure Rules and Practice Directions",
            subsection_id="B",
        ),
    ]

    merged = chat_approach._merge_documents_by_query_intent(query, docs, limit=4)
    merged_ids = [d.id for d in merged]
    # The focused query (with "Part 31" reference) uses the stricter 50% threshold,
    # so it keeps only the strongest match and drops weaker ones
    assert "d1" in merged_ids
    # The intro and annex docs should definitely be excluded
    assert "d3" not in merged_ids
    assert "d4" not in merged_ids


def test_merge_threshold_is_relaxed_for_broad_queries(chat_approach):
    """Queries without specific legal references should use the lower 35% threshold,
    keeping more secondary-topic results."""
    query = "What documents do I have to share with the other side?"
    # Create docs: 1 strong match, 1 related-but-secondary, 1 off-topic
    docs = [
        Document(
            id="d1",
            content="Standard disclosure requires each party to disclose documents on which it relies and which adversely affect its own case.",
            sourcepage="Part 31 – Disclosure and Inspection (p. 300)",
            sourcefile="Civil Procedure Rules",
            category="Civil Procedure Rules and Practice Directions",
            subsection_id="31.6",
        ),
        Document(
            id="d2",
            content="The right to inspect disclosed documents. Where a party has been given a list of documents they have a right to inspect those documents.",
            sourcepage="Part 31 – Disclosure and Inspection (p. 298)",
            sourcefile="Civil Procedure Rules",
            category="Civil Procedure Rules and Practice Directions",
            subsection_id="31.3",
        ),
        Document(
            id="d3",
            content="Annex B: Contact details for court offices.",
            sourcepage="Annex B (p. 500)",
            sourcefile="Civil Procedure Rules",
            category="Civil Procedure Rules and Practice Directions",
            subsection_id="B",
        ),
    ]

    merged = chat_approach._merge_documents_by_query_intent(query, docs, limit=4)
    merged_ids = [d.id for d in merged]
    # Broad query should keep secondary matches (d2 about inspection) due to relaxed threshold
    assert "d1" in merged_ids
    assert "d2" in merged_ids


# ---- Test for related_aspects tool schema ----


def test_query_rewrite_tool_includes_related_aspects_parameter(chat_approach):
    """The search_sources tool should include the related_aspects parameter."""
    tools = chat_approach.query_rewrite_tools
    assert len(tools) > 0
    search_sources_tool = tools[0]
    params = search_sources_tool["parameters"]["properties"]
    assert "related_aspects" in params
    assert params["related_aspects"]["type"] == "string"
