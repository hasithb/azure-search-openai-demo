import json

import pytest
from openai.types.responses import Response, ResponseOutputMessage, ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from approaches.approach import DataPoints, ExtraInfo


def build_answer_messages(chat_approach, *, selected_source_filter: str = ""):
    text_sources = [
        {
            "citation": "Part 29, Civil Procedure Rules",
            "content": "CPR Part 29 gives the court power to fix a case management conference.",
            "category": "Civil Procedure Rules and Practice Directions",
            "sourcepage": "Part 29",
        },
        {
            "citation": "Commercial Court Guide, Case management",
            "content": "The Commercial Court Guide gives additional case management guidance.",
            "category": "Commercial Court",
            "sourcepage": "Case management (p. 141)",
        },
    ]
    return chat_approach.prompt_manager.build_conversation(
        system_template_path="chat_answer.system.jinja2",
        system_template_variables={
            "include_follow_up_questions": False,
            "image_sources": [],
            "citations": ["1", "2"],
            "search_depth": "Standard",
            "available_sources": [
                "Civil Procedure Rules and Practice Directions",
                "Commercial Court Guide",
                "Patents Court Guide",
            ],
            "selected_source_filter": selected_source_filter,
        },
        user_template_path="chat_answer.user.jinja2",
        user_template_variables={
            "user_query": "What are the rules on case management conferences?",
            "text_sources": chat_approach.format_text_sources_for_prompt(text_sources),
        },
        user_image_sources=[],
        past_messages=[],
    )


def get_message_content(messages, role: str) -> str:
    return "\n".join(str(message["content"]) for message in messages if message["role"] == role)


def test_answer_prompt_includes_source_hierarchy_rules(chat_approach):
    messages = build_answer_messages(chat_approach)
    system_content = get_message_content(messages, "system")

    assert "IMPORTANT - Source hierarchy and cross-referencing:" in system_content
    assert "Lead your answer with CPR/Practice Direction sources" in system_content
    assert "Specific Court Guide selected" in system_content
    assert "Cross-guide contamination rule" in system_content
    assert "Never use one Court Guide's specific procedures to answer questions about a different court" in system_content


def test_answer_prompt_mentions_selected_source_filter(chat_approach):
    messages = build_answer_messages(chat_approach, selected_source_filter="Patents Court")
    system_content = get_message_content(messages, "system")

    assert "The user is currently filtering to: **Patents Court**." in system_content


def test_answer_prompt_renders_source_metadata_for_hierarchy(chat_approach):
    messages = build_answer_messages(chat_approach)
    user_content = get_message_content(messages, "user")

    assert "[1] (Category: Civil Procedure Rules and Practice Directions | Source: Part 29):" in user_content
    assert "[2] (Category: Commercial Court | Source: Case management (p. 141)):" in user_content


@pytest.mark.asyncio
async def test_run_until_final_call_passes_selected_filter_and_source_metadata(chat_approach, monkeypatch):
    captured: dict[str, object] = {}

    async def fake_run_search_approach(messages, overrides, auth_claims):
        assert messages[-1]["content"] == "What are the rules on case management conferences?"
        assert overrides["include_category"] == "Patents Court"
        assert auth_claims == {}
        return ExtraInfo(
            data_points=DataPoints(
                text=[
                    {
                        "citation": "Part 29, Civil Procedure Rules",
                        "content": "CPR Part 29 gives the court power to fix a case management conference.",
                        "category": "Civil Procedure Rules and Practice Directions",
                        "sourcepage": "Part 29",
                    },
                    {
                        "citation": "Commercial Court Guide, Case management",
                        "content": "The Commercial Court Guide gives additional case management guidance.",
                        "category": "Commercial Court",
                        "sourcepage": "Case management (p. 141)",
                    },
                ],
                images=[],
                citations=["Part 29, Civil Procedure Rules", "Commercial Court Guide, Case management"],
            )
        )

    async def fake_create_response(*args, **kwargs):
        captured["messages"] = args[2]
        return Response(
            id="test-answer",
            object="response",
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            created_at=0,
            model="gpt-4.1-mini",
            output=[
                ResponseOutputMessage(
                    id="msg-test",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[{"type": "output_text", "text": "Answer", "annotations": []}],
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

    monkeypatch.setattr(chat_approach, "run_search_approach", fake_run_search_approach)
    monkeypatch.setattr(chat_approach, "create_response", fake_create_response)

    _, chat_coroutine = await chat_approach.run_until_final_call(
        [{"role": "user", "content": "What are the rules on case management conferences?"}],
        {"include_category": "Patents Court"},
        {},
        False,
    )
    await chat_coroutine

    messages = captured["messages"]
    assert isinstance(messages, list)

    system_content = get_message_content(messages, "system")
    user_content = get_message_content(messages, "user")

    assert "The user is currently filtering to: **Patents Court**." in system_content
    assert "Specific Court Guide selected" in system_content
    assert "[1] (Category: Civil Procedure Rules and Practice Directions | Source: Part 29):" in user_content
    assert "[2] (Category: Commercial Court | Source: Case management (p. 141)):" in user_content


@pytest.mark.asyncio
async def test_run_until_final_call_adds_court_specific_only_hint_for_general_question(chat_approach, monkeypatch):
    captured: dict[str, object] = {}

    async def fake_run_search_approach(messages, overrides, auth_claims):
        assert messages[-1]["content"] == "What are the rules on case management conferences?"
        assert overrides == {}
        assert auth_claims == {}
        return ExtraInfo(
            data_points=DataPoints(
                text=[
                    {
                        "citation": "Commercial Court Guide, Case management",
                        "content": "The Commercial Court Guide gives additional case management guidance.",
                        "category": "Commercial Court",
                        "sourcepage": "Case management (p. 141)",
                    }
                ],
                images=[],
                citations=["Commercial Court Guide, Case management"],
            )
        )

    async def fake_create_response(*args, **kwargs):
        captured["messages"] = args[2]
        return Response(
            id="test-answer",
            object="response",
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            created_at=0,
            model="gpt-4.1-mini",
            output=[
                ResponseOutputMessage(
                    id="msg-test",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[{"type": "output_text", "text": "Answer", "annotations": []}],
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

    monkeypatch.setattr(chat_approach, "run_search_approach", fake_run_search_approach)
    monkeypatch.setattr(chat_approach, "create_response", fake_create_response)

    _, chat_coroutine = await chat_approach.run_until_final_call(
        [{"role": "user", "content": "What are the rules on case management conferences?"}],
        {},
        {},
        False,
    )
    await chat_coroutine

    messages = captured["messages"]
    assert isinstance(messages, list)

    system_content = get_message_content(messages, "system")
    assert "IMPORTANT - Request-specific mismatch:" in system_content
    assert "the retrieved sources are court guides only" in system_content
    assert "You MUST say that the retrieved material is **court-specific**" in system_content
