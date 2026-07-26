import pytest

from scripts.gate_common import (
    GateFailure,
    post_chat,
    response_answer,
    response_sources,
)


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"answer": "ok"}


class FakeClient:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None

    async def post(self, url: str, *, json: dict[str, object]) -> FakeResponse:
        self.payload = json
        return FakeResponse()


@pytest.mark.asyncio
async def test_post_chat_uses_context_overrides() -> None:
    client = FakeClient()

    result = await post_chat(
        client, "https://candidate.example.com", "What is the rule?", category="Court", top=7
    )

    assert result == {"answer": "ok"}
    assert client.payload == {
        "messages": [{"role": "user", "content": "What is the rule?"}],
        "context": {"overrides": {"top": 7, "include_category": "Court"}},
    }


def test_response_helpers_accept_backend_chat_shape() -> None:
    source = {"category": "Court", "sourcefile": "guide.pdf", "sourcepage": "1"}
    result = {
        "output_text": "The answer [1]",
        "message": {"content": "The answer [1]"},
        "context": {"data_points": {"text": [source]}},
    }

    assert response_answer(result) == "The answer [1]"
    assert response_sources(result) == [source]


def test_response_helpers_retain_legacy_chat_shape() -> None:
    source = {"sourcefile": "legacy.pdf", "sourcepage": "2"}

    assert response_answer({"answer": "Legacy answer"}) == "Legacy answer"
    assert response_sources({"sources": [source]}) == [source]


@pytest.mark.parametrize(
    "result",
    [
        {"choices": [{"message": {"content": "Chat completion answer"}}]},
        {"choices": [{"text": "Text completion answer"}]},
        {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Responses answer"}]}
            ]
        },
    ],
)
def test_response_answer_accepts_supported_api_response_shapes(result) -> None:
    assert response_answer(result).endswith("answer")


@pytest.mark.parametrize(
    ("helper", "result", "message"),
    [
        (response_answer, {"output_text": ""}, "Candidate chat response has no usable answer"),
        (response_sources, {"context": {"data_points": {"text": "invalid"}}}, "Candidate chat response has invalid sources"),
    ],
)
def test_response_helpers_fail_closed_for_malformed_payloads(helper, result, message) -> None:
    with pytest.raises(GateFailure, match=message):
        helper(result)