import json
from types import SimpleNamespace

import httpx
import pytest

from rolescout.extract.providers import AnthropicProvider, LLMError, LocalProvider, cost_usd

SCHEMA = {"type": "object"}


def test_local_provider_falls_back_to_json_object_when_schema_unsupported():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body["response_format"]["type"])
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, json={"error": "unsupported"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Sure! {\"a\": 1}"}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 3}})

    p = LocalProvider("http://local/v1", "qwen", client=httpx.Client(transport=httpx.MockTransport(handler)))
    r = p.complete_json("sys", "user", SCHEMA, "x")
    assert r.data == {"a": 1} and r.input_tokens == 10 and r.cost_usd == 0
    p.complete_json("sys", "user", SCHEMA, "x")
    assert seen == ["json_schema", "json_object", "json_object"]  # remembers lack of support


def test_local_provider_unreachable_is_retryable_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    p = LocalProvider("http://local/v1", "qwen", client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(LLMError) as exc:
        p.complete_json("s", "u", SCHEMA, "x")
    assert exc.value.retryable and "Ollama" in str(exc.value)


def test_local_provider_model_missing():
    p = LocalProvider("http://local/v1", "nope", client=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(404, json={}))))
    with pytest.raises(LLMError, match="ollama pull nope"):
        p.complete_json("s", "u", SCHEMA, "x")


class FakeMessages:
    def __init__(self, resp):
        self.resp, self.kwargs = resp, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.resp


def fake_client(text='{"ok": true}', stop="end_turn", model="claude-sonnet-5-5"):
    resp = SimpleNamespace(stop_reason=stop, model=model, usage=SimpleNamespace(input_tokens=500, output_tokens=100),
                           content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)])
    return SimpleNamespace(messages=FakeMessages(resp), beta=SimpleNamespace(messages=FakeMessages(resp)))


def test_anthropic_sonnet_uses_fallbacks_effort_and_structured_output():
    client = fake_client()
    r = AnthropicProvider("claude-sonnet-5-5", effort="medium", client=client).complete_json("sys", "u", SCHEMA, "x")
    kw = client.beta.messages.kwargs
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "medium"}
    assert "thinking" not in kw and "temperature" not in kw
    assert r.data == {"ok": True}
    assert r.cost_usd == cost_usd("claude-sonnet-5-5", 500, 100) == pytest.approx(0.002)


def test_anthropic_haiku_uses_plain_endpoint_without_effort():
    client = fake_client(model="claude-haiku-4-5")
    AnthropicProvider("claude-haiku-4-5", effort="low", client=client).complete_json("sys", "u", SCHEMA, "x")
    kw = client.messages.kwargs
    assert "fallbacks" not in kw and "effort" not in kw["output_config"]
    assert client.beta.messages.kwargs is None


@pytest.mark.parametrize("stop,match", [("refusal", "declined"), ("max_tokens", "max_tokens")])
def test_anthropic_bad_stop_reasons(stop, match):
    with pytest.raises(LLMError, match=match):
        AnthropicProvider("claude-sonnet-5-5", client=fake_client(stop=stop)).complete_json("s", "u", SCHEMA, "x")
