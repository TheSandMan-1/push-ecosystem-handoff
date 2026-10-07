"""Ollama is called natively so RoleScout controls the context window per request."""

import json

import httpx
import pytest

from rolescout.config import get_settings
from rolescout.extract.pipeline import ExtractStats, Extractor
from rolescout.extract.providers import LLMError, LocalProvider, make_provider

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


class FakeOllama:
    def __init__(self, models=("qwen2.5:14b-instruct",), reject_schema=False, version="0.40.0"):
        self.models = list(models)
        self.reject_schema = reject_schema
        self.version = version
        self.chats: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": self.version})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        if path == "/api/chat":
            body = json.loads(request.content)
            self.chats.append(body)
            if body["model"] not in self.models:
                return httpx.Response(404, json={"error": f"model \"{body['model']}\" not found, try pulling it first"})
            if self.reject_schema and isinstance(body["format"], dict):
                return httpx.Response(500, json={"error": "unsupported schema"})
            return httpx.Response(200, json={"model": body["model"], "message": {"role": "assistant", "content": '{"a": 1}'},
                                             "done": True, "prompt_eval_count": 321, "eval_count": 12})
        return httpx.Response(404)

    def provider(self, model="qwen2.5:14b-instruct", **kw):
        return LocalProvider("http://localhost:11434/v1", model,
                             client=httpx.Client(transport=httpx.MockTransport(self.handler)), **kw)


def test_detects_ollama_and_sets_context_window_per_request():
    ollama = FakeOllama()
    p = ollama.provider()
    r = p.complete_json("sys", "user", SCHEMA, "x")
    assert p.api == "ollama"
    body = ollama.chats[0]
    assert body["options"] == {"temperature": 0, "num_ctx": 8192}
    assert body["format"] == SCHEMA and body["stream"] is False
    assert r.data == {"a": 1} and (r.input_tokens, r.output_tokens) == (321, 12)


def test_context_window_is_configurable(monkeypatch):
    monkeypatch.setenv("ROLESCOUT_LOCAL_LLM_NUM_CTX", "16384")
    get_settings.cache_clear()
    p = make_provider("local", "qwen2.5:14b-instruct", get_settings())
    assert p.num_ctx == 16384


def test_schema_rejected_falls_back_to_json_mode_and_remembers():
    ollama = FakeOllama(reject_schema=True)
    p = ollama.provider()
    p.complete_json("s", "u", SCHEMA, "x")
    p.complete_json("s", "u", SCHEMA, "x")
    assert [c["format"] if isinstance(c["format"], str) else "schema" for c in ollama.chats] == ["schema", "json", "json"]


def test_preflight_reports_missing_model_with_the_fix():
    p = FakeOllama(models=["qwen2.5:14b-instruct"]).provider(model="qwen2.5:7b-instruct")
    with pytest.raises(LLMError) as exc:
        p.preflight()
    msg = str(exc.value)
    assert "ollama pull qwen2.5:7b-instruct" in msg and "--model qwen2.5:14b-instruct" in msg


def test_preflight_ok_and_untagged_name_matches_latest():
    notes = FakeOllama(models=["llama3.1:latest"]).provider(model="llama3.1").preflight()
    assert "ready" in notes[0] and "8,192" in notes[0]


def test_server_down_says_how_to_start_it():
    def down(request):
        raise httpx.ConnectError("refused")

    p = LocalProvider("http://localhost:11434/v1", "m", client=httpx.Client(transport=httpx.MockTransport(down)))
    with pytest.raises(LLMError, match="brew services start ollama"):
        p.preflight()


def test_timeout_is_explained_not_reported_as_unreachable():
    def slow(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.0"})
        raise httpx.ReadTimeout("timed out")

    p = LocalProvider("http://localhost:11434/v1", "m", timeout=180,
                      client=httpx.Client(transport=httpx.MockTransport(slow)))
    with pytest.raises(LLMError, match="longer than 180s"):
        p.complete_json("s", "u", SCHEMA, "x")


def test_non_ollama_server_uses_openai_endpoint():
    seen = []

    def lmstudio(request):
        seen.append(request.url.path)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"a": 2}'}}]})
        return httpx.Response(404)

    p = LocalProvider("http://localhost:1234/v1", "m", client=httpx.Client(transport=httpx.MockTransport(lmstudio)))
    assert p.complete_json("s", "u", SCHEMA, "x").data == {"a": 2}
    assert p.api == "openai" and "/v1/chat/completions" in seen


def test_extraction_run_falls_back_to_rules_when_model_server_is_down(session):
    def down(request):
        raise httpx.ConnectError("refused")

    provider = LocalProvider("http://localhost:11434/v1", "m", client=httpx.Client(transport=httpx.MockTransport(down)))
    ex = Extractor(get_settings(), extractor=provider, use_defaults=False)
    stats = ExtractStats()
    ex.preflight(stats)
    assert ex.extractor is None and stats.llm_errors == 1
    assert "brew services start ollama" in stats.errors[0]
