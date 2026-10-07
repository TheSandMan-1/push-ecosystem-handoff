"""Model providers.

- LocalProvider: any OpenAI-compatible server on your machine (Ollama, LM Studio,
  llama.cpp). Free per call; this is the bulk tier when you run it yourself.
- AnthropicProvider: the Claude API. Used for the hosted bulk tier and for the
  verification tier (default claude-sonnet-5-5).

Both return validated JSON plus token usage so every call can be costed.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

# USD per million tokens (input, output). Local models cost nothing per call.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Models that accept the server-side refusal fallback in its "default" form.
FALLBACK_DEFAULT_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
# Models that accept output_config.effort.
EFFORT_MODELS = {
    "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-sonnet-5",
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICES.get(model)
    if not price:
        return 0.0
    return round(input_tokens / 1e6 * price[0] + output_tokens / 1e6 * price[1], 6)


class LLMError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class LLMResult:
    data: dict[str, Any]
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)


class Provider(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, user: str, schema: dict, schema_name: str) -> LLMResult: ...


def _parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Small local models sometimes wrap JSON in prose; take the outermost object.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("Model did not return JSON")
        try:
            value = json.loads(text[start: end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMError("Model returned JSON that is not an object")
    return value


class LocalProvider:
    """OpenAI-compatible /chat/completions (Ollama, LM Studio, llama.cpp server, vLLM)."""

    name = "local"

    def __init__(self, base_url: str, model: str, api_key: str = "local", timeout: float = 180.0,
                 client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = client or httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})
        self._schema_supported: bool | None = None

    def _post(self, payload: dict) -> httpx.Response:
        try:
            return self._client.post(f"{self.base_url}/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Can't reach the local model server at {self.base_url} ({exc}). "
                "Is Ollama / LM Studio running?",
                retryable=True,
            ) from exc

    def complete_json(self, system: str, user: str, schema: dict, schema_name: str) -> LLMResult:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user + "\n\nRespond with a single JSON object that matches this schema:\n" + json.dumps(schema)},
        ]
        base = {"model": self.model, "messages": messages, "temperature": 0, "stream": False}
        started = time.monotonic()
        resp = None
        if self._schema_supported is not False:
            resp = self._post({**base, "response_format": {
                "type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}}})
            if resp.status_code in (400, 422):
                self._schema_supported = False
                resp = None
            else:
                self._schema_supported = True
        if resp is None:
            resp = self._post({**base, "response_format": {"type": "json_object"}})
        if resp.status_code == 404:
            raise LLMError(f"Model '{self.model}' not found on the local server. Pull it first (e.g. `ollama pull {self.model}`).")
        if resp.status_code != 200:
            raise LLMError(f"Local model server returned HTTP {resp.status_code}: {resp.text[:300]}",
                           retryable=resp.status_code >= 500)
        body = resp.json()
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected response shape from local model server") from exc
        usage = body.get("usage") or {}
        return LLMResult(
            data=_parse_json(text),
            provider=self.name,
            model=self.model,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=int((time.monotonic() - started) * 1000),
        )


class AnthropicProvider:
    """Claude via the official Anthropic SDK, with structured outputs."""

    name = "anthropic"

    def __init__(self, model: str, effort: str | None = "low", client: Any | None = None,
                 max_tokens: int = 8000):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client

    def complete_json(self, system: str, user: str, schema: dict, schema_name: str) -> LLMResult:
        import anthropic

        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort and self.model in EFFORT_MODELS:
            output_config["effort"] = self.effort
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        started = time.monotonic()
        try:
            if self.model in FALLBACK_DEFAULT_MODELS:
                # If a safety classifier declines, the API re-runs on its recommended
                # fallback model inside the same call instead of returning nothing.
                resp = self._client.beta.messages.create(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                resp = self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMError("Claude API rejected the credentials. Set ANTHROPIC_API_KEY or run `ant auth login`.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Claude model '{self.model}' not found") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude API rate limit hit", retryable=True) from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"Claude API rejected the request: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error HTTP {exc.status_code}", retryable=exc.status_code >= 500) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Can't reach the Claude API: {exc}", retryable=True) from exc
        latency = int((time.monotonic() - started) * 1000)
        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        if resp.stop_reason == "refusal":
            raise LLMError("Claude declined this request (refusal)")
        if resp.stop_reason == "max_tokens":
            raise LLMError("Claude hit max_tokens before finishing the JSON", retryable=True)
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise LLMError("Claude returned no text block")
        return LLMResult(
            data=_parse_json(text),
            provider=self.name,
            model=getattr(resp, "model", None) or self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency,
        )


def make_provider(kind: str, model: str, settings, *, effort: str | None = "low") -> Provider | None:
    kind = (kind or "").lower()
    if kind in ("", "none", "heuristic", "off"):
        return None
    if kind == "local":
        return LocalProvider(settings.local_llm_base_url, model, settings.local_llm_api_key)
    if kind in ("anthropic", "claude"):
        return AnthropicProvider(model, effort=effort)
    raise ValueError(f"Unknown model provider '{kind}'. Use heuristic, local, or anthropic.")
