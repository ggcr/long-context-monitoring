"""Monitor transport for a self-hosted vLLM OpenAI-compatible server.

vLLM serves the same ``/v1/chat/completions`` shape OpenRouter does, so
:class:`VLLMClient` reuses :class:`~long_context_monitoring.openrouter.OpenRouterClient`
for retries, response parsing and tool turns, and changes only what differs:

- the endpoint comes from ``--base-url`` / ``BASE_URL`` (e.g.
  ``http://localhost:8000/v1``) instead of openrouter.ai;
- the API key is optional (``VLLM_API_KEY``), matching ``vllm serve --api-key``;
- reasoning effort goes in vLLM's top-level ``reasoning_effort`` field rather
  than OpenRouter's ``reasoning`` object, which vLLM would silently ignore;
- no OpenRouter ``provider`` routing object or ``:nitro`` slug rewriting —
  there is exactly one host, so a provider pin is refused instead of dropped;
- every call costs $0.00, so the cost gate and the recorded cost stay honest;
- no timeouts by default. The OpenRouter limits exist so a stuck call is not
  paid for twice; a local server bills nothing, and a long non-streamed call
  (a ~100k-token chunk at high effort) legitimately sends nothing until done.

The context window is not taken from the static table: :func:`served_model_info`
reads ``max_model_len`` from ``GET {base_url}/models``, so chunk budgets match
whatever ``--max-model-len`` the server was started with.

Results are recorded with ``backend = "vllm"`` and never pool with OpenRouter
or OpenAI-direct results of a same-named model.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from long_context_monitoring.openrouter import (
    LLMError,
    LLMResponse,
    OpenRouterClient,
    ToolTurn,
)

VLLM_BACKEND = "vllm"
BASE_URL_ENV = "BASE_URL"
API_KEY_ENV = "VLLM_API_KEY"


def resolve_base_url(explicit: str | None = None) -> str:
    """The server's ``/v1`` root: ``--base-url`` first, then ``BASE_URL``."""

    base = explicit or os.environ.get(BASE_URL_ENV)
    if not base:
        raise LLMError(
            f"--backend {VLLM_BACKEND} needs a server address; pass --base-url "
            f"or set {BASE_URL_ENV} (e.g. http://localhost:8000/v1)"
        )
    return base.rstrip("/")


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


@dataclass(frozen=True)
class ServedModel:
    """One entry of vLLM's ``/v1/models`` listing."""

    id: str
    max_model_len: int


def served_model_info(base_url: str, model: str, *, api_key: str | None = None,
                      timeout: float = 30.0) -> ServedModel:
    """Look ``model`` up in ``GET {base_url}/models`` and return its window.

    vLLM reports ``max_model_len`` per served model; it is the effective limit
    (prompt + completion) the server enforces, so it is the right budget for
    chunk planning. Fails loudly when the server is unreachable, the model is
    not served under that name, or the field is missing.
    """

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers=_auth_headers(api_key or os.environ.get(API_KEY_ENV)),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise LLMError(
            f"GET {base_url}/models failed with HTTP {error.code}"
        ) from error
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        raise LLMError(f"could not query {base_url}/models: {error}") from error

    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise LLMError(f"unexpected /models response from {base_url}: {data!r}")
    served = [entry for entry in entries if isinstance(entry, dict)]
    for entry in served:
        if entry.get("id") == model:
            length = entry.get("max_model_len")
            if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
                raise LLMError(
                    f"{base_url}/models lists {model!r} without a usable "
                    f"max_model_len ({length!r}); is this a vLLM server?"
                )
            return ServedModel(id=model, max_model_len=length)
    names = ", ".join(repr(entry.get("id")) for entry in served) or "none"
    raise LLMError(
        f"{base_url} does not serve {model!r}; pass one of its models with "
        f"--model: {names}"
    )


def count_tokens(base_url: str, model: str, texts: list[str], *,
                 api_key: str | None = None, workers: int = 8,
                 timeout: float = 120.0) -> list[int]:
    """Exact token counts for ``texts`` from vLLM's ``POST /tokenize``.

    No special tokens are added, so counts of separately tokenized pieces sum
    to within a token per join of the joined text. ``/tokenize`` sits at the
    server root, beside ``/v1``.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    headers = {"Content-Type": "application/json",
               **_auth_headers(api_key or os.environ.get(API_KEY_ENV))}

    def one(text: str) -> int:
        body = json.dumps({"model": model, "prompt": text,
                           "add_special_tokens": False}).encode("utf-8")
        request = urllib.request.Request(f"{root}/tokenize", data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise LLMError(f"POST {root}/tokenize failed with HTTP {error.code}") from error
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            raise LLMError(f"could not tokenize via {root}/tokenize: {error}") from error
        count = data.get("count") if isinstance(data, dict) else None
        if not isinstance(count, int):
            raise LLMError(f"unexpected /tokenize response from {root}: {str(data)[:200]}")
        return count

    if not texts:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(texts)))) as pool:
        return list(pool.map(one, texts))


class VLLMClient(OpenRouterClient):
    """:class:`OpenRouterClient` pointed at a vLLM server (see module docstring)."""

    def __init__(self, base_url: str | None = None, *,
                 api_key: str | None = None, timeout: float | None = None,
                 **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        # None blocks until the server answers: no socket timeout and no
        # wall-clock deadline (whatever the caller passed as hard_timeout).
        self.timeout = timeout  # type: ignore[assignment]
        self.hard_timeout = None
        self.base_url = resolve_base_url(base_url)
        self.url = f"{self.base_url}/chat/completions"
        # Optional: a server started without --api-key accepts any request.
        self.api_key = api_key or os.environ.get(API_KEY_ENV)

    def _require_key(self) -> str:
        return self.api_key or ""

    def _timeout_for(self, body_bytes: int) -> float | None:  # type: ignore[override]
        return self.timeout

    def _headers(self, key: str) -> dict:
        return {"Content-Type": "application/json", **_auth_headers(key)}

    @staticmethod
    def _refuse_provider(provider: dict[str, object] | None) -> None:
        if provider and (provider.get("order") or provider.get("quantizations")):
            raise LLMError(
                f"--backend {VLLM_BACKEND} serves one model from one host and "
                f"cannot honour an OpenRouter provider pin ({provider!r}); drop "
                "--provider/--quantization"
            )

    @staticmethod
    def _apply_sampling(payload: dict, temperature: float,
                        reasoning_effort: str | None) -> None:
        # Same rule as the OpenRouter client (temperature only when no effort
        # is sent) so a run differs from its OpenRouter twin only by host.
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        else:
            payload["temperature"] = temperature

    def complete(
        self,
        user: str,
        model: str,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        response_format: dict[str, object] | None = None,
        provider: dict[str, object] | None = None,
    ) -> LLMResponse:
        self._refuse_provider(provider)
        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens,
        }
        self._apply_sampling(payload, temperature, reasoning_effort)
        if response_format is not None:
            payload["response_format"] = response_format
        body = json.dumps(payload).encode("utf-8")
        data = self._post(body, self._headers(self._require_key()),
                          timeout=self._timeout_for(len(body)))
        return replace(self._parse(data, model), cost_usd=0.0)

    def chat(
        self,
        messages: list[dict],
        model: str,
        *,
        tools: list[dict] | None = None,
        max_tokens: int = 4000,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        provider: dict[str, object] | None = None,
    ) -> ToolTurn:
        self._refuse_provider(provider)
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        self._apply_sampling(payload, temperature, reasoning_effort)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        body = json.dumps(payload).encode("utf-8")
        data = self._post(body, self._headers(self._require_key()))
        return replace(self._parse_tool_turn(data, model), cost_usd=0.0)


__all__ = [
    "API_KEY_ENV",
    "BASE_URL_ENV",
    "ServedModel",
    "VLLMClient",
    "VLLM_BACKEND",
    "count_tokens",
    "resolve_base_url",
    "served_model_info",
]
