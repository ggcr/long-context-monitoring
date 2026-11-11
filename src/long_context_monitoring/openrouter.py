"""Shared OpenRouter client — stdlib ``urllib`` only, no third-party deps.

One blocking POST to OpenRouter's OpenAI-compatible
``/api/v1/chat/completions`` endpoint. Both the monitor (one call per chunk)
and the attack designer (one call to design a plan) use this client, so it lives
outside either package. Tests inject a fake client implementing the same
``LLMClient`` protocol, so nothing here is exercised over the network in the test
suite.

Model slug / context window / pricing were checked against OpenRouter's docs
(Context7, Jul 2026): model ids are provider-prefixed slugs (``anthropic/…``),
the response is OpenAI-shaped (``choices[0].message.content`` + ``usage`` +
``id``). The exact newest Claude alias can drift; callers pass an explicit model
and the context/price tables fall back to conservative defaults for unlisted
slugs. Actual token usage comes back per call and is what gets recorded; the
tables only drive pre-run budget/cost estimates.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import signal
import threading
import warnings
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

MAX_BACKOFF_SECONDS = 300.0
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Transport id recorded in monitor results. Its counterpart lives in
# src/monitor/openai_direct.py; results under different backends never pool.
OPENROUTER_BACKEND = "openrouter"

# Context window (tokens) per model — for budgeting only. Conservative default
# for anything unlisted so prompts never overrun a real window. Sonnet 5 and
# Opus 4.8 both serve a 1M window (confirmed empirically: a single sonnet-5 call
# billed ~205k input tokens successfully); Haiku 4.5 caps at 200k.
DEFAULT_CONTEXT_TOKENS = 200_000
MODEL_CONTEXT_TOKENS: dict[str, int] = {
    "anthropic/claude-sonnet-5": 1_000_000,
    "anthropic/claude-opus-4-8": 1_000_000,
    "anthropic/claude-haiku-4.5": 200_000,
    "z-ai/glm-5.2": 1_000_000,
    "deepseek/deepseek-v4-pro": 1_000_000,
    "deepseek/deepseek-v4-flash": 1_000_000,
    "deepseek/deepseek-v4-flash-0731": 1_000_000,
    # 1.05M per the OpenRouter listing (unverified against a live call; the
    # largest chunk we plan today is ~440k, so the headroom is generous).
    "openai/gpt-5.6-luna": 1_047_576,
    "google/gemini-2.5-flash-lite": 1_048_576,
    "openai/gpt-4.1-nano": 1_047_576,
    "openai/gpt-4.1-mini": 1_047_576,
    "openai/gpt-6-luna": 1_050_000,
    "qwen/qwen3.7-flash": 1_000_000,
    "meta-llama/llama-4-scout": 1_310_720,
    "qwen/qwen3.5-flash-02-23": 1_000_000,
    "google/gemini-3.1-flash-lite": 1_048_576,
    "minimax/minimax-m3": 1_048_576,
    "z-ai/glm-5.3-flash": 1_048_576,
    "xiaomi/mimo-v2.6-flash": 1_050_000,
    "xiaomi/mimo-v2.6-pro": 1_050_000,
    "deepseek/deepseek-v4.1-flash": 1_048_576,
    "google/gemini-2.5-flash": 1_048_576,
    "qwen/qwen3.8-flash": 1_000_000,
    "poolside/laguna-s-2.1": 1_048_576,
}

# USD per (input, output) million tokens — estimates for the cost gate only; the
# recorded cost uses actual usage against these same numbers.
DEFAULT_PRICE_PER_MTOK = (3.0, 15.0)
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-5": (3.0, 15.0),
    "anthropic/claude-opus-4-8": (15.0, 75.0),
    "anthropic/claude-haiku-4.5": (1.0, 5.0),
    # Z.ai reference price; OpenRouter routes vary but this bounds the estimate.
    "z-ai/glm-5.2": (1.4, 4.4),
    "deepseek/deepseek-v4-pro": (0.6253, 1.2506),
    "deepseek/deepseek-v4-flash": (0.14, 0.28),
    "deepseek/deepseek-v4-flash-0731": (0.06, 0.18),
    # Use the non-batch base rate. OpenRouter promotions and batch discounts
    # must not make the synchronous preflight cost gate underestimate spend.
    "openai/gpt-5.6-luna": (0.20, 1.20),
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "openai/gpt-4.1-nano": (0.10, 0.40),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-6-luna": (0.10, 0.50),
    "qwen/qwen3.7-flash": (0.03, 0.13),
    "meta-llama/llama-4-scout": (0.10, 0.30),
    "qwen/qwen3.5-flash-02-23": (0.065, 0.26),
    "google/gemini-3.1-flash-lite": (0.25, 1.50),
    "minimax/minimax-m3": (0.24, 0.96),
    # GMICloud's fp8 rate, the host MODEL_DEFAULT_PROVIDER pins. Z.AI's own
    # endpoint ($0.150/$0.500) was tried first, on the reasoning that
    # first-party weights are the reference for a capability claim, but it
    # did not complete a single chunk in ten minutes at 32 concurrent
    # ~200k-token requests, while GMICloud served minimax-m3 at that same
    # concurrency and chunk size in two to three minutes. GMICloud is also
    # 40% cheaper on both rates.
    "z-ai/glm-5.3-flash": (0.09, 0.30),
    "xiaomi/mimo-v2.6-flash": (0.14, 0.28),
    "xiaomi/mimo-v2.6-pro": (0.435, 0.87),
    "deepseek/deepseek-v4.1-flash": (0.15, 0.60),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "qwen/qwen3.8-flash": (0.15, 0.47),
    "poolside/laguna-s-2.1": (0.09, 0.18),
}

GLM_52_MODEL = "z-ai/glm-5.2"
GLM_52_NITRO_MODEL = f"{GLM_52_MODEL}:nitro"

MODEL_DEFAULT_PROVIDER: dict[str, tuple[str, ...]] = {
    "deepseek/deepseek-v4-pro": ("Baidu",),
    # The endpoint slug, not the family name: with allow_fallbacks=false it
    # pins the fp8 endpoint itself, which is the variant priced in MODEL_PRICES.
    "deepseek/deepseek-v4-flash-0731": ("deepinfra/fp8",),
    "openai/gpt-5.6-luna": ("OpenAI",),
    "qwen/qwen3.7-flash": ("Alibaba",),
    "minimax/minimax-m3": ("GMICloud",),
    # Thirty-three hosts serve this model across fp4, nvfp4, fp8 and
    # unknown quantizations at $0.020 to $0.300 input, i.e. materially
    # different weights behind one slug. Z.AI is the first-party host.
    "z-ai/glm-5.3-flash": ("GMICloud",),
}

# Default OpenRouter provider.quantizations per model, applied only when the
# caller passes no explicit --quantization. Pairs with MODEL_DEFAULT_PROVIDER:
# pinning a provider alone only prefers that host, it does not guarantee which
# quantization it serves, so DeepSeek V4 Flash 0731's fp8 weights (the variant
# priced in MODEL_PRICES and compared across monitor runs) need their own pin.
MODEL_DEFAULT_QUANTIZATIONS: dict[str, tuple[str, ...]] = {
    "deepseek/deepseek-v4-flash-0731": ("fp8",),
    # GMICloud serves M3 as fp8 while other hosts serve fp4 and unknown
    # variants; pin the quantization too so a host-side change cannot
    # swap the weights under a batch that is already half scored.
    "minimax/minimax-m3": ("fp8",),
    # Pin the quantization as well: the cheap hosts for this slug are fp4
    # and nvfp4, and a host-side swap mid-batch would change the weights
    # under a comparison that is already half scored.
    "z-ai/glm-5.3-flash": ("fp8",),
}


def default_provider_order(
    model: str, explicit: tuple[str, ...] | list[str] | None
) -> tuple[str, ...]:
    """The provider.order to use: the caller's if given, else the model default.

    Returns an empty tuple when neither is set (OpenRouter routes freely). An
    explicit order is never overridden, so ``--provider`` stays authoritative.
    """
    if explicit:
        return tuple(explicit)
    return MODEL_DEFAULT_PROVIDER.get(base_model_slug(model), ())


def default_provider_quantizations(
    model: str, explicit: tuple[str, ...] | list[str] | None
) -> tuple[str, ...]:
    """The provider.quantizations to use: the caller's if given, else the model
    default.

    Mirrors :func:`default_provider_order`. Returns an empty tuple when neither
    is set (OpenRouter accepts any quantization).
    """
    if explicit:
        return tuple(explicit)
    return MODEL_DEFAULT_QUANTIZATIONS.get(base_model_slug(model), ())


def base_model_slug(model: str) -> str:
    """Return the concrete model behind an OpenRouter routing variant.

    Variants such as ``:nitro`` change provider routing, not the underlying
    model's context window or reference price.
    """

    return model.split(":", 1)[0]


def openrouter_model(model: str, *, use_nitro: bool = True) -> str:
    """Return the OpenRouter model slug to send on the wire.

    Unpinned GLM 5.2 calls use OpenRouter's ``:nitro`` throughput-routing
    variant. An exact ``provider.order`` pin already decides the host, and
    some hosts do not expose a ``:nitro`` endpoint, so those calls use the
    base model slug instead.
    """

    base = base_model_slug(model)
    if base != GLM_52_MODEL:
        return model
    if not use_nitro:
        return base
    if model == base:
        return GLM_52_NITRO_MODEL
    return model


def provider_pin(
    order: list[str] | tuple[str, ...] | None,
    *,
    require_parameters: bool = False,
    quantizations: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object] | None:
    """Build an OpenRouter ``provider`` routing object, or None when unpinned.

    ``order`` is the preferred provider list (e.g. ``["Anthropic"]``).
    ``allow_fallbacks`` is always false when an order is set so a pin failure
    surfaces instead of silently switching providers mid-experiment.
    ``quantizations`` (e.g. ``["fp8"]``) is a separate OpenRouter field —
    pinning a provider does not by itself pin which quantization it serves.
    """

    if not order and not require_parameters and not quantizations:
        return None
    payload: dict[str, object] = {}
    if order:
        payload["order"] = list(order)
        payload["allow_fallbacks"] = False
    if require_parameters:
        payload["require_parameters"] = True
    if quantizations:
        payload["quantizations"] = list(quantizations)
    return payload


class LLMError(RuntimeError):
    """Raised when the LLM call fails after retries (or config is missing)."""


@contextmanager
def _wall_clock_deadline(seconds: float | None):
    """Interrupt one HTTP request after an absolute wall-clock deadline."""

    if seconds is None:
        yield
        return
    if os.name == "nt" or threading.current_thread() is not threading.main_thread():
        warnings.warn(
            f"wall-clock request deadline of {seconds:g}s not applied: "
            f"POSIX alarms need the main thread on a POSIX platform. "
            f"Only the socket timeout bounds this request.",
            RuntimeWarning, stacklevel=3,
        )
        yield
        return

    def expire(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"wall-clock request deadline exceeded ({seconds:g}s)")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, expire)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0])


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    generation_id: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    cached_tokens: int = 0
    # The Responses API parser populates these fields. Chat-completions calls
    # keep the defaults below.
    cached_input_tokens: int = 0
    status: str | None = None
    incomplete_details: dict[str, object] | None = None


@dataclass
class ToolCall:
    """One tool the model asked to run this turn (name + parsed JSON args)."""

    id: str
    name: str
    arguments: dict


@dataclass
class ToolTurn:
    """One assistant turn from :meth:`OpenRouterClient.chat`.

    Either ``tool_calls`` is non-empty (the model wants tools run, then the
    conversation continued) or ``text`` holds its final answer.
    ``assistant_message`` is the raw message dict to append verbatim to the
    running ``messages`` list so follow-up tool results reference the right ids.

    ``cost_usd`` is OpenRouter's actual billed cost for this one turn (not an
    estimate) when the provider reports it, else ``None``. ``cached_tokens`` is
    the portion of ``input_tokens`` served from a prompt cache, when the
    provider reports it (0 otherwise) — a multi-turn tool loop resends the
    whole growing conversation every turn, so whether this is nonzero is the
    direct answer to "is caching actually reducing what we pay for that."
    """

    text: str
    tool_calls: list[ToolCall]
    assistant_message: dict
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = None
    cost_usd: float | None = None
    cached_tokens: int = 0
    # The provider label OpenRouter reports for this turn, so a pinned agentic
    # run can be checked with assert_pinned_provider like a single-shot call.
    provider: str | None = None


class LLMClient(Protocol):
    """The one method a caller needs from an LLM backend."""

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
        ...


def model_context_tokens(model: str) -> int:
    return MODEL_CONTEXT_TOKENS.get(base_model_slug(model), DEFAULT_CONTEXT_TOKENS)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD estimate for a call given token counts and the price table."""
    price_in, price_out = MODEL_PRICES.get(
        base_model_slug(model), DEFAULT_PRICE_PER_MTOK
    )
    return input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out


def assert_pinned_provider(
    response: LLMResponse,
    order: list[str] | tuple[str, ...] | None,
) -> None:
    """Fail when a pinned call returned a provider outside ``order``.

    OpenRouter should reject unsatisfiable pins when ``allow_fallbacks`` is
    false; this check still catches silent drift if a provider label changes or
    the pin object was dropped.
    """

    if not order:
        return
    # OpenRouter accepts endpoint-specific slugs such as ``baseten/fast`` in
    # provider.order, but the response reports only the provider label
    # (``BaseTen``).  The no-fallback routing object enforces the full endpoint
    # slug; this local check can therefore verify the provider family that the
    # response actually exposes without falsely rejecting a correct variant pin.
    def provider_family(name: str) -> str:
        base = name.split("/", 1)[0]
        return re.sub(r"[^a-z0-9]", "", base.casefold())

    allowed = {provider_family(name) for name in order}
    actual = (response.provider or "").strip()
    if not actual:
        raise LLMError(
            "OpenRouter response omitted provider while a pin was requested "
            f"(order={list(order)!r})"
        )
    if provider_family(actual) not in allowed:
        raise LLMError(
            f"OpenRouter provider {actual!r} is outside the pin "
            f"{list(order)!r}; refuse to continue with an unpinned route"
        )


# --- Prompt caching -----------------------------------------------------------
# Anthropic and Google routes use explicit cache breakpoints. Repeated monitor
# calls and tool loops reuse a stable prompt prefix, so those routes mark it.
# Other routes keep their existing request format without a breakpoint.
_CACHE_BREAKPOINT_FAMILIES = ("anthropic/", "google/")


def uses_cache_breakpoints(model: str) -> bool:
    """True when this model needs an explicit ``cache_control`` breakpoint.

    Anthropic and Gemini apply a caching discount only where the request marks a
    breakpoint; other providers on OpenRouter cache long identical prefixes
    automatically, so marking them is unnecessary and their requests are left
    untouched.
    """
    return base_model_slug(model).startswith(_CACHE_BREAKPOINT_FAMILIES)


def _mark_cache_control(message: dict) -> dict:
    """Return a copy of ``message`` with an ephemeral cache breakpoint on its
    last text block (converting a plain-string ``content`` to the block form).

    A provider caches everything up to and including a marked block, so marking a
    message's final text block caches that whole message and every message before
    it. A message with no text to mark — e.g. an assistant turn that is only
    ``tool_calls`` — is returned unchanged so the breakpoint lands on the seed.
    """
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return message
        marked = [{"type": "text", "text": content,
                   "cache_control": {"type": "ephemeral"}}]
        return {**message, "content": marked}
    if isinstance(content, list) and content:
        blocks = [dict(b) if isinstance(b, dict) else b for b in content]
        for block in reversed(blocks):
            if isinstance(block, dict) and block.get("type") == "text":
                block["cache_control"] = {"type": "ephemeral"}
                return {**message, "content": blocks}
    return message


def apply_cache_breakpoints(messages: list[dict], model: str) -> list[dict]:
    """Return a message list with cache breakpoints for cache-breakpoint models.

    Marks the first message (the stable seed / instruction-plus-transcript prefix
    that repeats on every rep and every tool turn) and, once the conversation has
    grown past it, the last message too (so accumulated turns are cached for the
    next round-trip). Anthropic permits up to four breakpoints; two is enough here
    and keeps the longest-matching-prefix behaviour simple. Returns the input
    unchanged for auto-caching providers, and never mutates the caller's list or
    its message dicts, so the agentic loop can re-mark a growing conversation
    fresh each turn.
    """
    if not messages or not uses_cache_breakpoints(model):
        return messages
    out = list(messages)
    out[0] = _mark_cache_control(out[0])
    if len(out) > 1:
        out[-1] = _mark_cache_control(out[-1])
    return out


def _is_timeout(error: Exception) -> bool:
    """Whether a network error is a socket timeout rather than a refused connection."""
    if isinstance(error, TimeoutError):
        return True
    reason = getattr(error, "reason", None)
    return isinstance(reason, TimeoutError) or "timed out" in str(error).lower()


class OpenRouterClient:
    """Blocking OpenRouter client. Key from ``OPENROUTER_API_KEY``.

    Retries 429 / 5xx up to ``max_attempts`` with exponential backoff; other
    HTTP errors and malformed responses raise :class:`LLMError` with the
    provider's message so failures are legible.
    """

    # Chat-completions endpoint every POST goes to. A subclass serving the same
    # OpenAI-shaped API elsewhere (``VLLMClient``) overrides it per instance.
    url: str = OPENROUTER_URL

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_attempts: int = 6,
        rate_limit_attempts: int = 20,
        hard_timeout: float | None = None,
        referer: str = "https://github.com/bgigurtsis/long-context-monitoring-project",
        title: str = "long-context-monitoring",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.rate_limit_attempts = rate_limit_attempts
        self.hard_timeout = hard_timeout
        self.referer = referer
        self.title = title

    def _require_key(self) -> str:
        if not self.api_key:
            raise LLMError(
                "OPENROUTER_API_KEY is not set — export it before running a real "
                "model call (use --dry-run to inspect prompts without a key)."
            )
        return self.api_key

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
        key = self._require_key()
        # A single user message — no separate system prompt (the whole
        # instruction is the caller's user text). Mark the whole prompt as a
        # cache prefix so a technique's repeated reps (byte-identical prompts)
        # read the transcript from cache on cache-breakpoint models.
        messages = apply_cache_breakpoints(
            [{"role": "user", "content": user}], model
        )
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # OpenRouter normalizes Anthropic/OpenAI thinking controls. Passing an
        # effort enables reasoning and sets its intensity; omit the field to
        # leave the provider default alone (Sonnet 5 defaults to high).
        #
        # Temperature and thinking are mutually exclusive on current Anthropic
        # models (Opus 4.8 / Sonnet 5 reject sampling params, and extended
        # thinking requires the default temperature) — so only send temperature
        # when reasoning is off, and let the provider default stand when it is on.
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        else:
            payload["temperature"] = temperature
        # Merge caller pin with structured-output require_parameters. Callers that
        # pass response_format always need require_parameters so routing cannot
        # silently land on a provider that drops the schema.
        merged_provider: dict[str, object] = dict(provider or {})
        if response_format is not None:
            payload["response_format"] = response_format
            merged_provider["require_parameters"] = True
        if merged_provider:
            payload["provider"] = merged_provider
        # Nitro selects a provider. An exact provider pin has already made
        # that choice and must use the base GLM endpoint for hosts that do
        # not expose the routing variant.
        payload["model"] = openrouter_model(
            model,
            use_nitro=not bool(merged_provider.get("order")),
        )
        body = json.dumps(payload).encode("utf-8")
        headers = self._headers(key)
        data = self._post(body, headers, timeout=self._timeout_for(len(body)))
        return self._parse(data, payload["model"])

    def _headers(self, key: str) -> dict:
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # OpenRouter attribution headers (optional but recommended).
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }

    def _timeout_for(self, body_bytes: int) -> float:
        """Socket timeout scaled to the request size.

        A 900k-token prompt takes far longer than the 300 s default to prefill;
        timing it out only to retry re-pays the same call. Allow roughly one
        extra second per thousand estimated tokens on top of the base timeout.
        """
        estimated_tokens = body_bytes / 2.5
        return max(self.timeout, 120.0 + estimated_tokens / 1000.0)

    def _classify_http_error(
        self, error: urllib.error.HTTPError
    ) -> tuple[str, LLMError, str]:
        """Sort one HTTP error into a retry class, a legible error, and its body.

        ``rate_limit`` was refused upstream before the model ran, so retrying
        it costs only time. ``server`` may already have been billed, so it
        draws on the smaller budget. ``fatal`` must not be retried at all.
        """
        detail = self._read_error(error)
        failure = LLMError(f"HTTP {error.code}: {detail}")
        if error.code == 429:
            return "rate_limit", failure, detail
        if 500 <= error.code < 600:
            return "server", failure, detail
        return "fatal", failure, detail

    def _attempt_once(self, body: bytes, headers: dict,
                      timeout: float | None) -> dict:
        """One POST, decoded. Raises for transport errors and error payloads.

        OpenRouter returns some upstream failures (rate limits, overload) under
        HTTP 200 with an ``error`` object, so that shape is raised here as an
        ``LLMError`` and retried by the caller exactly like an HTTP 429.
        """
        request = urllib.request.Request(self.url, data=body, headers=headers)
        socket_timeout = timeout or self.timeout
        deadline = None
        if self.hard_timeout is not None:
            # Scaled, not flat: the socket timeout already grows with the
            # request (about 1020 s for a 900k-token chunk), so a flat deadline
            # below that would kill the largest legitimate calls. Anything
            # beyond half again its own socket timeout is stuck, not slow.
            deadline = max(self.hard_timeout, socket_timeout * 1.5)
        with _wall_clock_deadline(deadline):
            with urllib.request.urlopen(
                request, timeout=socket_timeout
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        error = data.get("error")
        if isinstance(error, dict):
            raise LLMError(
                f"OpenRouter error payload (code={error.get('code')}): "
                f"{error.get('message') or error}"
            )
        return data

    def _post(self, body: bytes, headers: dict, timeout: float | None = None) -> dict:
        """POST one request body with retry/backoff; return the decoded JSON.

        Shared by :meth:`complete` and :meth:`chat`. Retries on HTTP 429/5xx,
        network errors, and HTTP-200 error payloads (upstream 429/overload). An
        error payload is raised as :class:`LLMError` inside the loop so it backs
        off exactly like the earlier inline behaviour, then the raw JSON is
        returned so each caller parses it (a plain completion vs a tool turn).
        """
        # A handler that falls off its end re-enters the loop; only the
        # LLMError branch needs an explicit ``continue``, because a ``raise``
        # follows it there.
        last_error: Exception | None = None
        timeouts = 0
        rate_limited = 0   # 429s, refused upstream: free to retry, own budget
        billed = 0         # everything else: may already have cost money
        attempt = 0        # scales the backoff delay only
        while True:
            try:
                return self._attempt_once(body, headers, timeout)
            except urllib.error.HTTPError as error:
                kind, failure, detail = self._classify_http_error(error)
                if kind == "fatal":
                    raise failure from error
                last_error = failure
                if kind == "rate_limit":
                    rate_limited += 1
                    exhausted = rate_limited >= self.rate_limit_attempts
                else:
                    billed += 1
                    exhausted = billed >= self.max_attempts
                if exhausted:
                    break
                self._backoff(
                    attempt, detail=detail if kind == "rate_limit" else None,
                    retry_after=self._retry_after_header(error))
                attempt += 1
            except (urllib.error.URLError, TimeoutError,
                    http.client.HTTPException, OSError) as error:
                last_error = LLMError(f"network error: {error}")
                if _is_timeout(error):
                    # A timeout on a large prompt may still be billed server
                    # side, so it gets one retry, not the full rate-limit budget.
                    timeouts += 1
                    if timeouts > 1:
                        raise LLMError(
                            f"OpenRouter call timed out twice ({self.timeout:.0f} s "
                            "each) and is not retried again: each attempt may be "
                            "billed"
                        ) from error
                billed += 1
                if billed >= self.max_attempts:
                    break
                self._backoff(attempt)
                attempt += 1
            except (json.JSONDecodeError, KeyError, IndexError) as error:
                raise LLMError(f"malformed OpenRouter response: {error}") from error
            except LLMError as error:
                # Error payloads returned under HTTP 200 (e.g. upstream 429)
                # should retry the same way as HTTP 429/5xx.
                detail = str(error)
                if "code=429" in detail or "rate" in detail.lower() or "overload" in detail.lower():
                    rate_limited += 1
                    last_error = error
                    if rate_limited >= self.rate_limit_attempts:
                        break
                    self._backoff(attempt, detail=detail)
                    attempt += 1
                    continue
                raise

        raise LLMError(
            f"OpenRouter call failed after {attempt + 1} attempt(s) "
            f"({rate_limited} rate-limited, {billed} possibly billed): {last_error}"
        )

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
    ) -> "ToolTurn":
        """One multi-message turn with optional tool-calling.

        Unlike :meth:`complete` (a single user message, one shot), this sends a
        full ``messages`` conversation plus ``tools`` and returns the
        assistant's turn — a final answer (``text``) or one/more ``tool_calls``
        the caller must execute and feed back. The caller drives the loop; the
        client does one round-trip. Reuses :meth:`_post`'s retry/backoff and the
        same reasoning/temperature and model-slug/provider handling as
        :meth:`complete`.
        """
        key = self._require_key()
        # Cache the seed (stable across every turn) and the latest turn (so the
        # growing conversation is cached for the next round-trip) on
        # cache-breakpoint models. Builds a fresh marked copy each turn without
        # mutating the caller's growing ``messages`` list.
        payload: dict = {
            "model": model,
            "messages": apply_cache_breakpoints(messages, model),
            "max_tokens": max_tokens,
        }
        # Temperature and thinking are mutually exclusive on current Anthropic
        # models; send temperature only when reasoning is off (see complete()).
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        else:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        merged_provider: dict[str, object] = dict(provider or {})
        if merged_provider:
            payload["provider"] = merged_provider
        payload["model"] = openrouter_model(
            model, use_nitro=not bool(merged_provider.get("order"))
        )
        body = json.dumps(payload).encode("utf-8")
        data = self._post(body, self._headers(key))
        return self._parse_tool_turn(data, payload["model"])

    @staticmethod
    def _parse_tool_turn(data: dict, model: str) -> "ToolTurn":
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"no choices in OpenRouter response: {data}")
        choice = choices[0]
        message = choice.get("message") or {}
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            args_raw = fn.get("arguments")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw or "{}")
                except json.JSONDecodeError:
                    args = {"_unparsed_arguments": args_raw}
            else:
                args = args_raw or {}
            calls.append(ToolCall(
                id=raw.get("id") or "",
                name=fn.get("name") or "",
                arguments=args if isinstance(args, dict) else {"value": args},
            ))
        usage = data.get("usage") or {}
        # Cached-token reporting differs by upstream provider: OpenAI-style puts
        # it under prompt_tokens_details.cached_tokens; Anthropic-style reports
        # cache_read_input_tokens directly on usage. Check both so caching is
        # visible regardless of which provider actually serves this model.
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached = prompt_details.get("cached_tokens")
        if cached is None:
            cached = usage.get("cache_read_input_tokens")
        return ToolTurn(
            text=message.get("content") or "",
            tool_calls=calls,
            assistant_message=message,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=choice.get("finish_reason"),
            cost_usd=(
                float(usage["cost"])
                if isinstance(usage.get("cost"), (int, float))
                and not isinstance(usage.get("cost"), bool)
                else None
            ),
            cached_tokens=int(cached or 0),
            provider=data.get("provider"),
        )

    @staticmethod
    def _retry_after_header(error: urllib.error.HTTPError) -> float | None:
        """Seconds from an HTTP ``Retry-After`` header; None when absent or a date."""
        value = error.headers.get("Retry-After") if error.headers else None
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None

    def _backoff(self, attempt: int, detail: str | None = None,
                 retry_after: float | None = None) -> None:
        """Sleep before the next try: 1s, 2s, 4s ... capped at MAX_BACKOFF_SECONDS."""
        delay = float(2 ** attempt)
        if detail:
            match = re.search(r'"retry_after_seconds"\s*:\s*(\d+(?:\.\d+)?)', detail)
            if match:
                delay = max(delay, float(match.group(1)))
        if retry_after is not None:
            delay = max(delay, retry_after)
        time.sleep(min(delay, MAX_BACKOFF_SECONDS))

    @staticmethod
    def _read_error(error: urllib.error.HTTPError) -> str:
        try:
            return error.read().decode("utf-8", "replace")[:500]
        except Exception:  # noqa: BLE001 - error body is best-effort
            return error.reason or "unknown error"

    @staticmethod
    def _parse(data: dict, model: str) -> LLMResponse:
        # Some providers return HTTP 200 with an error payload and no choices
        # (common for upstream 429/overloads). Treat those as call failures so
        # the retry loop can back off instead of looking like a parse bug.
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error
            code = error.get("code")
            raise LLMError(f"OpenRouter error payload (code={code}): {message}")
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"no choices in OpenRouter response: {data}")
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        usage = data.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        # Cached-token reporting differs by upstream provider: OpenAI-style puts
        # it under prompt_tokens_details.cached_tokens; Anthropic-style reports
        # cache_read_input_tokens directly on usage. Check both (see the
        # matching note in _parse_tool_turn).
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached = prompt_details.get("cached_tokens")
        if cached is None:
            cached = usage.get("cache_read_input_tokens")
        return LLMResponse(
            text=text,
            model=data.get("model") or model,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            generation_id=data.get("id"),
            provider=data.get("provider"),
            finish_reason=choice.get("finish_reason"),
            native_finish_reason=choice.get("native_finish_reason"),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
            cost_usd=(
                float(usage["cost"])
                if isinstance(usage.get("cost"), (int, float))
                and not isinstance(usage.get("cost"), bool)
                else None
            ),
            cached_tokens=int(cached or 0),
        )
