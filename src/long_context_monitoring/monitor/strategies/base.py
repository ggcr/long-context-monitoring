"""Shared plumbing for monitor strategies — budgeting, chunking, one call.

Every strategy is a thin file that composes the helpers here, so per-chunk
behaviour is identical across strategies (each view of a three-view run is
budgeted and chunked the same way; a view that fits its budget is one chunk).
The pieces:

- ``MonitorRequest`` — the immutable input a strategy receives (events already
  visibility-filtered by the CLI, plus model/render/budget knobs).
- ``chunk_transcript_budget`` — model window vs chunk size math: the per-chunk
  transcript token budget is ``WINDOW_FILL`` of the target (the monitor model's
  context window, or ``chunk_target_tokens`` when set) minus the *measured*
  prompt overhead (system + task + preamble + scoring instructions) and an
  output reserve. Overhead is counted against every chunk. So chunking only
  begins as a transcript nears the window; below it, everything is one chunk.
- ``split_chunks`` — greedy split by estimated rendered tokens, preferring
  turn/agent-return edges; an event larger than the budget is fragmented (never
  silently cut).
- ``score_chunk`` — render -> build prompt -> one LLM call -> parse -> checkpoint
  to the journal. Every view chunk goes through this, so behaviour can't drift.
- ``build_result`` — assemble the aggregate ``MonitorResult`` from chunk dicts.

Note: ``MonitorRequest``/``MonitorStrategy``/``ContextBudgetError`` are defined
here (not in ``__init__``) to avoid an import cycle with the concrete strategy
module; ``strategies/__init__`` re-exports them so callers still import them
from the package.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Protocol

from long_context_monitoring.monitor.events import Event
from long_context_monitoring.monitor.llm import (LLMClient, estimate_cost_usd, model_context_tokens)
from long_context_monitoring.openrouter import (OPENROUTER_BACKEND, assert_pinned_provider,
                            provider_pin)
from long_context_monitoring.prompts import (DEFAULT_MONITOR_TEMPLATE, PROMPT_TEMPLATE_HASHES,
                         build_monitor_prompt, prompt_version_for)
from long_context_monitoring.monitor.render import (CHARS_PER_TOKEN, Fragment, RenderOptions,
                                estimate_tokens, fragment_event, render_event,
                                render_units)
from long_context_monitoring.monitor.scoring import (MonitorResult, ScoreParseError, band_for_score,
                                 parse_monitor_reply)
from long_context_monitoring.monitor.visibility import is_diagnostic_visibility
from long_context_monitoring.monitor.task_context import FULL_TASK_CONTEXT_MODE

# Ceiling on the monitor's reply. Used two ways that must stay in step: sent to
# the model as ``max_tokens``, and reserved out of the input budget so a chunk's
# prompt plus its reply never overrun the context window (see
# chunk_transcript_budget). Sized well above the <=500-word reasoning + <score>
# tag the prompt asks for because reasoning models (GLM 5.2, Claude Sonnet 5)
# think by default and those thinking tokens count against the SAME
# ``max_tokens`` ceiling:
# at the old 2000 the thinking alone exhausted the budget, truncating the
# <reasoning> block mid-word or leaving the reply empty ("no <score> tag" / "empty
# monitor reply"). 16k leaves ample room for thinking + the tagged reply; against
# a 1M window the larger reserve costs nothing.
MONITOR_MAX_OUTPUT_TOKENS = 16000
MIN_CHUNK_BUDGET_TOKENS = 2000
# Fraction of the target window a chunk's transcript may fill. The remaining
# ~10% absorbs the prompt overhead, the output reserve, and estimator error, so
# a chunk built to this budget never overruns the real window. This is the
# only "safety margin" — proportional, because a flat 1k margin is meaningless
# against a 1M window.
WINDOW_FILL = 0.9
# A human prompt only forces an early chunk break once the current chunk is at
# least this full — keeps turn-aligned boundaries while packing chunks nearly
# full (only matters when a split is unavoidable; a transcript that fits is one
# chunk regardless — see split_chunks).
TURN_BOUNDARY_MIN_FILL = 0.9


class ResumeSettingsError(RuntimeError):
    """A journal line matches this prompt but was scored under other settings."""


class ContextBudgetError(RuntimeError):
    """Raised when a strategy's transcript cannot fit its token budget."""


def find_view(view_results: list[dict], name: str) -> dict:
    """The view summary called ``name``, also under an ensemble member prefix.

    ``ensemble`` namespaces every member's views as ``<model>/<name>`` so they
    never collide; a base strategy looking up its headline must accept both.
    """
    for view in view_results:
        if view["name"] == name or view["name"].endswith("/" + name):
            return view
    raise ContextBudgetError(
        f"no view named {name!r} among {[view['name'] for view in view_results]}"
    )


@dataclass(frozen=True)
class View:
    """One named event slice reviewed by a monitor strategy."""

    name: str
    label: str
    description: str
    events: list[Event]


def select_views(
    views: list[View],
    only_view: str | None,
    error_type: type[ContextBudgetError],
) -> list[View]:
    """Select one requested view while preserving a strategy's error type."""
    if only_view is None:
        return views
    selected = [view for view in views if view.name == only_view]
    if not selected:
        available = ", ".join(view.name for view in views) or "(none)"
        raise error_type(
            f"view {only_view!r} is not available for this transcript "
            f"(available: {available})"
        )
    return selected


def summarize_views(chunks: list[dict]) -> list[dict]:
    """Build one max-aggregate record per view, preserving strategy order."""
    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    for chunk in chunks:
        name = chunk["view"]
        if name not in grouped:
            grouped[name] = []
            order.append(name)
        grouped[name].append(chunk)

    views: list[dict] = []
    for name in order:
        view_chunks = grouped[name]
        top = max(view_chunks, key=lambda item: item["score"])
        views.append({
            "name": name,
            "label": top["view_label"],
            "index": top["view_index"],
            "score": top["score"],
            "band": top["band"],
            "reasoning": top["reasoning"],
            "chunk_count": len(view_chunks),
            "event_count": sum(chunk["event_count"] for chunk in view_chunks),
            "event_start": view_chunks[0]["event_start"],
            "event_end": view_chunks[-1]["event_end"],
            "chunk_indices": [chunk["index"] for chunk in view_chunks],
        })
    return views


@dataclass
class MonitorRequest:
    """Everything a strategy needs. ``events`` is already visibility-filtered."""

    events: list[Event]
    transcript_name: str
    task_description: str
    visibility: str
    render_options: RenderOptions
    model: str
    # None => use the monitor model's full context window (the default). An
    # explicit value caps the per-chunk target below the window, for experiments.
    chunk_target_tokens: int | None = None
    temperature: float = 0.0
    objectives: str | None = None
    max_output_tokens: int = MONITOR_MAX_OUTPUT_TOKENS
    only_view: str | None = None
    # OpenRouter reasoning.effort; None leaves the provider default alone
    # (Sonnet 5 defaults to high). Spec/CLI usually set this explicitly.
    reasoning_effort: str | None = None
    # OpenRouter provider.order pin; empty means unpinned routing.
    provider: tuple[str, ...] = ()
    # OpenRouter provider.quantizations pin (e.g. ("fp8",)); empty means any
    # quantization. Independent of ``provider`` — pinning a provider does not
    # by itself pin which quantization it serves.
    provider_quantizations: tuple[str, ...] = ()
    # Non-secret run provenance used by strategies when stamping results.
    run_metadata: dict = field(default_factory=dict)
    # Ensemble-only knobs (ignored by every other strategy): the OpenRouter
    # models to run the same base strategy under, and which registered
    # strategy key to run per model. Empty/default lets EnsembleMonitor pick
    # its own default trio and "intervals".
    ensemble_models: tuple[str, ...] = ()
    ensemble_base_strategy: str = "intervals"
    # How many times an agentic strategy repeats its investigation inside ONE
    # invocation (sandbox/scratchpad only; every other strategy ignores it).
    # Default 1 so a repetition means the same thing for every strategy — one
    # invocation, one score — and the rep count is owned by the caller via
    # --repetition rather than being baked into two strategies.
    internal_repetitions: int = 1
    # Per-view ``<task_description>`` text keyed by view name, built by
    # ``src.monitor.task_context``. Empty means every view gets the one
    # constant ``task_description`` above.
    view_task_descriptions: dict = field(default_factory=dict)
    # Recorded verbatim in every result; scores under different modes are
    # never pooled.
    task_context_mode: str = FULL_TASK_CONTEXT_MODE
    chunk_concurrency: int = 1
    # Which transport served the calls: "openrouter" (the default),
    # "openai-direct", or "vllm". Recorded in every result and part of a batch's
    # compatibility key — the same model served through a router and served
    # first-party is not the same monitor condition, so the two never pool.
    backend: str = OPENROUTER_BACKEND
    # Which scoring template renders each chunk prompt: ``monitor.md`` unless
    # the CLI's ``--monitor-template`` says otherwise. Only the strategies that
    # render through ``build_chunk_prompt`` honour it; the CLI refuses any
    # other strategy with a non-default template rather than ignoring it.
    monitor_template: str = DEFAULT_MONITOR_TEMPLATE
    # The monitor model's context window as reported by the serving host
    # (vLLM's ``max_model_len``). None falls back to the static per-model table,
    # which is right for OpenRouter/OpenAI where the window is a known constant.
    context_tokens: int | None = None
    # The OpenAI-compatible ``/v1`` root a self-hosted backend (vLLM) serves
    # from; None for hosted backends whose endpoint is fixed.
    base_url: str | None = None

    @property
    def context_window(self) -> int:
        """Tokens the monitor model can hold: the host's report, else the table."""
        if self.context_tokens is not None:
            return self.context_tokens
        return model_context_tokens(self.model)

    @property
    def objectives_provided(self) -> bool:
        return self.objectives is not None


class MonitorStrategy(Protocol):
    """A named, versioned scoring approach over a :class:`MonitorRequest`."""

    name: str  # versioned id recorded verbatim in results (e.g. "intervals-v1")

    def plan(self, request: MonitorRequest) -> list[ChunkPlan]:
        ...

    def run(self, request: MonitorRequest, client: LLMClient, journal,
            plans: list[ChunkPlan] | None = None) -> MonitorResult:
        ...


def prompt_overhead_tokens(request: MonitorRequest,
                           view_info: dict | None = None) -> int:
    """Estimated tokens of everything in a chunk prompt except the transcript.

    Built with an empty transcript and a 2-chunk ``chunk_info`` so the preamble
    is included — this is the fixed cost charged against every chunk's budget.
    """
    user = build_monitor_prompt(
        request.task_description, "",
        objectives=request.objectives,
        chunk_info={"index": 1, "total": 2},
        view_info=view_info,
        template=request.monitor_template,
    )
    return estimate_tokens(user)


def chunk_transcript_budget(request: MonitorRequest,
                            view_info: dict | None = None) -> int:
    """Per-chunk token budget for transcript text (overhead already removed).

    The target is the monitor model's context window (or ``chunk_target_tokens``
    if set, capped at the window). A chunk fills at most ``WINDOW_FILL`` of that
    target; from that we subtract the measured prompt overhead and the reply
    reserve (``request.max_output_tokens`` — the exact ceiling the reply is
    allowed, so input + reply can never overrun the window). On a 1M window this
    lands near ~884k transcript tokens per chunk, so chunking only kicks in as a
    transcript approaches the window — and when a split is unavoidable, chunks
    pack nearly full rather than at 60%.
    """
    window = request.context_window
    target = window if request.chunk_target_tokens is None else min(
        request.chunk_target_tokens, window)
    overhead = prompt_overhead_tokens(request, view_info=view_info)
    budget = int(target * WINDOW_FILL) - overhead - request.max_output_tokens
    return max(budget, MIN_CHUNK_BUDGET_TOKENS)


def _fragment_chars(budget_tokens: int) -> int:
    """Chars per fragment so a fragment (+ its header) fits the token budget.

    Converts the token budget back to characters with the same ratio
    ``estimate_tokens`` uses, minus a header allowance; otherwise a fragment
    sized in chars would re-estimate above the budget it was meant to fit.
    """
    return max(1000, int(budget_tokens * CHARS_PER_TOKEN) - 800)


def _unit_tokens(unit, options: RenderOptions) -> int:
    if isinstance(unit, Fragment):
        # A fragment's body is its slice; header adds a little, ignored here.
        return estimate_tokens(unit.text)
    return estimate_tokens(render_event(unit, options))


def _is_turn_start(unit) -> bool:
    event = unit.event if isinstance(unit, Fragment) else unit
    return event.kind == "human_prompt"


def _unit_event_id(unit) -> str:
    return unit.event.event_id if isinstance(unit, Fragment) else unit.event_id


# Kinds that constitute an agent *action* the monitor can judge. A chunk with none
# of these (e.g. only human prompts + a compaction summary) has nothing to review.
_ACTION_KINDS = frozenset({"tool_call", "tool_result", "agent_spawn", "agent_return"})


def chunk_has_actions(units: list) -> bool:
    """True if any unit is an agent action (tool call/result or sub-agent spawn/return)."""
    for unit in units:
        event = unit.event if isinstance(unit, Fragment) else unit
        if event.kind in _ACTION_KINDS:
            return True
    return False


def split_chunks(events: list[Event], options: RenderOptions,
                 budget_tokens: int) -> list[list]:
    """Greedily pack events into <= ``budget_tokens`` chunks (Events/Fragments).

    An event that alone exceeds the budget is fragmented up front (each fragment
    fits), then packing proceeds greedily: close the current chunk before an
    event would overflow it, and prefer to break at a human-prompt boundary once
    a chunk is reasonably full. Nothing is dropped — every event's text ends up
    in exactly one chunk (as itself or as ordered fragments).
    """
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")

    units: list = []
    for event in events:
        if _unit_tokens(event, options) > budget_tokens:
            units.extend(fragment_event(event, _fragment_chars(budget_tokens)))
        else:
            units.append(event)

    # If the whole (fragmented) transcript fits one budget, emit a single chunk.
    # The turn-boundary preference below only decides *where* to split when a
    # split is unavoidable; applying it to a transcript that already fits would
    # gratuitously carve off a tiny trailing chunk — extra API calls and failure
    # surface, plus the max-aggregation length bias — for no coverage gain. A
    # view that fits its budget is therefore scored in exactly one call.
    total_tokens = sum(_unit_tokens(unit, options) for unit in units)
    if total_tokens <= budget_tokens:
        return [units] if units else []

    chunks: list[list] = []
    current: list = []
    current_tokens = 0
    for unit in units:
        utok = _unit_tokens(unit, options)
        must_break = bool(current) and current_tokens + utok > budget_tokens
        prefer_break = (
            bool(current)
            and _is_turn_start(unit)
            and current_tokens >= budget_tokens * TURN_BOUNDARY_MIN_FILL
        )
        if must_break or prefer_break:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += utok
    if current:
        chunks.append(current)
    return chunks


def build_chunk_prompt(units: list, request: MonitorRequest,
                       chunk_info: dict,
                       view_info: dict | None = None) -> tuple[str, str, int]:
    """Render a chunk and build its ``(user, prompt_hash, tokens_est)``.

    The monitor instruction is a single user message (no separate system prompt);
    see ``prompts.build_monitor_prompt``. Shared by ``score_chunk`` and the CLI
    (dry-run + cost estimate) so the prompt a dry-run shows is exactly the prompt
    a real run sends and hashes.
    """
    text = render_units(units, request.render_options)
    user = build_monitor_prompt(
        request.task_description, text,
        objectives=request.objectives, chunk_info=chunk_info,
        view_info=view_info,
        template=request.monitor_template,
    )
    prompt_hash = hashlib.sha256(user.encode("utf-8")).hexdigest()
    tokens_est = estimate_tokens(user)
    return user, prompt_hash, tokens_est


@dataclass
class ChunkPlan:
    """One planned chunk: its units plus everything derivable before scoring.

    Built once (by ``build_chunk_plan`` / ``plan_chunks``) and consumed by both
    the CLI (pre-run summary + dry-run) and ``score_chunk``. Rendering the prompt
    and deciding whether the chunk has actions to review are done here, so the
    skip/score decision is made exactly once and the prompt is rendered once
    rather than re-derived at each site.
    """

    index: int
    units: list
    has_actions: bool
    user_prompt: str
    prompt_hash: str
    prompt_tokens_est: int
    event_start: str | None
    event_end: str | None
    event_count: int
    view: str = "whole"
    view_label: str = "whole trajectory"
    view_index: int = 1
    view_total: int = 1
    view_chunk_index: int = 1
    view_chunk_total: int = 1
    # Set only by EnsembleMonitor: which model this plan is scored against, so
    # the CLI's cost estimate can price each member at its own rate instead of
    # the request's nominal model. None everywhere else (falls back to
    # request.model).
    model: str | None = None


def build_chunk_plan(units: list, request: MonitorRequest,
                     chunk_info: dict,
                     view_info: dict | None = None) -> ChunkPlan:
    """Render one chunk's prompt and derive its skip/score inputs, once."""
    user, prompt_hash, prompt_tokens_est = build_chunk_prompt(
        units, request, chunk_info, view_info=view_info,
    )
    view_info = view_info or {}
    return ChunkPlan(
        index=chunk_info.get("index", 1),
        units=units,
        has_actions=chunk_has_actions(units),
        user_prompt=user,
        prompt_hash=prompt_hash,
        prompt_tokens_est=prompt_tokens_est,
        event_start=_unit_event_id(units[0]) if units else None,
        event_end=_unit_event_id(units[-1]) if units else None,
        event_count=len(units),
        view=str(view_info.get("name") or "whole"),
        view_label=str(view_info.get("label") or "whole trajectory"),
        view_index=int(view_info.get("index") or 1),
        view_total=int(view_info.get("total") or 1),
        view_chunk_index=int(chunk_info.get("index") or 1),
        view_chunk_total=int(chunk_info.get("total") or 1),
    )


def plan_chunks(request: MonitorRequest,
                view_info: dict | None = None) -> list[ChunkPlan]:
    """Split the transcript into budget-sized chunks and plan each one.

    Folds the budget split (``split_chunks``), prompt render (``build_chunk_prompt``)
    and no-action check (``chunk_has_actions``) into a single list of ChunkPlans,
    so the CLI and the chunked strategy plan identically and render prompts once.
    """
    budget = chunk_transcript_budget(request, view_info=view_info)
    chunk_unit_lists = split_chunks(request.events, request.render_options, budget)
    total = len(chunk_unit_lists)
    plans: list[ChunkPlan] = []
    for index, units in enumerate(chunk_unit_lists, start=1):
        plans.append(build_chunk_plan(
            units, request, {"index": index, "total": total},
            view_info=view_info,
        ))
    return plans


def plan_views(request: MonitorRequest, views: list[View]) -> list[ChunkPlan]:
    """Plan every named view and assign run-global chunk indices."""
    plans: list[ChunkPlan] = []
    for view_index, view in enumerate(views, start=1):
        view_request = replace(request, events=view.events)
        if request.view_task_descriptions:
            # Fail closed: a view with no registered brief must never fall
            # back to the full brief under the per-interval mode label.
            if view.name not in request.view_task_descriptions:
                raise ValueError(
                    f"no task brief registered for view {view.name!r}; "
                    f"known views: {sorted(request.view_task_descriptions)}"
                )
            view_request = replace(
                view_request,
                task_description=request.view_task_descriptions[view.name],
            )
        view_info = {
            "name": view.name,
            "label": view.label,
            "description": view.description,
            "index": view_index,
            "total": len(views),
        }
        plans.extend(plan_chunks(view_request, view_info=view_info))

    # Journal prompt filenames use run-global indices; positions within each
    # view remain available through view_chunk_index.
    for global_index, plan in enumerate(plans, start=1):
        plan.index = global_index
    return plans


def _chunk_record(request: MonitorRequest, plan: ChunkPlan, score: int,
                  reasoning: str, usage) -> dict:
    """Assemble one chunk-result dict from a plan + its score.

    ``usage`` is the LLM response (real token counts, model, generation ids) for a
    scored chunk, or ``None`` for a skipped no-action chunk — which spent no call,
    so its token counts are 0 and it keeps the configured model name. Both callers
    go through here, so the record schema exists in exactly one place (this is what
    replaces the per-field ``if response else`` ternaries).
    """
    if usage is None:
        input_tokens = 0
        output_tokens = 0
        cost_usd = 0.0
        generation_id = None
        provider = None
        model = request.model
        cached_tokens = 0
    else:
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        # Provider variants can differ materially from the model's reference
        # price. Prefer OpenRouter's billed cost when the response includes it;
        # retain the deterministic table estimate for providers that omit it.
        if usage.cost_usd is not None:
            cost_usd = usage.cost_usd
        else:
            cost_usd = estimate_cost_usd(
                usage.model, usage.input_tokens, usage.output_tokens
            )
        generation_id = usage.generation_id
        provider = usage.provider
        model = usage.model
        # 0 for a plain single-shot LLMResponse whose provider never reported
        # caching; an agentic strategy overwrites this with its own summed
        # total after calling _chunk_record (a single call's usage object
        # cannot carry a multi-turn loop's total).
        cached_tokens = getattr(usage, "cached_tokens", 0) or 0
    return {
        "index": plan.index,
        "view": plan.view,
        "view_label": plan.view_label,
        "view_index": plan.view_index,
        "view_total": plan.view_total,
        "view_chunk_index": plan.view_chunk_index,
        "view_chunk_total": plan.view_chunk_total,
        "event_start": plan.event_start,
        "event_end": plan.event_end,
        "event_count": plan.event_count,
        "score": score,
        "band": band_for_score(score),
        "reasoning": reasoning,
        "prompt_hash": plan.prompt_hash,
        "prompt_tokens_est": plan.prompt_tokens_est,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "cached_tokens": cached_tokens,
        "generation_id": generation_id,
        "provider": provider,
        "model": model,
        "status": "completed",
    }


def resume_settings(request: MonitorRequest) -> dict:
    """The settings a journal line must match before its score is reused.
    Why: the prompt hash covers the transcript and prompt text only, so a
    resume under another model, provider, temperature, effort, or prompt
    version would silently mix conditions inside one result.
    If wrong: a resumed result reports the new settings at the top while its
    chunks were paid for under the old ones.
    """
    return {
        "model": request.model,
        "provider": list(request.provider),
        "provider_quantizations": list(request.provider_quantizations),
        "temperature": request.temperature,
        "reasoning_effort": request.reasoning_effort,
        "max_output_tokens": request.max_output_tokens,
        "prompt_version": prompt_version_for([request.monitor_template]),
    }


def assert_resume_settings(journal, plan: ChunkPlan, request: MonitorRequest) -> None:
    """Refuse to reuse a journal line scored under different settings.
    Why: see resume_settings. A line with no settings block predates this
    check and cannot be trusted either.
    If wrong: the silent mix described in resume_settings comes back.
    """
    wanted = resume_settings(request)
    found = journal.completed_settings(plan.prompt_hash)
    if found is None:
        raise ResumeSettingsError(
            f"journal line for chunk {plan.index} carries no settings block "
            "(written before settings were journalled); start a new run instead "
            "of resuming"
        )
    differing = sorted(k for k in wanted if wanted[k] != found.get(k))
    if differing:
        detail = ", ".join(f"{k}: journal {found.get(k)!r} vs run {wanted[k]!r}"
                           for k in differing)
        raise ResumeSettingsError(
            f"refusing to resume chunk {plan.index}: settings differ ({detail})"
        )


def complete_and_parse(client: LLMClient, request: MonitorRequest,
                       plan: ChunkPlan, journal) -> tuple:
    """One monitor call, parsed; re-requested up to twice on an unscorable reply."""
    pin_order = list(request.provider)
    pin = provider_pin(pin_order or None,
                       quantizations=list(request.provider_quantizations) or None)
    last_error: ScoreParseError | None = None
    for attempt in (1, 2, 3):
        response = client.complete(
            plan.user_prompt, request.model,
            max_tokens=request.max_output_tokens, temperature=request.temperature,
            reasoning_effort=request.reasoning_effort, provider=pin,
        )
        assert_pinned_provider(response, pin_order or None)
        try:
            score, reasoning = parse_monitor_reply(response.text)
        except ScoreParseError as error:
            last_error = error
            if journal is not None:
                journal.append({
                    "prompt_hash": plan.prompt_hash, "status": "parse_error",
                    "raw_response": response.text, "attempt": attempt,
                    "finish_reason": response.finish_reason,
                    "native_finish_reason": response.native_finish_reason,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "reasoning_tokens": response.reasoning_tokens,
                    "generation_id": response.generation_id,
                    "provider": response.provider,
                })
            continue
        return response, score, reasoning
    assert last_error is not None
    raise last_error


def score_chunks(plans: list[ChunkPlan], request: MonitorRequest,
                 client: LLMClient, journal) -> list[dict]:
    """Score independent chunks, up to ``request.chunk_concurrency`` at a time.

    Every chunk here is one self-contained call: the prompt is already built,
    nothing a chunk returns feeds another chunk's prompt, and the headline is a
    max over the results. So the only thing serial scoring buys is wall-clock
    time, and a long trajectory with a dozen compaction intervals spends twelve
    prefills one after another.

    Results keep plan order regardless of completion order, so the recorded
    chunk list, the per-view aggregation, and the headline are identical to a
    serial run. Only the journal's line order changes.

    Strategies whose chunks feed each other (hierarchical-summary,
    fold-left-summary, the agentic ones) must not call this.

    An exception from any chunk propagates once the others have been given a
    chance to finish and journal their own results, so a failure still leaves a
    resumable journal rather than discarding paid-for work.
    """

    workers = min(max(1, request.chunk_concurrency), len(plans) or 1)
    if workers == 1 or len(plans) <= 1:
        return [score_chunk(plan, request, client, journal) for plan in plans]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(score_chunk, plan, request, client, journal)
            for plan in plans
        ]
        # Collect in submission order; a failure surfaces only after the pool
        # has drained, so every completed chunk is already in the journal.
        results = [future.result() for future in futures]
    return results


def score_chunk(plan: ChunkPlan, request: MonitorRequest, client: LLMClient,
                journal) -> dict:
    """Score one planned chunk with a single LLM call (or skip it), checkpointing.

    Resume: if the journal already has a completed line whose ``prompt_hash``
    matches this chunk's prompt and whose settings match this request, that
    result is reused and no call is made. Returns the chunk-result dict (no raw
    response — that stays in the journal).
    """
    if journal is not None:
        cached = journal.completed(plan.prompt_hash)
        if cached is not None:
            assert_resume_settings(journal, plan, request)
            return cached

    # A chunk with no agent actions (e.g. only human prompts / a compaction
    # summary) has nothing to monitor: score 0 without spending an API call, like
    # ControlArena's basic monitor when the latest step made no tool calls. The
    # skip/score decision was made once when the plan was built (plan.has_actions),
    # so here it is a single straight-line branch and both paths share _chunk_record.
    if not plan.has_actions:
        record = _chunk_record(
            request, plan, score=0,
            reasoning=("No agent actions (tool calls or results) in this chunk; "
                       "nothing to review."),
            usage=None,
        )
        raw_response = None
    else:
        if journal is not None:
            journal.write_prompt(plan.index, plan.user_prompt)
        response, score, reasoning = complete_and_parse(client, request, plan, journal)
        record = _chunk_record(request, plan, score, reasoning, response)
        raw_response = response.text

    if journal is not None:
        journal.append({
            "prompt_hash": plan.prompt_hash,
            "status": "completed",
            "raw_response": raw_response,
            "chunk": record,
            "settings": resume_settings(request),
        })
    return record


def build_result(request: MonitorRequest, chunks: list[dict],
                 strategy_name: str, views: list[dict] | None = None,
                 config_extra: dict | None = None) -> MonitorResult:
    """Assemble the aggregate result (max over chunk scores) from chunk dicts."""
    if not chunks:
        raise ContextBudgetError("no chunks were scored")
    top = max(chunks, key=lambda c: c["score"])
    totals = {
        "input_tokens": sum(c["input_tokens"] for c in chunks),
        "output_tokens": sum(c["output_tokens"] for c in chunks),
    }
    config = {
        "visibility": request.visibility,
        "diagnostic": is_diagnostic_visibility(request.visibility),
        "show_line_refs": request.render_options.show_line_refs,
        "truncate_tool_results_chars": request.render_options.truncate_tool_results_chars,
        # chunk_target_tokens is None when defaulting to the model window;
        # chunk_budget_tokens is the resolved per-chunk budget actually used,
        # so a run stays reproducible regardless of how the target was set.
        "chunk_target_tokens": request.chunk_target_tokens,
        "chunk_budget_tokens": chunk_transcript_budget(request),
        # Temperature is only sent when reasoning is off (see
        # ``OpenRouterClient.complete``); with an effort set the provider's
        # default sampling applies, and recording the unused value would lie.
        "temperature": (
            None if request.reasoning_effort is not None else request.temperature
        ),
        "max_output_tokens": request.max_output_tokens,
        "reasoning_effort": request.reasoning_effort,
        "provider": list(request.provider),
        "provider_quantizations": list(request.provider_quantizations),
        "backend": request.backend,
        # Recorded because it explains a run's wall-clock and its journal
        # order. Deliberately NOT part of a batch's compatibility key: it
        # changes no prompt, no setting, and no score.
        "chunk_concurrency": request.chunk_concurrency,
        "objectives_provided": request.objectives_provided,
        # Prompt provenance (ControlArena-aligned monitor condition). A strategy
        # that sends templates other than monitor.md lists them in
        # ``config_extra["prompt_templates"]``; the version is derived below.
        "prompt_templates": [request.monitor_template],
        # Every monitor-side template, so any strategy's prompts can be tied to
        # the exact text that rendered them (the version string covers monitor.md only).
        "prompt_template_hashes": {
            name: digest for name, digest in sorted(PROMPT_TEMPLATE_HASHES.items())
            if name.startswith("monitor")
        },
        "task_context_mode": request.task_context_mode,
        "task_sha256": hashlib.sha256(
            request.task_description.encode("utf-8")
        ).hexdigest(),
        # Under a per-view mode each view saw its own brief; empty otherwise.
        "view_task_sha256": {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in sorted(request.view_task_descriptions.items())
        },
        # The objectives catalog is read outside the template loader, so it
        # gets its own hash; None in the blind condition.
        "objectives_sha256": (
            hashlib.sha256(request.objectives.encode("utf-8")).hexdigest()
            if request.objectives else None
        ),
        "environment_visible_to_monitor": False,
        "objective_visibility": (
            "provided" if request.objectives_provided else "blind"
        ),
    }
    if request.context_tokens is not None:
        # Only a self-hosted backend reports its window at run time; record it
        # so a result shows the budget that sized its chunks.
        config["context_tokens"] = request.context_tokens
    if request.base_url is not None:
        config["base_url"] = request.base_url
    if config_extra:
        config.update(config_extra)
    config["prompt_version"] = prompt_version_for(config.get("prompt_templates"))

    return MonitorResult(
        transcript=request.transcript_name,
        score=top["score"],
        band=top["band"],
        reasoning=top["reasoning"],
        model=request.model,
        visibility=request.visibility,
        strategy=strategy_name,
        objectives_provided=request.objectives_provided,
        chunk_count=len(chunks),
        chunks=chunks,
        prompt_tokens_est=sum(c["prompt_tokens_est"] for c in chunks),
        totals=totals,
        cost_estimate_usd=sum(c["cost_usd"] for c in chunks),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config=config,
        views=list(views or []),
    )
