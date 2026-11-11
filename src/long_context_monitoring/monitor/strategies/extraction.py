"""Budgets and plans for the calls that are not one chunk of one view.

Extraction (map-reduce and the summary trees), merge, reduce, and agentic
tool turns build their prompts at run time, so they need the same
window-derived ceiling ``chunk_transcript_budget`` gives a single chunk, and a
planner that shows a dry run the extraction prompts a paid run will send.
Split out of ``base.py`` to keep that module within the readability limit.
"""

from __future__ import annotations

import hashlib

from long_context_monitoring.monitor.render import estimate_tokens, render_units
from long_context_monitoring.monitor.strategies.base import (
    MIN_CHUNK_BUDGET_TOKENS,
    WINDOW_FILL,
    ChunkPlan,
    ContextBudgetError,
    MonitorRequest,
    View,
    _unit_event_id,
    chunk_has_actions,
    split_chunks,
)
from long_context_monitoring.prompts import build_extraction_prompt


def stage_prompt_budget(request: MonitorRequest) -> int:
    """Tokens one whole prompt may hold on any call path, overhead included.
    Why: extraction, merge, reduce, and agentic turns never go through
    chunk_transcript_budget, so they need the same window-derived ceiling.
    If wrong: a stage sends a prompt the model window cannot hold and the
    call fails or is silently truncated by the provider.
    """
    window = request.context_window
    target = window if request.chunk_target_tokens is None else min(
        request.chunk_target_tokens, window)
    return int(target * WINDOW_FILL) - request.max_output_tokens


def assert_prompt_fits(request: MonitorRequest, prompt: str, stage: str) -> None:
    """Refuse to send a prompt that does not fit the stage budget.
    Why: merge and reduce prompts are built from model replies, so their size
    is only known at run time.
    If wrong: see stage_prompt_budget.
    """
    limit = stage_prompt_budget(request)
    tokens = estimate_tokens(prompt)
    if tokens > limit:
        raise ContextBudgetError(
            f"{stage} prompt is ~{tokens} tokens, over the ~{limit}-token budget "
            f"for {request.model}; lower --max-output-tokens or use a larger window"
        )


def assert_digests_fit(request: MonitorRequest, digest_count: int, stage: str) -> None:
    """Refuse before the first paid call when later replies could not share a prompt.
    Why: a merge or reduce prompt is built from ``digest_count`` model replies of
    up to ``max_output_tokens`` each, so its size is known in advance; failing
    at that stage would lose every leaf extraction already paid for.
    If wrong: leaves are paid for and the run fails on the merge or reduce.
    """
    needed = digest_count * request.max_output_tokens
    limit = stage_prompt_budget(request)
    if needed > limit:
        raise ContextBudgetError(
            f"{stage} prompt could hold {digest_count} digests of up to "
            f"{request.max_output_tokens} tokens (~{needed}), over the ~{limit}-token "
            f"budget for {request.model}; lower --max-output-tokens"
        )


def split_extraction_parts(request: MonitorRequest,
                           view: View) -> list[tuple[list, str, str]]:
    """Split one section into extraction prompts that each fit the budget.

    Returns ``[(units, label, prompt), ...]``; one part when the section fits.
    Why: a compaction interval can be larger than the window on its own.
    If wrong: an extraction prompt carries an unsplit section over the window.
    """
    probe_label = f"{view.label} (part 99/99)"
    overhead = estimate_tokens(
        build_extraction_prompt(request.task_description, "", probe_label))
    # The floor keeps a tiny --chunk-target-tokens from producing zero-size
    # chunks; a prompt built at the floor can still exceed a tiny target.
    budget = max(stage_prompt_budget(request) - overhead, MIN_CHUNK_BUDGET_TOKENS)
    chunks = split_chunks(view.events, request.render_options, budget)
    parts: list[tuple[list, str, str]] = []
    for index, units in enumerate(chunks, start=1):
        label = view.label
        if len(chunks) > 1:
            label = f"{view.label} (part {index}/{len(chunks)})"
        text = render_units(units, request.render_options)
        prompt = build_extraction_prompt(request.task_description, text, label)
        parts.append((units, label, prompt))
    return parts


def plan_extraction(request: MonitorRequest, views: list[View]) -> list[ChunkPlan]:
    """One ChunkPlan per extraction call, with the prompt run() will send.

    Shared by map-reduce and the two summary strategies so a dry run shows,
    and hashes, exactly the extraction prompts a paid run sends. The merge,
    reduce, and score calls depend on model replies and are not planned.
    """
    plans: list[ChunkPlan] = []
    index = 0
    for view_index, view in enumerate(views, start=1):
        parts = split_extraction_parts(request, view)
        for part_index, (units, label, prompt) in enumerate(parts, start=1):
            index += 1
            plans.append(ChunkPlan(
                index=index, units=list(units),
                has_actions=chunk_has_actions(units),
                user_prompt=prompt,
                prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                prompt_tokens_est=estimate_tokens(prompt),
                event_start=_unit_event_id(units[0]) if units else None,
                event_end=_unit_event_id(units[-1]) if units else None,
                event_count=len(units),
                view=view.name, view_label=label,
                view_index=view_index, view_total=len(views),
                view_chunk_index=part_index, view_chunk_total=len(parts),
            ))
    return plans
