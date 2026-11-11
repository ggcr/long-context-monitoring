"""CLI: score archived trajectories for suspicious behaviour.

    python -m src.monitor <transcript> [options]

``<transcript>`` resolves in four forms, tried in order:

  - a generation/task directory with arm subdirs → every arm beneath it;
  - an archive path or bare .jsonl → that single trajectory;
  - legacy ``<task>/<condition>`` and bare ``<task>`` names remain supported.

The CLI always prints a pre-run summary (event histogram, chunk count, per-chunk
+ total estimated tokens with prompt overhead counted, estimated cost) so a run
is legible before it costs anything. When several conditions are scored the whole
sweep is estimated and cost-gated as a unit — attack and honest run together or
not at all. ``--dry-run`` prints every chunk's full rendered prompt and exits 0
with no network, key, or files written. The cost gate aborts (exit 2) when the
estimate exceeds ``--max-cost-usd`` without ``--force`` — mechanizing the
AGENTS.md estimate-before-sweeps rule.
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from importlib.resources import files

from long_context_monitoring import console
from long_context_monitoring.env import load_dotenv
from long_context_monitoring.monitor import events as events_mod
from long_context_monitoring.monitor.llm import (DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, LLMError,
                             OPENROUTER_BACKEND, OpenRouterClient,
                             default_provider_order,
                             default_provider_quantizations, estimate_cost_usd)
from long_context_monitoring.monitor.openai_direct import (OPENAI_DIRECT_BACKEND,
                                       OpenAIDirectMonitorClient,
                                       direct_model_name)
from long_context_monitoring.monitor.vllm import (VLLM_BACKEND, VLLMClient,
                                                  resolve_base_url, served_model_info)
from long_context_monitoring.monitor.recording import (Journal, assert_batch_configuration_compatible,
                                   assert_batch_visibility_compatible,
                                   make_run_dir, update_batch_manifest,
                                   write_records)
from long_context_monitoring.monitor.render import RenderOptions, count_by_kind
from long_context_monitoring.monitor.scoring import ScoreParseError
from long_context_monitoring.monitor import task_context
from long_context_monitoring.prompts import (DEFAULT_MONITOR_TEMPLATE, MONITOR_TEMPLATE_VERSIONS,
                         prompt_version_for, strategy_prompt_templates)
from long_context_monitoring.monitor.strategies import DEFAULT_STRATEGY, get_strategy
from long_context_monitoring.monitor.strategies.base import (ChunkPlan, ContextBudgetError,
                                         MonitorRequest, MonitorStrategy,
                                         ResumeSettingsError)
from long_context_monitoring.monitor.visibility import DEFAULT_VISIBILITY, VISIBILITY_MODES, apply_visibility
from long_context_monitoring.run_layout import (DEFAULT_RUNS_ROOT, generation_result_subpath,
                            monitor_batch_root, monitor_results_root, tree_sha256,
                            resolve_repo_path,
                            validate_monitor_batch_name)
from long_context_monitoring.runlog import RunLog
from long_context_monitoring.textio import force_utf8_stdio
from long_context_monitoring.transcript.trajectory import Trajectory

TRANSCRIPT_ROOT = DEFAULT_RUNS_ROOT

DEFAULT_OBJECTIVES_FILE = str(files("prompts").joinpath("monitor_catalog.md"))
# Terminal per-request deadline, a floor that the client scales up with the
# request size. Chosen above the socket timeout of the largest chunk this pool
# produces (~1020 s) times its 1.5 margin, so only a stuck call trips it.
MONITOR_REQUEST_DEADLINE_SECONDS = 900.0


def _conditions_under(task_dir: Path) -> list[Path]:
    """Condition subdirs of ``task_dir`` whose ``transcript/`` holds a .jsonl."""
    conditions = []
    for condition in sorted(task_dir.iterdir()):
        if condition.is_dir() and any((condition / "transcript").glob("*.jsonl")):
            conditions.append(condition)
    return conditions


def _available_names(root: Path) -> list[str]:
    """Every trajectory path under ``root`` that holds a main JSONL."""
    names = []
    if root.is_dir():
        for transcript in sorted(root.rglob("transcript")):
            if not transcript.is_dir() or not any(transcript.glob("*.jsonl")):
                continue
            names.append(transcript.parent.relative_to(root).as_posix())
    return names


def resolve_targets(arg: str) -> list[Trajectory]:
    """Resolve an explicit transcript, arm, or matched-pair directory."""
    candidate = resolve_repo_path(arg)
    if not candidate.exists():
        raise SystemExit(f"transcript path does not exist: {candidate}")
    if candidate.is_dir():
        conditions = _conditions_under(candidate)
        if conditions:
            return [Trajectory(condition / "transcript") for condition in conditions]
        if (candidate / "transcript").is_dir():
            candidate = candidate / "transcript"
    return [Trajectory(candidate)]


def _task_context(args, traj: Trajectory, visible: list) -> tuple[str, str, dict]:
    """Choose the brief mode and return ``(mode, full brief, per-view briefs)``.
    Why: each view must receive the host requests that define its task.
    If wrong: scores for comparable views use different task descriptions.
    """
    choice = getattr(args, "task_context", None)
    explicit_spec = getattr(args, "spec", None)
    strategy_key = getattr(args, "strategy", None)
    per_interval_strategy = strategy_key in task_context.INTERVAL_TURN_STRATEGIES

    spec_path = Path(explicit_spec) if explicit_spec else None
    if spec_path is None and choice != "full" and per_interval_strategy:
        spec_path = task_context.find_owning_spec(traj.path)

    # The constant full brief: asked for by name, or no spec owns this
    # transcript (a donated pool session or an external trajectory).
    if choice == "full" or (spec_path is None and choice is None):
        if explicit_spec:
            raise SystemExit("--spec cannot be combined with --task-context full")
        return (task_context.FULL_TASK_CONTEXT_MODE,
                events_mod.extract_task_description(traj), {})
    if spec_path is None:
        raise SystemExit(
            "--task-context interval-turns needs --spec or a transcript under "
            "<spec-dir>/generations/"
        )
    if not per_interval_strategy:
        raise SystemExit(
            f"per-interval briefs are defined only for strategies "
            f"{sorted(task_context.INTERVAL_TURN_STRATEGIES)}, not {strategy_key!r}"
        )
    try:
        context = task_context.load_host_turn_context(spec_path)
        boundaries = [
            event for event in visible
            if event.kind == "compaction" and event.file_id == "main"
            and event.depth == 0
        ]
        briefs = task_context.build_view_briefs(
            context, len(boundaries) + 1, include_pairs=strategy_key == "pairs"
        )
    except task_context.TaskContextError as error:
        raise SystemExit(str(error)) from error
    return task_context.INTERVAL_TURNS_CONTEXT_MODE, briefs["whole"], briefs


# The strategies whose every scored prompt is rendered by
# ``strategies.base.build_chunk_prompt`` and so honour ``--monitor-template``.
# Every other strategy builds at least one prompt its own way; rather than
# let a non-default template be silently ignored there, the CLI refuses it.
TEMPLATE_AWARE_STRATEGIES = frozenset({
    "whole", "summaries-only", "cumulative", "sliding-window", "intervals",
    "pairs", "three-view", "final-ablation",
    # These four build their own prompts but select a variant per scoring
    # template (``prompts.STRATEGY_TEMPLATE_VARIANTS``); hierarchical and
    # fold-left send the scoring template itself for their root score.
    "annotated-history", "hierarchical-summary", "fold-left-summary",
    "sandbox", "scratchpad",
})


def _resolved_effort(effort: str | None) -> str | None:
    """Turn the CLI's ``none`` sentinel into the no-effort value the clients want.

    ``--reasoning-effort`` has always documented ``none`` as "send no effort",
    but the string was passed through verbatim, so the request carried
    ``reasoning: {"effort": "none"}`` instead of omitting the field. Nothing
    recorded has used it, so normalizing here changes no existing result. It
    matters for non-reasoning monitors such as gpt-4.1-nano, which have no
    effort setting to send.
    """

    return None if effort == "none" else effort


def _build_request(args, traj: Trajectory) -> tuple[MonitorRequest, list, list]:
    all_events = events_mod.extract_events(traj)
    visible = apply_visibility(all_events, args.visibility)
    objectives = None
    if not args.blind and args.objectives_file:
        objectives = Path(args.objectives_file).read_text(encoding="utf-8")
    manifest = traj.manifest() or {}
    source_path = traj.path.resolve()
    context_mode, full_brief, view_briefs = _task_context(args, traj, visible)
    request = MonitorRequest(
        events=visible,
        transcript_name=events_mod.transcript_name(traj),
        task_description=full_brief,
        view_task_descriptions=view_briefs,
        task_context_mode=context_mode,
        visibility=args.visibility,
        render_options=RenderOptions(
            show_line_refs=True,
            truncate_tool_results_chars=args.truncate_tool_results_chars,
        ),
        model=args.model,
        chunk_target_tokens=args.chunk_target_tokens,
        temperature=args.temperature,
        objectives=objectives,
        only_view=getattr(args, "only_view", None),
        ensemble_models=tuple(
            model.strip()
            for model in (getattr(args, "ensemble_models", None) or "").split(",")
            if model.strip()
        ),
        ensemble_base_strategy=getattr(args, "ensemble_base_strategy", None) or "intervals",
        internal_repetitions=getattr(args, "internal_reps", 1) or 1,
        reasoning_effort=_resolved_effort(getattr(args, "reasoning_effort", None)),
        provider=default_provider_order(
            args.model, getattr(args, "provider", None)
        ),
        provider_quantizations=default_provider_quantizations(
            args.model, getattr(args, "quantization", None)
        ),
        backend=getattr(args, "backend", OPENROUTER_BACKEND),
        context_tokens=getattr(args, "context_tokens", None),
        base_url=getattr(args, "base_url", None),
        chunk_concurrency=max(1, getattr(args, "chunk_concurrency", 1) or 1),
        monitor_template=getattr(args, "monitor_template", None) or DEFAULT_MONITOR_TEMPLATE,
        run_metadata={
            "compaction_mode": manifest.get("compaction_mode", "unknown"),
            "runtime_guidance_turns": manifest.get(
                "runtime_guidance_turns", []
            ),
            "source_transcript_path": str(source_path),
            "source_transcript_sha256": tree_sha256(source_path),
            "result_subpath": generation_result_subpath(source_path),
            "generation_id": manifest.get("generation_id"),
            "empty_compaction_summaries": (
                events_mod.count_empty_compaction_summaries(all_events)
            ),
        },
    )
    return request, all_events, visible


def _print_summary(request: MonitorRequest, all_events: list, visible: list,
                   plans: list[ChunkPlan], strategy_key: str,
                   strategy=None) -> tuple[int, float]:
    """Print the pre-run summary; return (total_prompt_tokens_est, est_cost).

    A strategy that makes calls its plan does not list (map-reduce's reduce,
    the summary trees' merges, the agentic tool turns) exposes
    ``expected_calls(plans)``; the estimate is scaled by expected over
    planned so the cost gate sees every call, and both counts are printed.

    A chunk with no agent actions (only human prompts / a compaction summary) is
    scored 0 by the strategy without an API call, so it is tagged as skipped here
    and its tokens are left out of the estimate — the cost shown matches what a
    run actually spends, and the cost gate is not tripped by prompts never sent.
    """
    total = len(plans)
    per_chunk_tokens = [plan.prompt_tokens_est for plan in plans]
    per_chunk_scored = [plan.has_actions for plan in plans]
    # Only chunks that will actually be sent contribute to the estimate.
    scored_plans = [plan for plan, scored in zip(plans, per_chunk_scored, strict=True) if scored]
    total_input_est = sum(plan.prompt_tokens_est for plan in scored_plans)
    scored_chunks = len(scored_plans)
    output_est = request.max_output_tokens * scored_chunks
    # plan.model is set only by ensemble members, so a multi-model sweep prices
    # each chunk at its own model's rate rather than the nominal request model's
    # (estimate_cost_usd is linear in tokens, so this equals the old bulk
    # calculation whenever every plan shares one model).
    # A self-hosted vLLM server bills nothing, so its estimate is exactly 0
    # rather than the table's conservative default for an unlisted model.
    est_cost = 0.0 if request.backend == VLLM_BACKEND else sum(
        estimate_cost_usd(plan.model or request.model, plan.prompt_tokens_est,
                          request.max_output_tokens)
        for plan in scored_plans
    )
    expected_calls = scored_chunks
    calls_note = ""
    if strategy is not None and hasattr(strategy, "expected_calls") and scored_chunks:
        expected_calls, calls_note = strategy.expected_calls(scored_plans)
        # Unplanned calls are priced at the mean planned prompt; for the
        # agentic loops this is a turn bound, and a long investigation costs more.
        est_cost = est_cost * expected_calls / scored_chunks

    plan_models = sorted({plan.model for plan in plans if plan.model})
    model_label = " + ".join(plan_models) if plan_models else request.model

    print(console.kv("Transcript", request.transcript_name, width=11))
    print(console.kv("Model", model_label, width=11))
    # Shown before any spend so a run in the wrong brief mode is visible.
    print(console.kv("Brief", request.task_context_mode, width=11))
    print(console.kv(
        "Strategy",
        f"{strategy_key}  ·  visibility {request.visibility}"
        f"  ·  objectives_provided {request.objectives_provided}",
        width=11,
    ))
    print(console.kv(
        "Events",
        f"{len(all_events)} total, {len(visible)} visible ({request.visibility})",
        width=11,
    ))
    visible_hist = count_by_kind(visible)
    for kind in sorted(visible_hist):
        print(f"             {console.dim(f'{visible_hist[kind]:>5}')}  {kind}")
    print(console.kv(
        "Chunks",
        f"{total} ({scored_chunks} scored via the API, "
        f"{total - scored_chunks} skipped as no-action)",
        width=11,
    ))
    for plan, tokens, scored in zip(plans, per_chunk_tokens, per_chunk_scored, strict=True):
        scope = ""
        if plan.view_total > 1:
            scope = (f" [{plan.view_label}, "
                     f"{plan.view_chunk_index}/{plan.view_chunk_total}]")
        if scored:
            print(f"             chunk {plan.index}{scope}: ~{tokens} prompt "
                  "tokens (overhead included)")
        else:
            print(console.dim(
                f"             chunk {plan.index}{scope}: skipped "
                "(no agent actions) — scored 0 without an API call"))
    print(console.kv(
        "Est. tokens",
        f"~{total_input_est} input + ~{output_est} output reserve",
        width=11,
    ))
    calls_line = f"{scored_chunks} planned, {expected_calls} expected"
    if calls_note:
        calls_line += f" ({calls_note})"
    print(console.kv("Calls", calls_line, width=11))
    print(console.kv(
        "Est. cost",
        f"~${est_cost:.4f} (rough; actual usage recorded per chunk)",
        width=11,
    ))
    return total_input_est, est_cost


def _print_dry_run(plans: list[ChunkPlan]) -> None:
    """Print every chunk's full rendered prompt (no network, no files)."""
    print()
    print(console.header("DRY RUN: rendered chunk prompts (no network)"))
    total = len(plans)
    for plan in plans:
        skip = ("" if plan.has_actions
                else console.dim("  [skipped: no agent actions — scored 0, not sent]"))
        scope = ""
        if plan.view_total > 1:
            scope = f" — {plan.view_label} view"
        print(f"\n{console.rule()}")
        print(f"chunk {plan.index}/{total}{scope} "
              f"{console.dim(f'(prompt_hash {plan.prompt_hash[:12]})')}{skip}")
        print(console.rule())
        print(plan.user_prompt)


def _make_client(backend: str, base_url: str | None = None):
    """The LLM client for ``backend``, under the monitor's request deadline.

    Every client honours the same ``LLMClient.complete`` contract, so every
    strategy is unchanged by the choice.
    """

    if backend == VLLM_BACKEND:
        return VLLMClient(base_url, hard_timeout=MONITOR_REQUEST_DEADLINE_SECONDS)
    if backend == OPENAI_DIRECT_BACKEND:
        return OpenAIDirectMonitorClient(
            hard_timeout=MONITOR_REQUEST_DEADLINE_SECONDS
        )
    return OpenRouterClient(hard_timeout=MONITOR_REQUEST_DEADLINE_SECONDS)


def _score_target(args, request: MonitorRequest, strategy,
                  plans: list[ChunkPlan], now: datetime) -> int:
    """Run one already-planned trajectory through the strategy and write results.

    Returns a process-style exit code: 0 on success, 1 if the strategy errored
    (progress is checkpointed for --resume).
    """
    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            raise SystemExit(f"--resume dir does not exist: {run_dir}")
        print(f"\nResuming run in {run_dir}")
    else:
        run_id = uuid.uuid4().hex[:6]
        transcript_subpath = (
            request.run_metadata.get("result_subpath")
            or request.transcript_name
        )
        if args.repetition:
            transcript_subpath = f"{transcript_subpath}/{args.repetition}"
        run_dir = make_run_dir(
            transcript_subpath, run_id, now, Path(args.recording_root)
        )

    journal = Journal(run_dir / "journal.jsonl")
    client = _make_client(request.backend, request.base_url)
    try:
        result = strategy.run(request, client, journal, plans=plans)
    except (ContextBudgetError, LLMError, ScoreParseError, ResumeSettingsError) as error:
        print(file=sys.stderr)
        console.error(str(error), command="monitor")
        print(console.dim(
            f"Progress is checkpointed in {run_dir / 'journal.jsonl'}; "
            f"resume with --resume {run_dir}"), file=sys.stderr)
        return 1

    write_records(result, run_dir)
    if args.batch_root is not None:
        source = {
            "transcript": request.transcript_name,
            "path": request.run_metadata.get("source_transcript_path"),
            "sha256": request.run_metadata.get("source_transcript_sha256"),
            "generation_id": request.run_metadata.get("generation_id"),
        }
        update_batch_manifest(args.batch_root, result, run_dir, source)
    _print_result(result)
    print(f"\n{console.ok(f'written to {run_dir}')}")
    return 0


def _colour_score(score: int, text: str) -> str:
    """Colour a score/band string by severity: high=red, mid=yellow, low=green."""
    if score >= 7:
        return console.red(text)
    if score >= 4:
        return console.yellow(text)
    return console.green(text)


def _print_result(result) -> None:
    """Print a run's outcome: the overall score, then a per-view breakdown.

    A strategy that reviews the trajectory from several angles records one
    aggregate per view in ``result.views``. The configured headline view can
    hide that, say, the
    pre-compaction view fired at 9 while the post-compaction view stayed at 2 —
    so when more than one view is present we print each view's own score and
    reasoning rather than only the top one.
    """
    score_text = _colour_score(result.score, f"{result.score}/10 ({result.band})")
    print(f"\n{console.bold('SCORE')}  {score_text}  "
          f"{console.dim(f'[{result.strategy}, {result.chunk_count} chunk(s), {result.visibility}]')}")
    print(console.kv(
        "COST",
        f"~${result.cost_estimate_usd:.4f} "
        f"(input {result.totals.get('input_tokens', 0)} tok, "
        f"output {result.totals.get('output_tokens', 0)} tok)",
        width=5,
    ))

    if len(result.views) > 1:
        label_width = max(len(view["label"]) for view in result.views)
        headline = result.config.get("headline_view")
        if headline:
            overall_text = f"(headline = {headline})"
        else:
            overall_text = f"(overall = max over {len(result.views)} views)"
        overall = console.dim(overall_text)
        print(f"\n{console.bold('PER-VIEW SCORES')} {overall}:")
        for view in result.views:
            score = _colour_score(view["score"], f"{view['score']}/10 ({view['band']})")
            print(f"  {view['label']:<{label_width}} : {score}")
        for view in result.views:
            heading = f"REASONING [{view['label']}, {view['score']}/10]"
            print(f"\n{console.bold(heading)}:")
            print(view["reasoning"])
    else:
        print(f"\n{console.bold('REASONING')}:")
        print(result.reasoning)


def _plan_targets(
    args, targets: list[Trajectory], strategy_key: str,
    strategy: MonitorStrategy,
) -> tuple[list[tuple[MonitorRequest, list[ChunkPlan]]], float]:
    """Plan (but do not score) every target, printing each one's summary.

    Plan every target before making network calls so the cost limit applies
    to the combined estimated total.
    Returns the per-target ``(request, chunk_plans)`` pairs and the combined
    estimated cost. Every target gets a ``[i/N]`` header; ``[1/1]`` for a lone
    target is fine and keeps the output uniform across single- and multi-runs.
    """
    plans: list[tuple[MonitorRequest, list[ChunkPlan]]] = []
    total_est = 0.0
    template = getattr(args, "monitor_template", None) or DEFAULT_MONITOR_TEMPLATE
    if template != DEFAULT_MONITOR_TEMPLATE and strategy_key not in TEMPLATE_AWARE_STRATEGIES:
        raise SystemExit(
            f"--monitor-template {template} is not honoured by strategy "
            f"{strategy_key!r}; it would be silently ignored. Use one of: "
            + ", ".join(sorted(TEMPLATE_AWARE_STRATEGIES))
        )

    total = len(targets)
    skipped: list[str] = []
    for index, traj in enumerate(targets, start=1):
        name = events_mod.transcript_name(traj)
        print(console.rule())
        print(f"[{index}/{total}] {console.bold(name)}")
        print(console.rule())
        request, all_events, visible = _build_request(args, traj)
        try:
            chunk_plans = strategy.plan(request)
        except ContextBudgetError as error:
            # A strategy that does not apply to this transcript (no compaction
            # to ablate, no summary to score) must not abort the other targets
            # of a pool sweep. The skip is printed and counted, not scored.
            skipped.append(name)
            print(console.dim(f"SKIPPED (strategy not applicable): {error}"))
            print()
            continue
        _, est_cost = _print_summary(request, all_events, visible,
                                     chunk_plans, strategy_key, strategy)
        plans.append((request, chunk_plans))
        total_est += est_cost
        print()
    if skipped:
        print(f"{len(skipped)} of {total} target(s) skipped as not applicable "
              f"to {strategy_key}: {', '.join(skipped)}\n")
    if not plans:
        raise ContextBudgetError(
            f"{strategy_key} applies to none of the {total} target(s)"
        )
    return plans, total_est


def _add_prompt_arguments(parser: argparse.ArgumentParser) -> None:
    """The flags that choose what instruction text the monitor is sent."""
    parser.add_argument("--objectives-file", default=DEFAULT_OBJECTIVES_FILE,
                        help="security-property catalog shown to the monitor "
                        "(the informed condition, stamped objectives_provided="
                        "true); defaults to prompts/monitor_catalog.md. Pass "
                        "--blind to withhold it. Score both arms the same way")
    parser.add_argument("--blind", action="store_true",
                        help="run the blind condition: withhold the objectives "
                        "catalog so the monitor hunts open-ended (stamped "
                        "objectives_provided=false). Overrides --objectives-file")
    parser.add_argument(
        "--monitor-template", default=DEFAULT_MONITOR_TEMPLATE,
        choices=sorted(MONITOR_TEMPLATE_VERSIONS),
        help="scoring template under prompts/ that renders each chunk prompt. "
             "The default is monitor.md for every model; monitor_nano.md is "
             "the variant being fitted to gpt-4.1-nano. Only strategies that "
             "render through the shared chunk prompt accept a non-default "
             "value. The template's version and content hash land in "
             "prompt_version, so results never pool across templates")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lcm monitor",
        description="Score an archived Claude Code trajectory for suspicious behaviour.",
    )
    parser.add_argument(
        "transcript",
        help=(
            "generation directory (scores every arm); arm/archive/.jsonl path "
            "(one trajectory)"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--backend",
        default=OPENROUTER_BACKEND,
        choices=[OPENROUTER_BACKEND, OPENAI_DIRECT_BACKEND, VLLM_BACKEND],
        help="transport for monitor calls (default %(default)s). "
             f"'{OPENAI_DIRECT_BACKEND}' sends an openai/* model straight to "
             "api.openai.com, skipping OpenRouter's routing hop; it needs "
             "OPENAI_API_KEY and reports an estimated, not a billed, cost. "
             f"'{VLLM_BACKEND}' sends --model to a self-hosted vLLM server at "
             "--base-url, sizes chunks from its /v1/models max_model_len, and "
             "costs $0",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"OpenAI-compatible /v1 root for --backend {VLLM_BACKEND} "
             "(default: $BASE_URL, e.g. http://localhost:8000/v1). "
             "Optional key: $VLLM_API_KEY",
    )
    parser.add_argument(
        "--chunk-concurrency",
        type=int,
        default=1,
        help="how many of a strategy's independent chunks to score at once "
             "(default %(default)s, i.e. serial). Chunk prompts are built "
             "before any call and the headline is a max over chunks, so this "
             "changes wall-clock time, not scores. Ignored by strategies whose "
             "chunks feed each other (hierarchical-summary, fold-left-summary, "
             "sandbox, scratchpad)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        help="OpenRouter reasoning.effort for each monitor call "
             f"(default {DEFAULT_REASONING_EFFORT}; pass 'none' to send no effort)",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help="pin OpenRouter provider.order (repeatable); "
             "allow_fallbacks=false when set (e.g. --provider Anthropic)",
    )
    parser.add_argument(
        "--quantization",
        action="append",
        default=None,
        help="pin OpenRouter provider.quantizations (repeatable); "
             "independent of --provider (e.g. --quantization fp8)",
    )
    parser.add_argument("--visibility", default=DEFAULT_VISIBILITY,
                        choices=VISIBILITY_MODES)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument(
        "--only-view",
        default=None,
        help="score only one strategy-defined view (for example interval_2, "
             "pre_compaction, post_compaction, or final_combined; the whole "
             "trajectory is the whole strategy, not a view of intervals)",
    )
    parser.add_argument(
        "--internal-reps", type=int, default=1,
        help="how many times the agentic strategies (sandbox, scratchpad) "
             "repeat their investigation within ONE invocation, aggregated by "
             "max (default 1). Left at 1, a repetition means the same thing "
             "for every strategy and reps come from separate --repetition "
             "runs; ignored by the other strategies",
    )
    parser.add_argument(
        "--chunk-target-tokens", type=int, default=None,
        help="per-chunk token target; defaults to the monitor model's full "
             "context window (chunk only as a transcript nears it). Set a lower "
             "value to request smaller chunks.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--truncate-tool-results-chars", type=int, default=None,
                        help="opt-in cap on tool-result length (changes the "
                        "monitor condition; recorded in the result)")
    parser.add_argument(
        "--spec", default=None,
        help="spec.yaml whose host_turns and compact_after define each "
             "interval's task brief; defaults to the spec.yaml that owns the "
             "transcript's generations/ folder")
    parser.add_argument(
        "--task-context", default=None, choices=("interval-turns", "full"),
        help="interval-turns: each interval is told only its own host turns "
             "(the default whenever a spec is found, for the intervals and "
             "whole strategies). full: every view is told the one constant "
             "brief. The mode is recorded as task_context_mode; never pool "
             "scores across modes")
    _add_prompt_arguments(parser)
    parser.add_argument("--dry-run", action="store_true",
                        help="print rendered chunk prompts and exit; no network")
    parser.add_argument("--resume", default=None,
                        help="an existing run dir to resume (reuses completed "
                        "journal chunks by prompt hash)")
    parser.add_argument("--max-cost-usd", type=float, default=5.0)
    parser.add_argument("--force", action="store_true",
                        help="run even if the cost estimate exceeds --max-cost-usd")
    parser.add_argument("--results-dir", help="directory for newly generated monitor outputs")
    parser.add_argument(
        "--batch-name",
        help=(
            "required for new scoring; use "
            "<scope-and-source>-scored-by-<monitor> beneath "
            "<results-dir>/monitor-runs/<UTC-date>/<name>/"
        ),
    )
    parser.add_argument(
        "--repetition",
        choices=[f"rep-{index:02d}" for index in range(1, 100)],
        help="optional repetition directory inside a dated monitor batch",
    )
    args = parser.parse_args(argv)
    if args.repetition and not args.batch_name:
        parser.error("--repetition requires --batch-name")
    if not args.dry_run and not args.resume and not args.batch_name:
        parser.error(
            "new monitor runs require --batch-name so results use the dated "
            "monitor-runs layout"
        )
    if args.resume:
        if not args.batch_name:
            parser.error("--resume requires --batch-name (the batch the run dir belongs to)")
        # A relative run dir crashed inside the manifest update after the
        # paid calls had completed (recording.py relative_to); resolve it now.
        args.resume = str(Path(args.resume).resolve())
    if args.batch_name:
        try:
            validate_monitor_batch_name(args.batch_name)
        except ValueError as error:
            parser.error(str(error))

    # Console codepage must not decide whether a run survives (see the helper).
    force_utf8_stdio()

    runlog = RunLog("monitor", list(argv if argv is not None else sys.argv[1:]))
    runlog.install()
    try:
        return _run(args, runlog)
    finally:
        for path in runlog.paths:
            print(console.dim(f"log -> {path}"))
        runlog.close()


def batch_root_for_resume(run_dir: Path, batch_name: str) -> Path:
    """The batch directory a resumed run dir belongs to.
    Why: a resume must land in the batch that already holds the journal, not
    a new batch named for today's date.
    If wrong: the resumed result is written under a second batch root and the
    original manifest never lists it.
    """
    for ancestor in run_dir.parents:
        if ancestor.name == batch_name and (ancestor / "results") in (run_dir, *run_dir.parents):
            return ancestor
    raise ValueError(
        f"--resume dir {run_dir} is not inside a batch named {batch_name!r} "
        "(expected <runs>/monitor-runs/<date>/<batch-name>/results/...)"
    )


def _open_run_logs(runlog: RunLog, targets: list[Trajectory], results_dir: str,
                   batch_root: Path | None = None) -> None:
    """Open one console log per run tree this sweep touches.

    A named batch gets one shared log. Other multi-target runs get one log
    per task, each with the complete planning and estimated-cost output.
    """
    if batch_root is not None:
        runlog.open_in(batch_root, "batch")
        return

    by_task: dict[str, list[str]] = {}
    for trajectory in targets:
        task, _, condition = events_mod.transcript_name(trajectory).partition("/")
        conditions = by_task.setdefault(task, [])
        if condition and condition not in conditions:
            conditions.append(condition)
    for task, conditions in by_task.items():
        runlog.open_in(Path(results_dir) / task, "+".join(conditions) or None)


def _resolve_recording_roots(args: argparse.Namespace, now: datetime) -> None:
    """Set ``args.batch_root`` and ``args.recording_root`` for this run."""
    if args.resume:
        # The batch is the one the run dir already sits in, whatever today's
        # date is; recomputing it from today crashed a resume across a UTC day.
        try:
            args.batch_root = batch_root_for_resume(Path(args.resume), args.batch_name)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        args.recording_root = args.batch_root / "results"
    elif args.batch_name:
        try:
            args.batch_root = monitor_batch_root(
                args.results_dir, args.batch_name, now.strftime("%Y-%m-%d")
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        args.recording_root = args.batch_root / "results"
    else:
        args.batch_root = None
        args.recording_root = Path(args.results_dir)


def _assert_batch_compatible(args: argparse.Namespace, request: MonitorRequest,
                             strategy: MonitorStrategy) -> None:
    """Refuse a run whose visibility or configuration differs from its batch."""
    configuration = {
        "model": request.model,
        "strategy": strategy.name,
        "visibility": request.visibility,
        "objectives_provided": request.objectives_provided,
        # The same derived version build_result records, so a strategy that
        # sends extra templates is not refused by its own batch.
        "prompt_version": prompt_version_for(strategy_prompt_templates(
            getattr(strategy, "prompt_templates", ()), request.monitor_template)),
        "reasoning_effort": request.reasoning_effort,
        "provider": list(request.provider),
        "provider_quantizations": list(request.provider_quantizations),
        "backend": request.backend,
    }
    try:
        assert_batch_visibility_compatible(args.batch_root, args.visibility)
        assert_batch_configuration_compatible(args.batch_root, configuration)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def _run(args: argparse.Namespace, runlog: RunLog) -> int:
    """Plan, gate, and score the sweep, with ``runlog`` capturing the console."""
    # Repo-local .env supplies the backend's API key when the shell has not
    # exported it (OPENROUTER_API_KEY, or OPENAI_API_KEY for --backend
    # openai-direct).
    load_dotenv()

    # A backend/model mismatch is free to catch here and expensive to discover
    # after a sweep has been planned and the first chunk paid for.
    if getattr(args, "backend", OPENROUTER_BACKEND) == OPENAI_DIRECT_BACKEND:
        try:
            direct_model_name(args.model)
        except LLMError as error:
            raise SystemExit(str(error)) from error
        if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                f"--backend {OPENAI_DIRECT_BACKEND} needs OPENAI_API_KEY; set it "
                "in the environment or the repo-local .env"
            )

    # A vLLM server reports its own window; read it before planning so chunk
    # budgets match the --max-model-len it was started with. The query is
    # local and free, so a dry run makes it too (its plan depends on it).
    if getattr(args, "backend", OPENROUTER_BACKEND) == VLLM_BACKEND:
        if getattr(args, "provider", None) or getattr(args, "quantization", None):
            raise SystemExit(
                f"--backend {VLLM_BACKEND} serves one model from one host; "
                "--provider/--quantization are OpenRouter routing pins"
            )
        try:
            args.base_url = resolve_base_url(getattr(args, "base_url", None))
            served = served_model_info(args.base_url, args.model)
        except LLMError as error:
            raise SystemExit(str(error)) from error
        args.context_tokens = served.max_model_len
        print(console.kv(
            "Server", f"{args.base_url}  ·  max_model_len {served.max_model_len}",
            width=11))
    elif getattr(args, "base_url", None):
        raise SystemExit(f"--base-url is only used by --backend {VLLM_BACKEND}")

    if getattr(args, "chunk_concurrency", 1) > 1 and not args.dry_run:
        print(console.yellow(
            f"note: --chunk-concurrency {args.chunk_concurrency} scores chunks "
            f"on worker threads, so the "
            f"{MONITOR_REQUEST_DEADLINE_SECONDS:g}s wall-clock request "
            f"deadline does not apply (POSIX alarms are main-thread only); "
            f"only the socket timeout bounds each call. For pool throughput "
            f"prefer more concurrent invocations."
        ), file=sys.stderr)

    # Validate the strategy name early with a clear error.
    try:
        strategy = get_strategy(args.strategy)
    except KeyError as error:
        raise SystemExit(str(error)) from error
    if args.resume and not getattr(strategy, "supports_resume", True):
        raise SystemExit(
            f"--resume is not supported by --strategy {args.strategy}: it never "
            "reads its journal back, so a resume would pay for and resample every "
            "call. Start a new run instead."
        )
    if args.only_view and not getattr(strategy, "supports_only_view", False):
        raise SystemExit(
            f"--only-view is ignored by --strategy {args.strategy}; only strategies "
            "with named views (intervals) accept it"
        )
    targets = resolve_targets(args.transcript)
    if args.resume and len(targets) > 1:
        names = ", ".join(events_mod.transcript_name(t) for t in targets)
        raise SystemExit(
            f"--resume resumes a single run dir, but {args.transcript!r} expands "
            f"to {len(targets)} conditions ({names}). Pass an explicit "
            f"<task>/<condition> to resume one."
        )

    # Plan the whole sweep first (summaries + combined estimate), then gate, then
    # score — one control eval's arms are planned and cost-gated as a unit.
    try:
        plans, total_est = _plan_targets(args, targets, args.strategy, strategy)
    except ContextBudgetError as error:
        print(file=sys.stderr)
        console.error(str(error), command="monitor")
        return 1

    if args.dry_run:
        # No sink for a dry run: the rendered prompts are the whole transcript,
        # and a real run already persists them as chunk-NN.prompt.txt.
        for _request, chunk_plans in plans:
            _print_dry_run(chunk_plans)
        return 0

    if args.results_dir is None:
        args.results_dir = str(monitor_results_root([target.path for target in targets]))
    now = datetime.now(timezone.utc)
    _resolve_recording_roots(args, now)
    if args.batch_root is not None:
        _assert_batch_compatible(args, plans[0][0], strategy)

    # Opened before the cost gate so the planning summary and an aborted run are
    # both on disk; the output printed so far was buffered and lands here too.
    _open_run_logs(runlog, targets, args.results_dir, args.batch_root)

    # Only a real multi-condition sweep needs a combined line; a single target's
    # own summary already carries its estimate.
    if len(plans) > 1:
        print(console.kv(
            "Combined", f"est. cost across {len(plans)} conditions: ~${total_est:.4f}",
            width=11))
    if total_est > args.max_cost_usd and not args.force:
        print(file=sys.stderr)
        console.abort(
            f"estimated cost ${total_est:.4f} exceeds --max-cost-usd "
            f"${args.max_cost_usd:.2f}. Re-run with --force to proceed.",
            command="monitor")
        return 2

    exit_code = 0
    for index, (request, chunk_plans) in enumerate(plans, start=1):
        print(f"\n{console.rule()}")
        print(console.run(f"[{index}/{len(plans)}] scoring {console.bold(request.transcript_name)}"))
        print(console.rule())
        rc = _score_target(args, request, strategy, chunk_plans, now)
        if rc != 0 and exit_code == 0:
            exit_code = rc
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
