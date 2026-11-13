"""Action only — score every tool call on its own, with no transcript before it.

The no-context baseline for sliding-window: the same calls, the same rendering
of the scored call (arguments unpacked, multi-line text fenced, result
withheld), the same question and output format, but the prompt holds only the
task, the objectives and the call. Comparing it with sliding-window at several
context lengths asks whether the monitor's view of the trajectory matters for
multi-step attacks, whose steps each look harmless alone.

One call per tool call; prompts are a few thousand tokens, so a run is cheap
and decode-bound. Records and resume work as in sliding-window (one record per
call with its ``action_index``), so the same action-level safety applies.
"""
from __future__ import annotations

import hashlib

from long_context_monitoring.monitor.llm import LLMClient
from long_context_monitoring.monitor.render import _ActionNumbering
from long_context_monitoring.monitor.scoring import MonitorResult
from long_context_monitoring.monitor.strategies.base import ChunkPlan, ContextBudgetError, MonitorRequest
from long_context_monitoring.monitor.strategies.sliding_window import (ActionPlan, SlidingWindowMonitor,
                                                                       _action_text)
from long_context_monitoring.prompts import build_action_only_prompt


class ActionOnlyMonitor(SlidingWindowMonitor):
    name = "action-only-v1"
    prompt_templates = ("monitor_action_only.md",)

    def plan(self, request: MonitorRequest) -> list[ChunkPlan]:
        events = request.events
        numbering = _ActionNumbering()
        plans: list[ChunkPlan] = []
        for position, event in enumerate(events):
            if event.kind not in ("tool_call", "agent_spawn"):
                continue
            number = numbering.next_for(event)
            if event.kind != "tool_call":
                continue
            action = _action_text(event, number)
            user = build_action_only_prompt(request.task_description, action,
                                            objectives=request.objectives)
            index = len(plans)
            plans.append(ActionPlan(
                index=index + 1,
                units=[event],
                has_actions=True,
                user_prompt=user,
                prompt_hash=hashlib.sha256(user.encode("utf-8")).hexdigest(),
                prompt_tokens_est=len(user) // 4,
                event_start=event.event_id,
                event_end=event.event_id,
                event_count=1,
                view="action-only",
                view_label="action only",
                view_chunk_index=index + 1,
                action_index=index,
                action_number=number,
                tool_name=event.tool_name,
                tool_use_id=event.tool_use_id,
                action_text=action,
                history_events=0,
                history_cut=position > 0,
            ))
        for plan in plans:
            plan.view_chunk_total = len(plans)
        if not plans:
            raise ContextBudgetError("no tool calls to score in this trajectory")
        return plans

    def run(self, request: MonitorRequest, client: LLMClient, journal,
            plans: list[ChunkPlan] | None = None) -> MonitorResult:
        result = super().run(request, client, journal, plans)
        for view in result.views:
            view.update(name="action-only", label="action only")
        result.config.update(window="none", prompt_templates=list(self.prompt_templates))
        for key in ("prompt_budget_tokens", "exact_token_counts"):
            result.config.pop(key, None)
        return result
