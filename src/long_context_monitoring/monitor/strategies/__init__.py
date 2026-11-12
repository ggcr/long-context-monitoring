"""The monitoring strategies included in this release."""
from long_context_monitoring.monitor.strategies.annotated_history import AnnotatedHistoryMonitor
from long_context_monitoring.monitor.strategies.base import ContextBudgetError, MonitorRequest, MonitorStrategy
from long_context_monitoring.monitor.strategies.hierarchical_summary import HierarchicalSummaryMonitor
from long_context_monitoring.monitor.strategies.intervals import CompactionIntervalsMonitor
from long_context_monitoring.monitor.strategies.sandbox import SandboxMonitor
from long_context_monitoring.monitor.strategies.sliding_window import SlidingWindowMonitor
from long_context_monitoring.monitor.strategies.scratchpad import ScratchpadMonitor
from long_context_monitoring.monitor.strategies.summaries_only import SummariesOnlyMonitor
from long_context_monitoring.monitor.strategies.whole import WholeMonitor

STRATEGIES: dict[str, MonitorStrategy] = {
    "whole": WholeMonitor(),
    "intervals": CompactionIntervalsMonitor(),
    "summaries-only": SummariesOnlyMonitor(),
    "hierarchical-summary": HierarchicalSummaryMonitor(),
    "annotated-history": AnnotatedHistoryMonitor(),
    "sandbox": SandboxMonitor(),
    "scratchpad": ScratchpadMonitor(),
    "sliding-window": SlidingWindowMonitor(),
}
DEFAULT_STRATEGY = "intervals"
__all__ = ["ContextBudgetError", "MonitorRequest", "MonitorStrategy", "STRATEGIES", "DEFAULT_STRATEGY", "get_strategy"]


def get_strategy(name: str) -> MonitorStrategy:
    """Return a registered strategy or report the available names."""
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; valid strategies: {', '.join(sorted(STRATEGIES))}") from None
