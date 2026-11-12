"""Write monitor results .  a self-contained, human-readable dir per scoring."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from long_context_monitoring.monitor.scoring import MonitorResult
from long_context_monitoring.monitor.visibility import is_diagnostic_visibility
from long_context_monitoring.openrouter import OPENROUTER_BACKEND
from long_context_monitoring.run_layout import DEFAULT_RUNS_ROOT

_LENGTH_BIAS_CAVEAT = (
    "Max-aggregation caveat: taking the maximum over chunks, including chunks "
    "from overlapping strategy views, increases false-positive opportunity as "
    "the number of scored calls grows. Do not compare max scores across runs "
    "with different strategies or scored-call counts without calibration."
)


def run_dir_name(now_utc: datetime, run_id: str) -> str:
    """Legible, collision-resistant dir name: UTC timestamp + short run id.

    Dashes (not colons) so it is a valid directory name on every filesystem.
    """
    return now_utc.strftime("%Y-%m-%d_%H-%M-%SZ") + "_" + run_id


def make_run_dir(transcript_name: str, run_id: str, now_utc: datetime,
                 results_dir: Path = DEFAULT_RUNS_ROOT) -> Path:
    """Create and return the run dir (parents included). Called before the run
    so the journal can be written into it as chunks complete."""
    # ``results_dir`` is either a named batch's ``results/`` tree or the legacy
    # runs root. Keeping ``monitor/`` in both shapes preserves the per-source
    # rollup contract: write_records puts index.json at run_dir.parent.
    run_dir = Path(results_dir) / transcript_name / "monitor" / run_dir_name(now_utc, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class Journal:
    """Append-only per-chunk checkpoint log, and the resume source.

    Each completed chunk writes one JSON line ``{prompt_hash, status,
    raw_response, chunk, settings}``. On construction we load any existing
    completed lines so ``completed(prompt_hash)`` can short-circuit a re-run,
    and ``completed_settings(prompt_hash)`` lets the caller refuse a reuse
    under a different model, provider, temperature, effort, or prompt version.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.dir = self.path.parent
        self._completed: dict[str, dict] = {}
        self._settings: dict[str, dict | None] = {}
        # Chunks may be scored on several threads (see ``score_chunks``). A
        # journal line carries the whole raw response, so it can exceed the
        # size a single append is atomic at, and two threads writing at once
        # would interleave halves of two lines into one unparseable record.
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("status") == "completed" and record.get("prompt_hash"):
                    self._completed[record["prompt_hash"]] = record.get("chunk")
                    self._settings[record["prompt_hash"]] = record.get("settings")

    def completed(self, prompt_hash: str) -> dict | None:
        return self._completed.get(prompt_hash)

    def completed_settings(self, prompt_hash: str) -> dict | None:
        """The settings a completed chunk was scored under; None for a line
        written before settings were journalled."""
        return self._settings.get(prompt_hash)

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            if record.get("status") == "completed" and record.get("prompt_hash"):
                self._completed[record["prompt_hash"]] = record.get("chunk")
                self._settings[record["prompt_hash"]] = record.get("settings")

    def write_prompt(self, index: int, user: str) -> None:
        """Persist a chunk's rendered prompt for debug / dry-run parity."""
        path = self.dir / f"chunk-{index:02d}.prompt.txt"
        path.write_text(user + "\n", encoding="utf-8")


def _append_view_report(out: list[str], result: MonitorResult) -> None:
    """Keep per-view scores and their reasoning together in the rendered report."""
    if len(result.views) > 1:
        out.append("## Per-view scores")
        out.append("")
        out.append("| view | score | band | chunks | events |")
        out.append("| --- | ---: | --- | ---: | ---: |")
        for view in result.views:
            out.append(
                f"| {view['label']} | {view['score']} | {view['band']} | "
                f"{view['chunk_count']} | {view['event_count']} |"
            )
        out.append("")
        for view in result.views:
            out.append(f"### {view['label']} reasoning")
            out.append("")
            out.append(view["reasoning"])
            out.append("")


def _append_chunk_report(out: list[str], result: MonitorResult) -> None:
    """Pair chunk scores with the warning about max-aggregation bias."""
    if result.chunk_count > 1:
        out.append("## Per-chunk scores")
        out.append("")
        out.append("| chunk | view | score | band | events | input tok | output tok |")
        out.append("| ---: | --- | ---: | --- | ---: | ---: | ---: |")
        for chunk in result.chunks:
            out.append(
                f"| {chunk['index']} | {chunk.get('view_label', 'whole trajectory')} | "
                f"{chunk['score']} | {chunk['band']} | {chunk['event_count']} | "
                f"{chunk['input_tokens']} | "
                f"{chunk['output_tokens']} |"
            )
        out.append("")
        out.append(f"_{_LENGTH_BIAS_CAVEAT}_")
        out.append("")


def build_report(result: MonitorResult) -> str:
    """Render a concise human-readable report for one monitor run."""
    out: list[str] = []
    out.append(
        f"# {result.transcript} — {result.model} — {result.strategy} — "
        f"score {result.score}/10 ({result.band}, {result.chunk_count} chunk"
        f"{'s' if result.chunk_count != 1 else ''})"
    )
    out.append("")
    out.append(f"_Generated {result.timestamp} by `python -m src.monitor`._")
    out.append("")
    out.append("| field | value |")
    out.append("| --- | --- |")
    out.append(f"| visibility | {result.visibility} |")
    out.append(f"| diagnostic | {'yes' if result.config.get('diagnostic') else 'no'} |")
    out.append(f"| objectives provided | {result.objectives_provided} |")
    out.append(f"| strategy | {result.strategy} |")
    if result.views:
        out.append(f"| views | {len(result.views)} |")
    out.append(f"| chunks | {result.chunk_count} |")
    out.append(f"| est. prompt tokens | {result.prompt_tokens_est} |")
    out.append(f"| actual input tokens | {result.totals.get('input_tokens', 0)} |")
    out.append(f"| actual output tokens | {result.totals.get('output_tokens', 0)} |")
    out.append(f"| cost estimate (USD) | {result.cost_estimate_usd:.4f} |")
    truncate = result.config.get("truncate_tool_results_chars")
    out.append(
        f"| tool-result truncation | {truncate if truncate else 'none (full)'} |"
    )
    out.append("")
    headline_view = result.config.get("headline_view")
    if headline_view:
        out.append(f"## Reasoning (headline view: {headline_view})")
    else:
        out.append("## Reasoning (max-scoring chunk)")
    out.append("")
    out.append(result.reasoning)
    out.append("")
    _append_view_report(out, result)
    _append_chunk_report(out, result)
    out.append(
        "_Visibility is stamped above; scores from different visibility modes are "
        "not comparable._"
    )
    out.append("")
    return "\n".join(out) + "\n"


def _update_index(index_path: Path, result: MonitorResult, run_dir: Path) -> None:
    """Load-append-write the per-transcript rollup; latest run is runs[-1]."""
    data = {"transcript": result.transcript, "runs": []}
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    data.setdefault("runs", [])
    # A resumed run rewrites its own entry rather than appending a second one.
    data["runs"] = [run for run in data["runs"] if run.get("run_dir") != run_dir.name]
    data["runs"].append({
        "run_dir": run_dir.name,
        "timestamp": result.timestamp,
        "score": result.score,
        "band": result.band,
        "model": result.model,
        "strategy": result.strategy,
        "visibility": result.visibility,
        "diagnostic": bool(result.config.get("diagnostic")),
        "objectives_provided": result.objectives_provided,
        "chunk_count": result.chunk_count,
        "view_count": len(result.views),
        "cost_estimate_usd": result.cost_estimate_usd,
    })
    index_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a reader sees the old file or the whole new one."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_records(result: MonitorResult, run_dir: str | Path) -> Path:
    """Write result.json + report.md into ``run_dir`` and update the index.

    The journal and chunk prompts are already in ``run_dir`` from the run. Returns
    the run dir.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # result.json marks a run as finished (a sweep skips arms that have one),
    # so it must never exist half-written: write a temporary file, then rename.
    _write_atomic(run_dir / "result.json",
                  json.dumps(asdict(result), indent=2, ensure_ascii=False) + "\n")
    _write_atomic(run_dir / "report.md", build_report(result))
    _update_index(run_dir.parent / "index.json", result, run_dir)
    return run_dir


@contextmanager
def _exclusive_lock(lock_path: Path):
    """Hold an exclusive lock on ``lock_path`` for the duration of the block.

    Why: many `lcm monitor` processes may score into the same batch at once,
    and the manifest update below is a read-modify-write. Without a lock two
    processes finishing at the same moment silently drop each other's entry.
    The lock file sits beside the manifest and is never deleted (deleting a
    lock file while another process holds it reopens the race).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as handle:
        if os.name == "nt":
            _lock_windows_file(handle)
        else:
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _lock_windows_file(handle) -> None:
    """Wait for a one-byte Windows file lock without imposing a timeout."""

    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()

    while True:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            time.sleep(0.05)


def update_batch_manifest(batch_root: str | Path, result: MonitorResult,
                          run_dir: str | Path, source: dict) -> Path:
    """Append one completed result to a dated monitor batch manifest.

    The result/report remain the authoritative scoring artifacts. This compact
    index makes a cross-spec batch self-describing without copying transcripts.
    Safe to call from concurrent processes: the whole read-modify-write runs
    under an exclusive file lock.
    """
    batch_root = Path(batch_root)
    run_dir = Path(run_dir)
    manifest_path = batch_root / "manifest.json"
    with _exclusive_lock(batch_root / ".manifest.lock"):
        return _update_batch_manifest_locked(
            batch_root, manifest_path, result, run_dir, source
        )


# Compared arms in one batch must share these settings. Reject differences
# before combining results, including changes to the monitor prompt.
BATCH_CONFIGURATION_KEYS = (
    "model", "strategy", "visibility", "objectives_provided", "prompt_version",
    "reasoning_effort", "provider", "provider_quantizations", "backend",
)

BATCH_CONFIGURATION_DEFAULTS = {"backend": OPENROUTER_BACKEND}


def assert_batch_configuration_compatible(
    batch_root: str | Path, configuration: dict
) -> None:
    """Refuse a run whose configuration differs from what the batch holds.
    Why: the manifest appends every distinct configuration it sees, so a
    second model or strategy would sit silently beside the first.
    If wrong: one batch pools two conditions and any rate read from it is
    a mixture.
    """
    manifest_path = Path(batch_root) / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    for recorded in data.get("configurations") or []:
        if not isinstance(recorded, dict):
            continue
        def value(source: dict, key: str):
            fallback = BATCH_CONFIGURATION_DEFAULTS.get(key)
            return _normalised(source.get(key) if source.get(key) is not None
                               else fallback)

        differing = [
            key for key in BATCH_CONFIGURATION_KEYS
            if value(recorded, key) != value(configuration, key)
        ]
        if differing:
            detail = "; ".join(
                f"{key}: batch {recorded.get(key)!r} vs run {configuration.get(key)!r}"
                for key in differing
            )
            raise ValueError(
                f"monitor batch {Path(batch_root).name!r} already holds a different "
                f"configuration ({detail}); use a new --batch-name"
            )


def _normalised(value):
    """Lists and tuples compare equal; None and an empty list mean 'unpinned'."""
    if isinstance(value, (list, tuple)):
        return list(value) or None
    return value


def assert_batch_visibility_compatible(
    batch_root: str | Path, visibility: str
) -> None:
    """Reject mixing diagnostic and official scores in one batch."""

    manifest_path = Path(batch_root) / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    expected = is_diagnostic_visibility(visibility)
    recorded = data.get("diagnostic")
    if recorded is None:
        configurations = data.get("configurations") or []
        recorded_modes = {
            is_diagnostic_visibility(item.get("visibility", "tools-only"))
            for item in configurations
            if isinstance(item, dict)
        }
        if len(recorded_modes) == 1:
            recorded = recorded_modes.pop()
    if recorded is not None and bool(recorded) != expected:
        raise ValueError(
            "monitor batch cannot mix official and diagnostic visibility modes"
        )


def _update_batch_manifest_locked(batch_root: Path, manifest_path: Path,
                                  result: MonitorResult, run_dir: Path,
                                  source: dict) -> Path:
    """The manifest read-modify-write; caller holds the batch lock."""
    data = {
        "batch_name": batch_root.name,
        "date_utc": batch_root.parent.name,
        "sources": [],
        "configurations": [],
        "results": [],
        "result_count": 0,
        "recorded_cost_usd": 0.0,
    }
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            loaded = None
        if isinstance(loaded, dict):
            data = loaded

    expected_diagnostic = bool(result.config.get("diagnostic"))
    recorded_diagnostic = data.get("diagnostic")
    if (
        recorded_diagnostic is not None
        and bool(recorded_diagnostic) != expected_diagnostic
    ):
        raise ValueError(
            "monitor batch cannot mix official and diagnostic visibility modes"
        )
    data["diagnostic"] = expected_diagnostic

    sources = data.setdefault("sources", [])
    if not any(existing.get("sha256") == source.get("sha256") for existing in sources):
        sources.append(source)
        sources.sort(key=lambda item: str(item.get("transcript", "")))

    configuration = {
        "model": result.model,
        "strategy": result.strategy,
        "visibility": result.visibility,
        "diagnostic": expected_diagnostic,
        "objectives_provided": result.objectives_provided,
        "prompt_version": result.config.get("prompt_version"),
        "reasoning_effort": result.config.get("reasoning_effort"),
        "provider": result.config.get("provider"),
        "provider_quantizations": result.config.get("provider_quantizations"),
        # Which transport served the calls. Part of the compatibility key, so
        # the manifest has to carry it or the check has nothing to compare.
        "backend": result.config.get("backend"),
    }
    configurations = data.setdefault("configurations", [])
    if configuration not in configurations:
        configurations.append(configuration)

    record = {
        "transcript": result.transcript,
        "source_sha256": source.get("sha256"),
        "path": run_dir.relative_to(batch_root).as_posix(),
        "timestamp": result.timestamp,
        "score": result.score,
        "band": result.band,
        "model": result.model,
        "strategy": result.strategy,
        "visibility": result.visibility,
        "diagnostic": expected_diagnostic,
        "objectives_provided": result.objectives_provided,
        "prompt_version": result.config.get("prompt_version"),
        "reasoning_effort": result.config.get("reasoning_effort"),
        "provider": result.config.get("provider"),
        "provider_quantizations": result.config.get("provider_quantizations"),
        "backend": result.config.get("backend"),
        "cost_estimate_usd": result.cost_estimate_usd,
    }
    results = data.setdefault("results", [])
    if not any(existing.get("path") == record["path"] for existing in results):
        results.append(record)
    data["result_count"] = len(results)
    data["recorded_cost_usd"] = round(
        sum(float(item.get("cost_estimate_usd") or 0.0) for item in results),
        12,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path
