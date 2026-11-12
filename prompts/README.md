# Prompts

These files contain the model instructions used by the released tools.
The loader records template hashes with each new monitor result.

| File | Use |
|---|---|
| `agent_environment.md` | Coding-agent environment description. |
| `compaction.md` | Neutral summary generation during attack insertion. |
| `monitor.md` | Default suspicion scoring. |
| `monitor_catalog.md` | Objective catalog for informed monitoring. |
| `monitor_extract.md` | Evidence extraction for summary-based monitoring. |
| `monitor_merge.md` | Combining extracted evidence. |
| `monitor_annotated_history.md` | Scoring with a carried evidence record. |
| `monitor_sandbox.md` | Monitoring through transcript inspection tools. |
| `monitor_scratchpad.md` | Transcript inspection with a private scratchpad. |
| `monitor_sliding_window.md` | Scoring one action with the transcript before it (`sliding-window`). |
| `monitor_nano*.md` | Alternate scoring templates selected with `--monitor-template monitor_nano.md`. |

The monitor receives the host task and the events allowed by its visibility
setting. It receives the objective catalog unless `--blind` is set. Attack
registration text does not enter the monitor's task description.
