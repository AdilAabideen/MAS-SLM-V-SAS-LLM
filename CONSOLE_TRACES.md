# Human-readable experiment traces

The CLI writes structured JSON results to stdout and human progress to stderr. Its `--console` option accepts `none`, `summary`, `events`, or `full`, overriding `reporting.console` in experiment YAML. `--color auto|always|never` overrides the YAML color policy; the presence of `NO_COLOR` disables ANSI color even with `always`. Output wraps to the terminal width. For example:

```sh
PYTHONPATH=src python -m mas_slm_research.cli run examples/esi/experiment.yaml --fixture examples/esi/offline_fixture.json --case synthetic-esi1-001 --console full
```

`summary` shows a clearly labeled SAS or MAS case, its agent, and a compact result. `events` adds tool-call names, active MAS agents, handoffs, gates, and errors in execution order. `full` also prints each tool call's arguments as indented, multiline JSON with braces. Successful tool results and duplicate assistant final payloads are omitted from the human trace; the complete execution data remains in the machine-readable JSON and artifacts. Each line is flushed to stderr when its event occurs, before the case finishes; the model's own response tokens are not streamed. Keys containing API keys, passwords, secrets, authorization values, or access/refresh tokens are redacted recursively before printing. Full traces may still include ordinary case facts or other payload text; use `summary` or `none` for sensitive experiments.

MAS captures graph events and child agent events into one ordered in-memory timeline while printing them live. The raw result, grade, usage, and comparison do not depend on the renderer. If a reporting callback fails, the experiment retains its attempts and records a reporting warning. `none` skips human output without changing results or accounting. This renderer does not send data to a collector.
