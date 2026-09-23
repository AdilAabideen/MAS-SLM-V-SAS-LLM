# Human-readable experiment traces

The CLI writes structured JSON results to stdout and human progress to stderr. Its `--console` option accepts `none`, `summary`, `events`, or `full`, overriding `reporting.console` in experiment YAML. `--color auto|always|never` overrides the YAML color policy; the presence of `NO_COLOR` disables ANSI color even with `always`. Output wraps to the terminal width. For example:

```sh
PYTHONPATH=src python -m mas_slm_research.cli run examples/esi/experiment.yaml --fixture examples/esi/offline_fixture.json --system multi --case synthetic-esi1-001 --console events
```

`summary` names the case/system and final grade. `events` adds model and tool calls, agent starts/completions, handoffs, gate decisions, and recovery events in the captured sequence, with agent names on parallel branches. `full` adds available arguments, outcomes, and measurements. Keys containing API keys, passwords, secrets, authorization values, or access/refresh tokens are redacted recursively before printing. Full traces may still include ordinary case facts or other payload text; use `summary` or `none` for sensitive experiments.

MAS captures graph events and child agent events into one ordered in-memory timeline. The raw result, grade, usage, and comparison do not depend on the renderer. If a reporting callback fails, the experiment retains its attempts and records a reporting warning. `none` skips human output without changing results or accounting. This renderer observes completed attempts; RF-27 will add optional spans and RF-28 will persist selected events. It does not send data to a collector.
