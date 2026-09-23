# Run the research toolkit from a clone

From the repository root, use `PYTHONPATH=src python -m mas_slm_research.cli`. The same commands are available as `mas-slm` after an editable installation. The CLI calls the same configuration, dataset, runner, grader, preview, and comparison APIs used by Python integrations; it does not start the old API or a database.

The tracked [`examples/esi/offline_fixture.json`](examples/esi/offline_fixture.json) supplies fixed fake model responses and dummy environment references for a software smoke test. It is not clinical data or an accuracy benchmark. For example:

```sh
PYTHONPATH=src python -m mas_slm_research.cli validate examples/esi/experiment.yaml --fixture examples/esi/offline_fixture.json
PYTHONPATH=src python -m mas_slm_research.cli inspect examples/esi/experiment.yaml --fixture examples/esi/offline_fixture.json
PYTHONPATH=src python -m mas_slm_research.cli run examples/esi/experiment.yaml --fixture examples/esi/offline_fixture.json --system single --case synthetic-esi1-001
PYTHONPATH=src python -m mas_slm_research.cli compare examples/esi/experiment.yaml --fixture examples/esi/offline_fixture.json --no-artifacts > /tmp/esi-report.json
PYTHONPATH=src python -m mas_slm_research.cli summarize /tmp/esi-report.json
PYTHONPATH=src python -m mas_slm_research.cli compare examples/esi/experiment.yaml --fixture examples/esi/offline_fixture.json --output-dir /tmp/esi-artifact-run --console summary
PYTHONPATH=src python -m mas_slm_research.cli summarize /tmp/esi-artifact-run
```

`validate` checks the full YAML/workflow/dataset/label contract before inference. `inspect --details` includes the assembled prompts and tool schemas. Both commands use no model requests. `run` selects one `single` or `multi` system and one case ID, returning a terminal result and a distinct grade. `compare` uses the configured schedule and repetitions, returning a complete JSON comparison. `summarize` accepts either a captured report JSON or an artifact directory; the directory path recomputes totals from per-attempt records without model calls.

Exit code `0` means the command itself completed; an incorrect prediction or a recorded failed case remains a research result in its JSON. Exit code `2` means invalid configuration, dataset, fixture, case selection, or report input. Exit code `3` means an unhandled infrastructure failure outside the case runner. Exit code `130` means a keyboard interruption or cancelled comparison. Errors go to stderr; JSON data goes to stdout. `compare` saves portable artifacts in the configured output directory by default; use `--output-dir` for a new directory or `--no-artifacts` for JSON only. See [ARTIFACTS.md](ARTIFACTS.md).

Without `--fixture`, model construction uses the explicitly registered providers and the environment variables named in the experiment YAML. The fixture path is deliberately opt-in and never sends model requests. The CLI uses the existing character-based token estimate for fixture runs when the tokenizer cache is unavailable; the general offline tokenizer repair remains KI-06/RF-35.

Human progress from `run` and `compare` is written to stderr while result JSON stays on stdout. Choose `--console none|summary|events|full` and `--color auto|always|never`, or use the experiment's reporting settings. [CONSOLE_TRACES.md](CONSOLE_TRACES.md) describes event order, redaction, and renderer failure behavior.

Tracing is disabled by default. `--trace` records the configured in-memory or OTLP spans, while `--no-trace` turns it off for that command. See [TRACING.md](TRACING.md) for the optional exporter and environment references.

For an unrelated task, run the [arithmetic example](examples/arithmetic/README.md) with `PYTHONPATH=src:examples/arithmetic`. The [external extension](examples/external_extension/README.md) shows a copied researcher module and a per-role model override.
