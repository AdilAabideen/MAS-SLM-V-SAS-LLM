# Single-agent versus multi-agent research toolkit

This Python project runs paired experiments comparing a single-agent system (SAS) with a multi-agent system (MAS) on the same cases. Researchers register models, agent definitions, tools, workflows, dataset loaders, and graders in Python, then select those components in strict YAML. The toolkit records the effective configuration, model and tool calls, handoffs, gates, grades, timings, and portable per-case artifacts. Its main research path runs from a clone without a database or API server.

The preserved Emergency Severity Index (ESI) comparison is one benchmark, not a requirement of the core. A non-medical [arithmetic benchmark](examples/arithmetic/README.md) and a [copyable external word-count extension](examples/external_extension/README.md) demonstrate the same interfaces. Offline fixtures use fabricated responses; their passing grades only show that the software route works. The ESI benchmark is not clinically validated.

## Clone-first quickstart

Use Python 3.11 or newer. From a clone of this repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research.txt
PYTHONPATH=src python -m mas_slm_research.cli validate examples/esi/experiment.yaml --fixture examples/esi/offline_fixture.json
PYTHONPATH=src python -m mas_slm_research.cli inspect examples/esi/experiment.yaml --fixture examples/esi/offline_fixture.json
PYTHONPATH=src python -m mas_slm_research.cli compare examples/esi/experiment.yaml --fixture examples/esi/offline_fixture.json --no-artifacts --console events
```

The fixture supplies all model responses and dummy environment values; these commands make no provider requests. `validate` rejects incomplete configuration or data before inference. `inspect` shows the SAS/MAS models and agents, ordered tools, workflow routes, payload schemas, gates, and effective model choices. `compare` runs both systems on the same three synthetic ESI cases, writes the ordered human trace to stderr, and emits machine-readable JSON to stdout. To save a run, omit `--no-artifacts` and choose a new `--output-dir results/my-run`; `summarize results/my-run` reconstructs its totals without inference. See [CLI.md](CLI.md) and [ARTIFACTS.md](ARTIFACTS.md).

For a standard OpenAI API baseline against DR7 MedGemma specialists, use [`examples/esi/experiment-dr7.yaml`](examples/esi/experiment-dr7.yaml). Export `OPENAI_API_KEY`, `BASELINE_MODEL_ID`, `DR7_API_KEY`, and `DR7_BASE_URL`, then run `validate`, `inspect`, or `compare` against that file **without** `--fixture` for live requests. The DR7 adapter appends `/chat/completions` to the configured base URL. The existing `provider: openai` ID remains an Azure adapter for historical configurations; `provider: openai_api` selects the standard OpenAI API. For a no-key software check of this configuration, pass `--fixture examples/esi/offline_fixture_openai_dr7.json`; it replaces both live providers with scripted responses.

An editable package install is optional:

```sh
python -m pip install -e .
mas-slm --help
```

The package also builds as a wheel and includes a copyable ESI benchmark fixture. The wheel and source archive include the three examples; a wheel installs them under its environment's `share/mas-slm-research/examples/` directory. The project has not been published to a package index. [ESI_BENCHMARK.md](ESI_BENCHMARK.md) explains what is preserved and how to materialize bundled assets.

## Configure a study

The top-level YAML identifies the SAS agent/model, MAS role-to-agent mapping and default/overridden models, split workflow file, dataset loader, grader, schedule, repetitions, console mode, artifact location, and optional tracing. The workflow YAML explicitly declares start/final agents, allowed handoffs, source groups, gates, payload builders, and handoff schemas. YAML names registered IDs; it never executes Python. `inspect` is the way to verify the effective assignments and communication path before spending inference resources. [CONFIGURATION.md](CONFIGURATION.md) describes validation and model precedence.

Implement your components in a Python module exposing `register_components(registry)`, then name that module under `extensions` in the experiment YAML. `AgentDefinition.build_kernel` is subclassable; `BaseGrader` has `validate_expected`, `evaluate`, and `aggregate`. A dataset row has a stable `case_id`, `input` visible to agents, and a separate `expected` label visible only to the grader. The [external extension](examples/external_extension/README.md) shows a registered workflow, typed handoff, payload builder, grader subclass, per-agent model override, and a copied-directory run with no core edits. See [COMPONENTS.md](COMPONENTS.md), [DATASETS.md](DATASETS.md), and [GRADING.md](GRADING.md).

## Live models and research output

Without `--fixture`, registered provider factories construct the actual models. Set the environment variables named in your YAML before `validate`, `inspect`, or `compare`. In the ESI example, `BASELINE_MODEL_ID`, `BASELINE_API_KEY`, `BASELINE_AZURE_ENDPOINT`, and `BASELINE_AZURE_API_VERSION` configure the baseline Azure/OpenAI adapter; `SPECIALIST_MODEL_ID`, `SPECIALIST_API_KEY`, and `SPECIALIST_BASE_URL` configure the vLLM specialist endpoint. A different provider is a registered Python factory selected by ID, not a hardcoded server setting. Never put credentials directly in YAML or shared artifacts. Provider requests may incur cost and transfer case content to the configured service.

The console renderer has `none`, `summary`, `events`, and `full` modes and `auto`, `always`, or `never` color. It shows model/tool events, agent handoffs, gate readiness, final grades, and errors. [CONSOLE_TRACES.md](CONSOLE_TRACES.md) explains the trace. Artifacts save a versioned manifest, resolved non-secret configuration, per-attempt JSONL, a paired summary and CSV, and optional payload-free event metadata. The summary distinguishes graded attempts from failures; wall time is separate from summed child durations. [COMPARISON.md](COMPARISON.md) explains the measurements.

Tracing is off by default. `--trace` enables the configured in-memory span tree or optional OTLP HTTP export; `--no-trace` disables it. For OTLP, install the optional `otel` extra and name endpoint/header environment variables in YAML. [TRACING.md](TRACING.md) gives the exporter configuration and limits.

Named runtime policies (`legacy_v1`, `strict_v1`, and `slm_assisted_v1`) make recovery behavior and finite budgets explicit. Reports distinguish clean first-pass completions from repaired completions and record their extra model work. [POLICIES.md](POLICIES.md) describes profile settings and model-size controls.

The retired FastAPI/SQLite application has been removed. The research CLI runs directly from the extracted package without a server or database. Historical ESI prompts, tool ordering, routes, and scoring have preservation tests; corrected policies use new versioned registrations so old and new experiment results remain distinguishable. Token telemetry uses a deterministic character estimate by default, including offline runs. Call `TokenEstimator(prefer_tiktoken=True)` to opt into a locally available tokenizer; if it cannot load, estimation falls back to characters.
