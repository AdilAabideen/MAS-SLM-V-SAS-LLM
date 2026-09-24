# Split experiment configuration

An experiment selects registered Python components in YAML. The experiment
file names the SAS agent and model, the MAS default model and role assignments,
the dataset loader, and the grader. Its `mas.workflow` points to a second YAML
file containing the communication graph, payload builders, payload schemas,
source groups, and gates. See
[`examples/esi/experiment.yaml`](examples/esi/experiment.yaml) and
[`examples/esi/workflow.yaml`](examples/esi/workflow.yaml).

`load_configuration(path, registry=...)` validates both files without a model
call. Start with `ComponentRegistry()` and `register_builtin_components(...)`,
then explicitly register your own implementations or list extension modules
under `extensions`. An extension module must export
`register_components(registry)`; there is no automatic discovery or executable
YAML expression. The ESI agents, tools, schemas, workflow, payload builders,
final-acuity grader, and synthetic-case dataset loader are registered now.
The experiment scheduler and comparison CLI use these registered contracts;
the included ESI files are a fabricated offline software example.

Each model entry chooses one of `model_env`, `model_id`, or `catalog`. A
`request_policy` chooses `configured_v2` (default for new experiments) or
`legacy_v1` (explicitly retained by the ESI preservation example). Under
`configured_v2`, vLLM sends the resolved temperature and maximum output
tokens without the legacy hard-coded sampling/250-token override. Direct
legacy wrapper construction continues to default to `legacy_v1`. The built-in
`openai` provider ID is historically named but constructs Azure OpenAI; the
clearer `azure_openai` alias selects the same adapter. Both require an Azure
endpoint, API version, and key; use a separately registered
provider for standard OpenAI API access. `max_tokens` is forwarded to Azure.
Provider call records include the actual checkpoint and decoding parameters.
Provider-reported tokens are separated from estimates; retrying requests are
counted, while their unreported token/cost totals stay unknown.

`model_env` points to an environment variable containing the provider model ID.
`api_key_env`, `base_url_env`, and `api_version_env` name variables whose values must be present;
the resolved specification stores those names, never their values. The loader
resolves the workflow, dataset, and output paths relative to the experiment
file. `safe_snapshot()` contains the chosen IDs, model names, graph, and paths
without credentials. It can be saved or printed for review.

The loader rejects duplicate YAML keys, unknown fields and registration IDs,
blank required environment variables, unknown agents/models, incomplete role
assignments, missing route payload schemas, unreachable agents or finalizers,
and cycles in handoff routes. If `definition` selects a registered workflow
instance, the declarative graph must match it exactly; a disagreement is an
error rather than a silently ignored override. Error messages include the
declaring filename and the relevant field or route.

Per-role `payloads.<role>.input_schema` identifies the legacy case-input
contract for inspection; it is resolved at load time. The preserved ESI graph
does not enforce those strict models on each projected payload because its
case text can have missing observations. Generated handoff tools do enforce
the route-specific `handoff_schemas`. The `esi.jsonl_v1` loader validates and
normalizes selected case facts before inference; see [`DATASETS.md`](DATASETS.md).

The `sas.model` selection is explicit. If its agent also declares `model`, the
two must agree. For MAS, each role names an agent under `mas.agents`;
`mas.model_overrides` can select a different configured model per role. Agent
level model choices and the MAS default use this precedence: role override,
agent model, then MAS default. `build_configured_systems(loaded)` constructs
the SAS and MAS case runners. The optional `model_factory(resolved_model, role)`
argument supplies offline fake models without connecting to a provider.

Call `inspect_configuration(loaded)` before spending inference resources. Its
`concise` view shows each SAS/MAS model, agent definition, provider checkpoint,
tool order, payload builder/schema, route, gate, and runtime policy. Its
`details` view adds the exact assembled prompts and tool descriptions/JSON
schemas. `render(details=False)` and `render(details=True)` return formatted
JSON. The preview constructs the actual kernels with an inert model whose
inference method raises if called; it never instantiates a registered provider.
For the preserved vLLM adapter it shows both configured model settings and
the effective request values, including the legacy 250-token cap when
`request_policy: legacy_v1` is selected.

`agents.<alias>.runtime` can set `max_model_calls`, `max_tool_calls_total`, and
`max_elapsed_seconds` for a case's agent loop. MAS can additionally set
`mas.max_handoffs` and `mas.max_elapsed_seconds` for the whole graph. Limits
must be positive; when absent, the preserved legacy runtime remains unbounded.
Exhaustion produces a failed attempt with a budget or timeout reason and
observed call counts, not a successful prediction. A future named strict
profile will provide finite defaults without silently changing legacy runs.
