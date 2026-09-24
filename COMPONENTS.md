# Registering research components

Create a Python module with `register_components(registry)`. The experiment
loader will import only the module names explicitly selected by the researcher.
For now, a clone can call `registry.load_module("my_research.components")`.
Registration IDs are unique within each component kind; a duplicate or unknown
ID raises an error before inference. A failed extension registration is rolled
back as one unit.

```python
from mas_slm_research.registry import ComponentRegistry

def register_components(registry: ComponentRegistry) -> None:
    registry.register("agents", "my.agent_v1", MyAgentDefinition)
    registry.register("dataset_loaders", "my.jsonl_v1", load_my_cases)
    registry.register("graders", "my.score_v1", MyGrader)
```

The component kinds are `providers`, `models`, `agents`, `tools`, `schemas`,
`payload_builders`, `workflows`, `dataset_loaders`, and `graders`. Providers,
payload builders, and dataset loaders are callable. Agent definitions may be
callable or subclass `AgentDefinition` and implement `build_kernel`. Tools are callable
or expose `invoke`. Schemas are Pydantic `BaseModel` subclasses. Workflows are
`WorkflowDefinition` instances or factories. Models are `ModelSpec` instances
or factories. A grader is a class/factory or an instance exposing
`validate_expected`, `evaluate`, and `aggregate`. Registration itself never invokes a model or loads data. The current runner
contracts are demonstrated by the copyable [external extension](examples/external_extension/README.md).

Call `register_builtin_components(registry)` to install the preserved ESI
workflow, its role payload builders, and the existing model catalog. The
registration remains separate from runtime construction, so inspecting the
inventory does not connect to any provider. Built-in ESI agents, tools, and
schemas are also registered. Built-in provider IDs are `openai` and
`azure_openai` for the historical Azure adapter, `openai_api` for the standard
OpenAI API, `dr7`, and `vllm`. Use `registry.inventory()` or
`registry.ids(kind)` to see effective IDs, and `registry.resolve(kind, id)` to
obtain an implementation.

The built-in dataset loaders are `jsonl` and `esi.jsonl_v1`. A registered
loader receives a local path and yields case records with separate `input`
and `expected` fields; see [`DATASETS.md`](DATASETS.md).

The grader contract is now `BaseGrader` or a structurally compatible object
with the same three methods. See [`GRADING.md`](GRADING.md). The built-in
`esi.final_acuity_v1` grader scores the same final task for both systems;
legacy specialist and doctor diagnostics do not become headline accuracy.

An `AgentDefinition.build_kernel` method receives the selected `model`,
`runtime_config`, resolved `workflow` (or `None` for SAS), route-to-schema
`handoff_schemas`, and the `registry`. It returns a fresh `AgentKernel` for
each case attempt. A provider factory receives a `ResolvedModel` and an
environment mapping; it returns a LangChain-compatible chat model. A role
payload builder receives the scoped MAS state and returns a dictionary with
an `llm_payload` dictionary. The configured constructor validates declared role names, route schemas, and
final-output schemas before a case can run.
