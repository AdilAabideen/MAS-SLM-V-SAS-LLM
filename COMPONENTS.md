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
agents, payload builders, and dataset loaders are callable. Tools are callable
or expose `invoke`. Schemas are Pydantic `BaseModel` subclasses. Workflows are
`WorkflowDefinition` instances or factories. Models are `ModelSpec` instances
or factories. A grader is a class/factory or an instance exposing `evaluate`
and `aggregate`. The runner-specific arguments and return contracts will be
stabilized with the configuration and experiment-runner tickets; registration
itself never invokes a model or loads data.

Call `register_builtin_components(registry)` to install the preserved ESI
workflow, its role payload builders, and the existing model catalog. The
registration remains separate from runtime construction, so inspecting the
inventory does not connect to any provider. Use `registry.inventory()` or
`registry.ids(kind)` to see effective IDs, and `registry.resolve(kind, id)` to
obtain an implementation.
