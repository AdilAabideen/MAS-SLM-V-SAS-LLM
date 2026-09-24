# Preserved ESI research benchmark

The ESI benchmark is a versioned research example of a single-agent system (SAS) and a multi-agent system (MAS) evaluating the same synthetic triage cases. It is not a clinical decision aid or a clinically validated ESI implementation. The offline fixture contains fabricated model responses so a clone can demonstrate the complete route without credentials or paid inference.

Run it from a clone:

```sh
PYTHONPATH=src python -m mas_slm_research.cli compare examples/esi/experiment.yaml \
  --fixture examples/esi/offline_fixture.json --no-artifacts --console events
```

`examples/esi/experiment.yaml` names the systems, model environments, dataset loader, grader, workflow, and reporting policy. `workflow.yaml` is the explicit MAS communication contract. `cases.jsonl` contains three synthetic cases; `offline_fixture.json` scripts deterministic responses for every role. These four files are also bundled byte-for-byte in `mas_slm_research.benchmarks.esi`. A researcher using an installed wheel can create an editable copy with:

```python
from pathlib import Path
from mas_slm_research.benchmarks.esi import copy_esi_benchmark

copy_esi_benchmark(Path("my-esi-benchmark"))
```

The SAS uses `esi.single_agent_v1` and a baseline model. Its ordered tools are danger-zone calculation, shock-index calculation, plan creation, and thought logging. The MAS assigns a specialist model to five roles: `esi.esi1_v1`, `esi.esi2_v1`, `esi.esi345_v1`, `esi.vitals_v1`, and `esi.doctor_v1`. The three acuity stages and doctor have the ordered plan/log tools; vitals also has danger-zone and shock-index tools before those two. Each definition carries its preserved prompt and Pydantic input/output schemas in `src/mas_slm_research/agents/esi/`. The registered IDs are defined in `src/mas_slm_research/agents/esi/definitions.py` and exposed by `register_builtin_components`.

The MAS starts `esi1_agent` and `vitals_agent` in parallel. The acuity branch can hand off from ESI1 to ESI2 or doctor, and from ESI2 to ESI345 or doctor; ESI345 hands off to doctor. Vitals hands off to doctor. Every edge has an explicit handoff payload schema in `workflow.yaml`, backed by the agent's `handoffs.py`. Each receiving role has a registered payload builder in `src/mas_slm_research/workflows/esi/payloads/`, which selects the appropriate case and handoff information from in-memory state. `doctor_gate` waits for handoffs from both the acuity and vitals sources before doctor runs; doctor is the only finalizing role. This prevents a single completed branch from prematurely producing the MAS answer. The workflow file records the participating agents, allowed edges, source membership, gate, metadata, payload builders, and schemas in inspectable form.

Both systems use `esi.final_acuity_v1`, a subclass of the public grader contract that compares each final ESI level with the dataset's expected acuity. Individual specialist agents are not graded separately. This is a **legacy-preservation** benchmark: its original prompts, ordered tools, handoff policy, graph routes, and final exact-acuity rule are intentionally identifiable and protected by compatibility tests. Any corrected clinical policy or altered scoring should receive a new versioned registration and an explicit experiment configuration, so results cannot be mistaken for the captured baseline. The example does not claim that the legacy policy is clinically correct.

The `inspect` command resolves model/agent roles and communication routes before inference. The `compare` command records tool calls, handoffs, gate outcomes, final results, and paired grades through the same research runner used by other benchmarks. See [CONFIGURATION.md](CONFIGURATION.md), [CLI.md](CLI.md), [GRADING.md](GRADING.md), and [ARTIFACTS.md](ARTIFACTS.md) for the corresponding public contracts.
