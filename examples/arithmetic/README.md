# Offline arithmetic comparison

This deliberately small, non-medical benchmark proves that an experiment can use researcher-owned definitions without changing the toolkit. Its one case asks for the sum of `[2, 3]`. The scripted responses make the example deterministic and credential-free; they do not represent model quality. The SAS baseline and MAS specialists use separately named model configurations.

From the repository root, run:

```sh
PYTHONPATH=src:examples/arithmetic python -m mas_slm_research.cli compare \
  examples/arithmetic/experiment.yaml \
  --fixture examples/arithmetic/offline_fixture.json \
  --output-dir results/arithmetic-demo --console events --color never
PYTHONPATH=src:examples/arithmetic python -m mas_slm_research.cli summarize results/arithmetic-demo
```

Choose a new output directory for each run; artifact directories are never overwritten. Use `--no-artifacts` for a disposable comparison. `experiment.yaml` selects the extension, agents, models, generic JSONL loader, exact-sum grader, and split `workflow.yaml`. `arithmetic_components.py` registers all Python implementations. The source data and expected label are separate in `cases.jsonl`.

The MAS starts `adder_agent` and `checker_agent` together. Each invokes the `sum_numbers` tool and sends a validated `SumHandoff(total, method)` to `final_agent`. The `final_gate` requires handoffs from both sources. The final payload builder includes both results and the original numbers; the finalizer emits `SumOutput`. `ExactSumGrader`, a `BaseGrader` subclass, grades both systems by the same integer total. The saved summary and event stream show tool calls, both handoffs, gate readiness, final output, and paired grades.

The model IDs in the YAML are placeholders because `offline_fixture.json` supplies the responses. Remove `--fixture` only after configuring real providers. For a more portable external researcher module and per-agent model override, see the following extension example in this milestone.
