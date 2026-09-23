# Copyable researcher extension

This directory is a standalone example of definitions owned by a researcher. It can be copied to another folder without editing `src/mas_slm_research`. The data and model responses are scripted; the example tests interoperability, not language-model ability.

From the repository root, run the copied directory with its parent on the Python import path:

```sh
cp -R examples/external_extension /tmp/my-word-count-study
PYTHONPATH=src:/tmp/my-word-count-study python -m mas_slm_research.cli inspect \
  /tmp/my-word-count-study/experiment.yaml \
  --fixture /tmp/my-word-count-study/offline_fixture.json
PYTHONPATH=src:/tmp/my-word-count-study python -m mas_slm_research.cli compare \
  /tmp/my-word-count-study/experiment.yaml \
  --fixture /tmp/my-word-count-study/offline_fixture.json \
  --no-artifacts --console events
```

`experiment.yaml` explicitly loads `research_extension`, selects the baseline model for SAS, the small model for `counter_agent`, and a `mas.model_overrides.reviewer_agent` model for final review. `inspect` shows each effective role model, its ordered tools, the `counter_agent → reviewer_agent` route, the handoff schema, and each payload builder before inference. `workflow.yaml` defines the route and `reviewer_gate`; the extension registers the same workflow under `research.word_count_v1`. The loader checks that the declared file and registered implementation agree.

`research_extension.py` registers a custom `count_words` tool, Pydantic input/output/handoff schemas, three `AgentDefinition` instances, a registered workflow, two payload builders, and an `ExactWordCountGrader` subclass. The reviewer payload carries the source text and the validated handoff. `cases.jsonl` separates expected count from agent input. Invalid registration or an unknown ID is rejected at validation before a provider is called. The `offline_fixture.json` responses let the entire example run without credentials.

For your own study, replace the word-count schemas, agents, tool, grader, case rows, model choices, and fixture or live-provider settings. Keep Python implementation in an explicitly named extension module and keep runtime selections in YAML; YAML is data, never executable Python.
