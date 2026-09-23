# Case datasets

A dataset loader is an explicitly registered Python callable that accepts a
local `Path` and yields case records. The built-in `jsonl` loader reads one
JSON object per line; `esi.jsonl_v1` also converts documented ESI input aliases
to the preserved canonical field names. Every case has `case_id`, `input`, and
`expected`, with optional `metadata`, `split`, and `tags`:

```json
{"case_id":"toy-1","input":{"question":"A?"},"expected":{"answer":"A"},"split":"test","tags":["synthetic"]}
```

`load_configured_dataset(loaded)` resolves the experiment's loader and grader,
reads the entire file, rejects malformed records and duplicate IDs, and
validates selected expected labels before inference. You may select a `split`
or `case_ids`; a missing ID or empty selection is an error. `DatasetCase` keeps
the label separate from `agent_input()`. The latter returns only a fresh copy
of the input facts, so the same canonical facts can be passed to SAS and MAS
without exposing the expected answer. A custom loader can perform a domain
projection before returning records; it must keep targets in `expected`.

[`examples/esi/cases.jsonl`](examples/esi/cases.jsonl) contains three wholly
fabricated examples for offline wiring and fake-provider tests. They are not
patient records, a clinical benchmark, or validated triage guidance. Their
metadata explicitly records synthetic provenance and lack of clinical
validation. The old database seed cases are not copied into this dataset.
