# Portable experiment artifacts

When `reporting.output_directory` is set, `compare` creates that directory and saves the research run there. Use `--output-dir NEW_PATH` to choose another new directory or `--no-artifacts` for a JSON-only run. The writer refuses to overwrite an existing directory. `--events-file` additionally saves payload-free event metadata; the default leaves that file out. `results/` is ignored by Git so local experiments are not accidentally committed.

The directory contains:

| File | Contents |
| --- | --- |
| `manifest.json` | Artifact schema version, run state, experiment ID, schedule, case/repetition keys, source revision and dirty flag when available, package version or `null`, source/config/file SHA-256 hashes, attempted count, and explicit cost rates. |
| `resolved-config.json` | Validated configuration snapshot with environment variable names but no credential values. |
| `results.jsonl` | One flushed line per completed attempt, including terminal result, grade, model/policy choices, timestamps, and the minimal call facts needed to recompute totals. |
| `summary.json` | Final paired comparison and system totals, written only after reconciliation. |
| `comparison.csv` | Case/repetition pairs, both run IDs and scores, and winner where comparable. |
| `events.jsonl` | Optional ordered event names, agent/hand-off IDs, tool names, and statuses without prompts or payloads. |

The writer flushes each result and updates the manifest hash before the next attempt. A normal finish marks the manifest `completed`; a cancelled run marks it `cancelled`; an interrupt leaves `interrupted` with the attempts written so far. `in_progress` is also readable if the process stops before its final state update. Failed cases remain result lines and pair members. Missing usage or cost stays `null`, not zero. Config/workflow/dataset digests record provenance even if the source files are absent when the artifact directory is moved.

`PYTHONPATH=src python -m mas_slm_research.cli summarize PATH_TO_DIRECTORY` reads `results.jsonl` and recomputes generic system totals and pair outcomes without constructing a model. It uses recorded per-attempt grades, call facts, and explicit rates. A task-specific grader summary is preserved as a snapshot because a custom grader may not be available on the receiving machine. The reader checks the authoritative records' hash and identities; it can recompute even if derived `summary.json` or `comparison.csv` is missing. `summarize_artifacts(path, verify_derived=True)` also checks those derived files against the manifest.

The writer recursively redacts credential-key fields and any configured API-key or trace-header values before saving. It does not make arbitrary case text anonymous: outputs and grade diagnostics may include task content. Review data provenance and distribution rights before sharing an artifact directory. The included ESI examples are explicitly fabricated software fixtures, not clinical validation data.

Artifact schema version 2 adds explicit runtime profile, grader ID, network attempts, provider request facts, and repair-call provenance. `summarize` still reads version 1 manifests, defaulting a missing profile to `legacy_v1` and a missing grader ID to unknown. Version 1 call facts did not label repair calls, so its first-pass/repaired classification can use saved recovery decisions, but exact extra repair tokens and time cannot be reconstructed when those call facts were absent. Newly written version 2 artifacts retain the call kind and parse source needed for exact reconstruction.
