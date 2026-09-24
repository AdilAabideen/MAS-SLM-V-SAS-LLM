# Paired experiment scheduling

`mas_slm_research.experiment.run_configured_experiment(loaded)` is the high-level asynchronous entry point. It loads and validates every selected dataset case and expected label before constructing any models. Pass `model_factory=` for deterministic offline providers. The lower-level `run_experiment(loaded, dataset=..., systems=..., grader=...)` accepts already assembled components and repeats expected-label preflight before the first case.

The default `system_major` schedule runs every case and repetition through the single-agent system, followed by every case and repetition through the multi-agent workflow. `case_major` is also supported when requested in the experiment configuration. Both arms receive independent copies of the same normalized case facts through `DatasetCase.agent_input()`; only the grader sees `expected_label()`.

Each `ExperimentAttempt` records a sequence number, `(case_id, repetition, system_id)` correlation key, UTC start/end timestamps, resolved provider/model/decoding choices, effective runtime policy, the raw in-memory execution trace when available, the terminal `CaseResult`, and its separate `GradeResult`. Unexpected runner exceptions become failed attempts and do not prevent the other arm from running. The `ExperimentRun.pairs` collection has a single and multi slot for every case/repetition, including failures. A normal completed experiment fills both slots exactly once.

An explicit cancellation callback can stop scheduling before the next attempt. Cancellation during a runner call marks that attempt failed with `experiment_cancelled`, grades it as an execution failure, and returns a run with `status="cancelled"`. Unstarted pair slots remain empty so partial work is visible. Execution is sequential; this scheduler makes no distributed or resumable-run claim.

`run_experiment` performs execution and per-attempt grading. The comparison report and saved artifacts carry the selected `runtime_profile` and registered grader ID, with each role's effective policy in the attempt record. The committed ESI cases are fabricated software examples and are not clinical evidence.
