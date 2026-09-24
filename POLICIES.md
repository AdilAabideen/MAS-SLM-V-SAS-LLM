# Versioned research policies and controls

Set `runtime_profile` at the top of an experiment YAML. The bundled ESI
preservation example explicitly selects `legacy_v1`; it is a historical
comparison baseline, not the recommended policy for a new study. The same
dataset, grader, schedule, and model assignments can be run under another
profile by copying the YAML, changing only `runtime_profile` and experiment
name/output directory, then comparing the resulting artifacts.

| Profile | Finalization and recovery | Finite defaults |
| --- | --- | --- |
| `legacy_v1` | Preserved plain-JSON/text-tool recovery and one malformed-call retry per tool | No whole-run limits unless configured |
| `strict_v1` | Requires the authorized final-answer tool; disables text-tool recovery and malformed-call retry | 8 model calls, 16 tool calls, 120 s per agent; 8 MAS handoffs and 240 s per graph |
| `slm_assisted_v1` | Allows text-tool recovery, one malformed-call retry per tool, and plain JSON final output | The same finite defaults as strict |

Agent-level `runtime` fields and MAS graph budget fields in YAML explicitly
override profile defaults. `inspect` shows the resolved policy for every role,
the MAS limits, and the chosen model request policy before inference. A failed
or budget-exhausted arm is never graded as a completed prediction. Recovery
cannot establish clinical or task accuracy by itself.

The paired summary reports `first_pass_valid` (completed without text recovery
or a malformed-call repair) separately from `repaired_valid` (completed after
one of those interventions). It also reports extra repair model calls, their
tokens when known, and their measured duration. Saved call facts identify
`call_kind=malformed_repair` and the parse source; `summarize` reconstructs
these counts without model access. A retry that never receives a response has
unknown output tokens. The grader ID and runtime profile appear in the
manifest and summary. `esi.final_acuity_v1` scores the final ESI level for
both SAS and MAS; the historical doctor always-pass rule is a diagnostic
placeholder and is excluded from headline accuracy. No ESI scoring rule was
changed in this milestone, so no new scoring ID was invented.

For model-size controls, hold cases, grader, prompts, runtime profile,
repetitions, and provider request policy constant while varying only model
assignments. The included ESI YAML is **large SAS versus small MAS** when its
environment variables point to those sizes. A second copy with
`sas.model: specialist` and `agents.baseline.model: specialist` is **small SAS
versus small MAS**. Compare its single arm with the original single arm for
the large/small SAS control, and compare the two small arms for the system
control. If available, a third copy with large MAS models can test the model
size interaction. Record exact checkpoint IDs, decoding parameters, usage
source, runtime profile, and dataset digest from each artifact. Model-size
labels are study metadata supplied by the researcher; the toolkit does not
infer parameter count from an ID. The offline fixtures only prove execution
paths and should not be interpreted as an effectiveness result.
