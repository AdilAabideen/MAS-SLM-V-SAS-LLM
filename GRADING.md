# Grading a research case

The runner returns a `CaseResult` for each SAS or MAS attempt. A grader reads
the expected label separately from the case facts sent to agents. Subclass
`BaseGrader` and implement `validate_expected`, `evaluate`, and `aggregate`,
or register an object with the same three methods. `evaluate` receives only a
completed, validated final output and returns a `GradeDecision` with a boolean
judgment, a score from 0 to 1, and optional diagnostic details.

```python
from mas_slm_research.grading import BaseGrader, GradeDecision

class ExactAnswerGrader(BaseGrader):
    def validate_expected(self, expected):
        if set(expected) != {"answer"}:
            raise ValueError("answer label required")

    def evaluate(self, expected, actual):
        correct = actual.get("answer") == expected["answer"]
        return GradeDecision(passed=correct, score=float(correct))

    def aggregate(self, results):
        return {
            "attempts": len(results),
            "correct": sum(item.passed is True for item in results),
        }
```

Register the class or an instance under `graders`, then call `require_grader`
and `grade_case(grader, expected=..., result=...)`. Every attempt receives a
`GradeResult`: `graded`, `execution_failed`, or `grader_error`. Execution
failure keeps its provider/tool/runtime classification and scores zero;
grader errors have unknown judgment and score. `aggregate_grades` passes all
case-grade records to the grader's aggregate hook. The registered dataset
loader validates selected expected labels before inference; see
[`DATASETS.md`](DATASETS.md).

The built-in `esi.final_acuity_v1` grader applies the preserved exact-acuity
rule to the final `final_esi_level` from either SAS or MAS. Its summary reports
both all-attempt and graded-only denominators. Specialist agent diagnostics
may be recorded separately; they do not substitute for final-task accuracy.
The legacy doctor's `doctor_always_pass` evaluator is explicitly labeled a
placeholder diagnostic and is never the headline final-acuity grader.

For preserved specialist analyses, `evaluate_legacy_diagnostic` accepts the
old `evaluate(expected_json, actual_json, agent_status=...)` interface and
returns a separate `AgentDiagnostic`. `aggregate_legacy_diagnostics` calls
that evaluator's original aggregate on matching diagnostic records. These
records carry a `diagnostic_only` marker; the doctor placeholder also carries
`placeholder: true`. Neither changes a `GradeResult` for the final system
output.
