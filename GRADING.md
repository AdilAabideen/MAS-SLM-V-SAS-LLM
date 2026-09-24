# Grading a research case

The runner returns a `CaseResult` for each SAS or MAS attempt. A grader reads
the expected label separately from the case facts sent to agents. Subclass
`BaseGrader` and implement `evaluate`; override `validate_expected` when your
labels need a stricter shape, or `aggregate` when the default counts and
accuracy are insufficient. `evaluate` receives only a completed, validated
final output and returns a `GradeDecision` with a boolean judgment, a score
from 0 to 1, and optional diagnostic details.

```python
from mas_slm_research.grading import BaseGrader, GradeDecision

class ExactAnswerGrader(BaseGrader):
    def validate_expected(self, expected):
        if set(expected) != {"answer"}:
            raise ValueError("answer label required")

    def evaluate(self, expected, actual):
        correct = actual.get("answer") == expected["answer"]
        return GradeDecision(passed=correct, score=float(correct))

```

Register the class or an instance under `graders`; the registry resolves it to
a `BaseGrader` object. Call `grade_case(grader, expected=..., result=...)`.
Every attempt receives a
`GradeResult`: `graded`, `execution_failed`, or `grader_error`. Execution
failure keeps its provider/tool/runtime classification and scores zero;
grader errors have unknown judgment and score. The comparison calls
`grader.aggregate(results)` once per system. Its default counts attempted,
graded, passed, execution failures, and grader errors, and reports accuracy
over all attempts and graded cases. The registered dataset loader validates
selected expected labels before inference; see
[`DATASETS.md`](DATASETS.md).

The built-in `esi.final_acuity_v1` grader compares the expected `acuity` with
the final `final_esi_level` from either SAS or MAS. A matching level scores 1;
a mismatch or invalid prediction scores 0. Its per-case diagnostics report
the expected and predicted levels and whether the prediction was invalid.
Its inherited summary reports both all-attempt and graded-only denominators. Individual
ESI agent decisions are visible in traces but are not graded separately.
