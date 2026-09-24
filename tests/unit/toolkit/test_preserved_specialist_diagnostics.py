"""Legacy specialist rules remain available as diagnostics after API retirement."""

from __future__ import annotations

from mas_slm_research.evaluation.diagnostics import evaluate_legacy_diagnostic
from mas_slm_research.evaluation.legacy_doctor import DoctorAlwaysPassEvaluator
from mas_slm_research.evaluation.legacy_esi1 import ES1AcuityEvaluator
from mas_slm_research.evaluation.legacy_esi2 import ESI2AcuityEvaluator
from mas_slm_research.evaluation.legacy_esi345 import ESI345AcuityEvaluator
from mas_slm_research.evaluation.legacy_vitals import VitalsUptriageEvaluator


def test_preserved_specialist_rules_keep_distinct_outcomes() -> None:
    first = ES1AcuityEvaluator().evaluate(
        {"acuity": 1}, {"is_esi1": True}, agent_status="succeeded"
    )
    second = ESI2AcuityEvaluator().evaluate(
        {"acuity": 2}, {"is_esi2": True}, agent_status="succeeded"
    )
    resource_warning = ESI345AcuityEvaluator().evaluate(
        {"acuity": 3, "resources_used": 2}, {"esi_level": 3, "num_resources": 1},
        agent_status="succeeded",
    )
    vitals = VitalsUptriageEvaluator().evaluate(
        {"recommendation": {"consider_uptriage": True}},
        {"recommendation": {"consider_uptriage": True}}, agent_status="succeeded",
    )
    assert first.passed is second.passed is vitals.passed is True
    assert resource_warning.passed is True and resource_warning.score == 0.5


def test_doctor_always_pass_remains_explicitly_placeholder_only() -> None:
    diagnostic = evaluate_legacy_diagnostic(
        DoctorAlwaysPassEvaluator(), agent_name="doctor_agent",
        expected={"acuity": 1}, actual=None, agent_status="failed",
    )
    assert diagnostic.passed is True
    assert diagnostic.placeholder is True
    assert diagnostic.metrics["always_pass"] is True
