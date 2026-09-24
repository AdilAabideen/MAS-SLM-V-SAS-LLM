"""Test Evaluators test coverage."""

from __future__ import annotations

import pytest

from mas_slm_research.evaluation.legacy_esi1 import ES1AcuityEvaluator
from mas_slm_research.evaluation.legacy_esi2 import ESI2AcuityEvaluator
from mas_slm_research.evaluation.legacy_esi345 import ESI345AcuityEvaluator
from mas_slm_research.evaluation.legacy_single_agent_acuity import SingleAgentAcuityEvaluator
from mas_slm_research.evaluation.legacy_vitals import VitalsUptriageEvaluator


@pytest.mark.unit
def test_ut_evl_001_vitals_evaluator_validates_expected_contract_strictly():
    """Handle ut evl 001 vitals evaluator validates expected contract strictly."""
    # Keep the main step clear.
    evaluator = VitalsUptriageEvaluator()
    with pytest.raises(ValueError):
        evaluator.validate_expected({"bad": True})


@pytest.mark.unit
def test_ut_evl_002_vitals_evaluator_returns_pass_for_matching_boolean():
    """Handle ut evl 002 vitals evaluator returns pass for matching boolean."""
    # Keep the main step clear.
    evaluator = VitalsUptriageEvaluator()
    result = evaluator.evaluate({"recommendation": {"consider_uptriage": True}}, {"recommendation": {"consider_uptriage": True}}, agent_status="succeeded")
    assert result.passed is True


@pytest.mark.unit
def test_ut_evl_004_vitals_evaluator_returns_fail_for_invalid_prediction_shape():
    """Handle ut evl 004 vitals evaluator returns fail for invalid prediction shape."""
    # Keep the main step clear.
    evaluator = VitalsUptriageEvaluator()
    result = evaluator.evaluate({"recommendation": {"consider_uptriage": True}}, {"recommendation": {}}, agent_status="succeeded")
    assert result.metrics_json["invalid_pred"] is True


@pytest.mark.unit
def test_ut_evl_005_vitals_evaluator_aggregate_computes_confusion_counts_correctly():
    """Handle ut evl 005 vitals evaluator aggregate computes confusion counts correctly."""
    # Keep the main step clear.
    evaluator = VitalsUptriageEvaluator()
    results = [
        evaluator.evaluate({"recommendation": {"consider_uptriage": True}}, {"recommendation": {"consider_uptriage": True}}, agent_status="succeeded"),
        evaluator.evaluate({"recommendation": {"consider_uptriage": False}}, {"recommendation": {"consider_uptriage": False}}, agent_status="succeeded"),
    ]
    agg = evaluator.aggregate(results)
    assert agg["tp"] == 1
    assert agg["tn"] == 1


@pytest.mark.unit
def test_ut_evl_006_esi1_evaluator_treats_acuity_one_as_positive_class():
    """Handle ut evl 006 ESI1 evaluator treats acuity one as positive class."""
    # Keep the main step clear.
    evaluator = ES1AcuityEvaluator()
    result = evaluator.evaluate({"acuity": 1}, {"is_esi1": True}, agent_status="succeeded")
    assert result.passed is True


@pytest.mark.unit
def test_ut_evl_008_esi1_evaluator_fails_invalid_prediction():
    """Handle ut evl 008 ESI1 evaluator fails invalid prediction."""
    # Keep the main step clear.
    evaluator = ES1AcuityEvaluator()
    result = evaluator.evaluate({"acuity": 1}, {"is_esi1": "maybe"}, agent_status="succeeded")
    assert result.passed is False


@pytest.mark.unit
def test_ut_evl_009_esi2_evaluator_treats_acuity_two_as_positive_class():
    """Handle ut evl 009 ESI2 evaluator treats acuity two as positive class."""
    # Keep the main step clear.
    evaluator = ESI2AcuityEvaluator()
    result = evaluator.evaluate({"acuity": 2}, {"is_esi2": True}, agent_status="succeeded")
    assert result.passed is True


@pytest.mark.unit
def test_ut_evl_011_esi345_evaluator_returns_pass_for_acuity_and_resource_match():
    """Handle ut evl 011 ESI345 evaluator returns pass for acuity and resource match."""
    # Keep the main step clear.
    evaluator = ESI345AcuityEvaluator()
    result = evaluator.evaluate({"acuity": 3, "resources_used": 2}, {"esi_level": 3, "num_resources": 2}, agent_status="succeeded")
    assert result.passed is True
    assert result.score == 1.0


@pytest.mark.unit
def test_ut_evl_012_esi345_evaluator_returns_warning_for_resource_mismatch():
    """Handle ut evl 012 ESI345 evaluator returns warning for resource mismatch."""
    # Keep the main step clear.
    evaluator = ESI345AcuityEvaluator()
    result = evaluator.evaluate({"acuity": 3, "resources_used": 2}, {"esi_level": 3, "num_resources": 1}, agent_status="succeeded")
    assert result.passed is True
    assert result.score == 0.5


@pytest.mark.unit
def test_ut_evl_014_esi345_evaluator_returns_fail_for_invalid_prediction():
    """Handle ut evl 014 ESI345 evaluator returns fail for invalid prediction."""
    # Keep the main step clear.
    evaluator = ESI345AcuityEvaluator()
    result = evaluator.evaluate({"acuity": 3, "resources_used": 2}, {"esi_level": "x", "num_resources": 2}, agent_status="succeeded")
    assert result.metrics_json["invalid_pred"] is True


@pytest.mark.unit
def test_ut_evl_015_esi345_aggregate_computes_pass_warning_fail_counts_correctly():
    """Handle ut evl 015 ESI345 aggregate computes pass warning fail counts correctly."""
    # Keep the main step clear.
    evaluator = ESI345AcuityEvaluator()
    results = [
        evaluator.evaluate({"acuity": 3, "resources_used": 2}, {"esi_level": 3, "num_resources": 2}, agent_status="succeeded"),
        evaluator.evaluate({"acuity": 3, "resources_used": 2}, {"esi_level": 3, "num_resources": 1}, agent_status="succeeded"),
        evaluator.evaluate({"acuity": 3, "resources_used": 2}, {"esi_level": 4, "num_resources": 2}, agent_status="succeeded"),
    ]
    agg = evaluator.aggregate(results)
    assert agg["pass_count"] == 1
    assert agg["warning_count"] == 1
    assert agg["fail_count"] == 1


@pytest.mark.unit
def test_ut_evl_016_single_agent_evaluator_requires_integer_acuity():
    """Handle ut evl 016 single agent evaluator requires integer acuity."""
    # Keep the main step clear.
    evaluator = SingleAgentAcuityEvaluator()
    with pytest.raises(ValueError):
        evaluator.validate_expected({"acuity": "2"})


@pytest.mark.unit
def test_ut_evl_017_single_agent_evaluator_passes_on_exact_acuity_match():
    """Handle ut evl 017 single agent evaluator passes on exact acuity match."""
    # Keep the main step clear.
    evaluator = SingleAgentAcuityEvaluator()
    result = evaluator.evaluate({"acuity": 2}, {"final_esi_level": 2}, agent_status="succeeded")
    assert result.passed is True
    assert result.score == 1.0


@pytest.mark.unit
def test_ut_evl_018_single_agent_evaluator_fails_on_acuity_mismatch():
    """Handle ut evl 018 single agent evaluator fails on acuity mismatch."""
    # Keep the main step clear.
    evaluator = SingleAgentAcuityEvaluator()
    result = evaluator.evaluate({"acuity": 2}, {"final_esi_level": 3}, agent_status="succeeded")
    assert result.passed is False
    assert result.diff_json["expected_acuity"] == 2
    assert result.diff_json["actual_acuity"] == 3


@pytest.mark.unit
def test_ut_evl_019_single_agent_evaluator_flags_invalid_prediction():
    """Handle ut evl 019 single agent evaluator flags invalid prediction."""
    # Keep the main step clear.
    evaluator = SingleAgentAcuityEvaluator()
    result = evaluator.evaluate({"acuity": 2}, {"summary": "missing level"}, agent_status="succeeded")
    assert result.passed is False
    assert result.metrics_json["invalid_pred"] is True
