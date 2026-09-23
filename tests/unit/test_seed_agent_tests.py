"""Test Seed Agent Tests test coverage."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from tests.backend_fixtures import db_engine, db_session, session_factory, sqlite_url  # noqa: F401

from app.models.agent_test_case import AgentTestCase
from app.seed_agent_tests import (
    SINGLE_AGENT_NAME,
    _load_single_agent_seed_rows,
    ensure_seed_single_agent_test_cases,
)


@pytest.mark.unit
def test_ut_seed_001_single_agent_seed_rows_normalize_known_bad_keys():
    """Handle ut seed 001 single agent seed rows normalize known bad keys."""
    # Keep the main step clear.
    rows = _load_single_agent_seed_rows()

    assert rows
    assert all(isinstance(row["expected_json"]["acuity"], int) for row in rows)

    medication_refill_case = next(
        row for row in rows if row["input_json"].get("chiefcomplaint") == "Medication refill request"
    )
    assert medication_refill_case["input_json"]["heartrate"] == 74.0

    unresponsive_case = next(
        row for row in rows if row["input_json"].get("chiefcomplaint") == "Unresponsive"
    )
    assert unresponsive_case["input_json"]["arrival_transport"] == "AMBULANCE"


@pytest.mark.unit
def test_ut_seed_002_single_agent_seeder_resets_cases(db_session):
    """Handle ut seed 002 single agent seeder resets cases."""
    # Keep the main step clear.
    ensure_seed_single_agent_test_cases(db_session)
    first_count = db_session.execute(
        select(AgentTestCase).where(AgentTestCase.agent_name == SINGLE_AGENT_NAME)
    ).scalars().all()

    assert len(first_count) == len(_load_single_agent_seed_rows())

    ensure_seed_single_agent_test_cases(db_session)
    second_count = db_session.execute(
        select(AgentTestCase).where(AgentTestCase.agent_name == SINGLE_AGENT_NAME)
    ).scalars().all()

    assert len(second_count) == len(first_count)
    assert {row.name for row in second_count} == {row.name for row in first_count}
