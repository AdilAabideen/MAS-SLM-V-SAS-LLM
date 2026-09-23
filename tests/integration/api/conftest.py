"""HTTP contract fixtures scoped to API integration tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.api import api_router
from app.database import get_db
from tests.backend_fixtures import db_engine, db_session, session_factory, sqlite_url  # noqa: F401


@pytest.fixture
def test_app(session_factory) -> FastAPI:
    app = FastAPI()
    app.include_router(api_router, prefix="/api")

    @app.get("/")
    def root():
        return {"message": "Welcome to Emergency Severity Index Multi Agent V Monolithic Agent System"}

    @app.get("/health")
    def health_check():
        return {"status": "healthy"}

    def _override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    return app


@pytest.fixture
def client(test_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(test_app) as test_client:
        yield test_client
