"""Tests for persistent inference job API behavior."""

from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api.routes import router
from backend.core import database
from backend.core.database import Base, get_db


app = FastAPI()
app.include_router(router, prefix="/api")


@pytest.fixture
def api_client(tmp_path):
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'inference_jobs.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=test_engine)
    test_session_local = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )

    def override_get_db():
        db = test_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_db, None)
        test_engine.dispose()


def test_create_job_persists_with_queued_status_and_existing_response_fields(
    api_client,
):
    response = api_client.post(
        "/api/inference/generate",
        json={
            "model_id": "test-model",
            "prompt": "Hello",
            "max_new_tokens": 12,
            "temperature": 0.5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    UUID(payload["job_id"])
    assert payload == {
        "job_id": payload["job_id"],
        "status": "queued",
        "model_id": "test-model",
        "prompt": "Hello",
        "max_new_tokens": 12,
        "temperature": 0.5,
        "websocket_url": f"/ws/inference/{payload['job_id']}",
    }

    stored_response = api_client.get(f"/api/inference/jobs/{payload['job_id']}")
    assert stored_response.status_code == 200
    stored_job = stored_response.json()
    assert stored_job["job_id"] == payload["job_id"]
    assert stored_job["status"] == "queued"
    assert stored_job["model_id"] == "test-model"
    assert stored_job["prompt"] == "Hello"
    assert stored_job["max_new_tokens"] == 12
    assert stored_job["temperature"] == 0.5
    assert stored_job["output"] is None
    assert stored_job["error"] is None


def test_generate_retains_supplied_job_id(api_client):
    response = api_client.post(
        "/api/inference/generate",
        json={"request_id": "client-job-123", "prompt": "Hello"},
    )

    assert response.status_code == 200
    assert response.json()["job_id"] == "client-job-123"


def test_get_nonexistent_job_returns_404(api_client):
    response = api_client.get("/api/inference/jobs/missing-job")

    assert response.status_code == 404
    assert response.json() == {"detail": "Inference job not found"}


def test_get_db_closes_session_after_dependency_finishes(monkeypatch):
    class FakeSession:
        closed = False

        def close(self):
            self.closed = True

    fake_session = FakeSession()
    monkeypatch.setattr(database, "SessionLocal", lambda: fake_session)
    dependency = get_db()

    assert next(dependency) is fake_session
    with pytest.raises(StopIteration):
        next(dependency)

    assert fake_session.closed
