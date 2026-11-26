"""Tests for authenticated inference job API behavior."""

import os
from uuid import UUID

TEST_JWT_SECRET = "inference-tests-only-" + ("j" * 40)
os.environ["JWT_SECRET_KEY"] = TEST_JWT_SECRET

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api.main import app as main_app
from backend.api.routes import router
from backend.core import database
from backend.core.database import Base, get_db
from backend.models.inference_job import InferenceJob


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
        autocommit=False,
        autoflush=False,
        bind=test_engine,
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
            yield client, test_session_local
    finally:
        app.dependency_overrides.pop(get_db, None)
        test_engine.dispose()


def register_and_authenticate(client, email="owner@example.com"):
    registration = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123"},
    )
    assert registration.status_code == 201

    login_response = client.post(
        "/api/auth/login",
        data={"username": email, "password": "StrongPass123"},
    )
    assert login_response.status_code == 200
    return (
        {"Authorization": f"Bearer {login_response.json()['access_token']}"},
        registration.json()["id"],
    )


def test_authenticated_creation_preserves_response_and_persists_owner(api_client):
    client, session_factory = api_client
    headers, user_id = register_and_authenticate(client)

    response = client.post(
        "/api/inference/generate",
        json={
            "model_id": "test-model",
            "prompt": "Hello",
            "max_new_tokens": 12,
            "temperature": 0.5,
        },
        headers=headers,
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
        "websocket_url": f"/api/ws/inference/{payload['job_id']}",
    }

    with session_factory() as db:
        job = db.get(InferenceJob, payload["job_id"])
        assert job is not None
        assert job.user_id == user_id

    stored_response = client.get(
        f"/api/inference/jobs/{payload['job_id']}",
        headers=headers,
    )
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


def test_websocket_route_is_mounted_under_api_prefix():
    assert str(
        main_app.url_path_for("websocket_inference_stream", job_id="test-job")
    ) == "/api/ws/inference/test-job"


def test_generate_retains_supplied_job_id(api_client):
    client, _ = api_client
    headers, _ = register_and_authenticate(client)

    response = client.post(
        "/api/inference/generate",
        json={"request_id": "client-job-123", "prompt": "Hello"},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["job_id"] == "client-job-123"


def test_another_user_cannot_retrieve_job(api_client):
    client, _ = api_client
    owner_headers, _ = register_and_authenticate(client, "owner@example.com")
    other_headers, _ = register_and_authenticate(client, "other@example.com")
    create_response = client.post(
        "/api/inference/generate",
        json={"prompt": "Private prompt"},
        headers=owner_headers,
    )

    response = client.get(
        f"/api/inference/jobs/{create_response.json()['job_id']}",
        headers=other_headers,
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Inference job not found"}


def test_unauthenticated_creation_is_rejected(api_client):
    response = api_client[0].post(
        "/api/inference/generate",
        json={"prompt": "Hello"},
    )

    assert response.status_code == 401


def test_unauthenticated_retrieval_is_rejected(api_client):
    client, _ = api_client
    owner_headers, _ = register_and_authenticate(client)
    create_response = client.post(
        "/api/inference/generate",
        json={"prompt": "Hello"},
        headers=owner_headers,
    )

    response = client.get(
        f"/api/inference/jobs/{create_response.json()['job_id']}"
    )

    assert response.status_code == 401


def test_nonexistent_job_returns_404_for_authenticated_user(api_client):
    client, _ = api_client
    headers, _ = register_and_authenticate(client)

    response = client.get(
        "/api/inference/jobs/missing-job",
        headers=headers,
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Inference job not found"}


def test_legacy_job_without_owner_is_not_visible_to_users(api_client):
    client, session_factory = api_client
    headers, _ = register_and_authenticate(client)
    with session_factory() as db:
        legacy_job = InferenceJob(
            job_id="legacy-job",
            model_id="old-model",
            prompt="Legacy prompt",
            max_new_tokens=8,
            temperature=0.4,
            status="queued",
            user_id=None,
        )
        db.add(legacy_job)
        db.commit()

    response = client.get(
        "/api/inference/jobs/legacy-job",
        headers=headers,
    )

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
