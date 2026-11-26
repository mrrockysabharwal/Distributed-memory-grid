"""Tests for authentication on the remaining REST API endpoints."""

import os

TEST_JWT_SECRET = "rest-protection-tests-only-" + ("r" * 40)
os.environ["JWT_SECRET_KEY"] = TEST_JWT_SECRET

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api.main import app
from backend.core.database import Base, get_db


@pytest.fixture
def api_client(tmp_path):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'protected_api.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    test_session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )

    def override_get_db():
        db = test_session_factory()
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
        engine.dispose()


def register_and_authenticate(client):
    registration = client.post(
        "/api/auth/register",
        json={
            "email": "monitoring-user@example.com",
            "password": "StrongPass123",
        },
    )
    assert registration.status_code == 201

    login_response = client.post(
        "/api/auth/login",
        data={
            "username": "monitoring-user@example.com",
            "password": "StrongPass123",
        },
    )
    assert login_response.status_code == 200
    return {
        "Authorization": f"Bearer {login_response.json()['access_token']}"
    }


def test_workers_require_authentication_and_keep_response_shape(api_client):
    unauthenticated = api_client.get("/api/workers")
    assert unauthenticated.status_code == 401

    headers = register_and_authenticate(api_client)
    response = api_client.get("/api/workers", headers=headers)

    assert response.status_code == 200
    assert set(response.json()) == {"workers"}
    assert response.json()["workers"][0] == {
        "worker_id": "worker-1",
        "address": "127.0.0.1:50052",
        "device_type": "cpu",
        "available_memory_mb": 4096,
        "model_partition": "layers 0-2",
        "status": "online",
    }


def test_cluster_status_requires_authentication_and_keeps_response_shape(
    api_client,
    monkeypatch,
):
    unauthenticated = api_client.get("/api/cluster/status")
    assert unauthenticated.status_code == 401

    telemetry = {"cpu_percent": 12.5, "memory_percent": 37.0}
    monkeypatch.setattr(
        "backend.api.routes.get_system_telemetry",
        lambda: telemetry,
    )
    headers = register_and_authenticate(api_client)
    response = api_client.get("/api/cluster/status", headers=headers)

    assert response.status_code == 200
    assert response.json() == {
        "cluster_state": "healthy",
        "active_workers": 2,
        "total_active_requests": 0,
        "master_telemetry": telemetry,
    }


def test_metrics_require_authentication_and_keep_response_shape(api_client):
    unauthenticated = api_client.get("/api/metrics")
    assert unauthenticated.status_code == 401

    headers = register_and_authenticate(api_client)
    response = api_client.get("/api/metrics", headers=headers)

    assert response.status_code == 200
    assert response.json() == {
        "ttft_seconds_avg": 0.12,
        "tokens_per_second_avg": 24.5,
        "total_processed_requests": 0,
        "per_node_ram_usage_mb": {"worker-1": 850, "worker-2": 820},
    }


def test_model_loading_is_disabled_for_authenticated_and_unauthenticated_users(
    api_client,
):
    body = {"model_id": "test-model"}
    detail = (
        "Model loading is disabled because administrative authorization "
        "is not configured."
    )

    unauthenticated = api_client.post("/api/models/load", json=body)
    assert unauthenticated.status_code == 501
    assert unauthenticated.json() == {"detail": detail}

    headers = register_and_authenticate(api_client)
    authenticated = api_client.post(
        "/api/models/load",
        json=body,
        headers=headers,
    )
    assert authenticated.status_code == 501
    assert authenticated.json() == {"detail": detail}


def test_health_endpoint_remains_public_and_minimal(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "DMG Gateway"}
