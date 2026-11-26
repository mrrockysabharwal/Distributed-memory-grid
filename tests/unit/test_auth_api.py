"""Authentication API tests using isolated SQLite databases."""

import os
from datetime import datetime, timedelta, timezone

TEST_JWT_SECRET = "auth-api-test-secret-" + ("z" * 40)
os.environ["JWT_SECRET_KEY"] = TEST_JWT_SECRET

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api.auth import get_current_user
from backend.api.routes import router
from backend.core.database import Base, get_db
from backend.models.user import User
from backend.schemas.user import UserResponse


app = FastAPI()
app.include_router(router, prefix="/api")


@app.get("/test/current-user", response_model=UserResponse)
def resolve_current_user(
    current_user: User = Depends(get_current_user),
) -> User:
    return current_user


@pytest.fixture
def api_client(tmp_path):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'auth.db').as_posix()}",
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
            yield client, test_session_factory
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()


def register(api_client, email="user@example.com", password="StrongPass123"):
    client, _ = api_client
    return client.post(
        "/api/auth/register",
        json={"email": email, "password": password},
    )


def login(client, email="user@example.com", password="StrongPass123"):
    return client.post(
        "/api/auth/login",
        data={"username": email, "password": password},
    )


def test_successful_registration_returns_safe_user_response(api_client):
    response = register(api_client)

    assert response.status_code == 201
    payload = response.json()
    assert payload["email"] == "user@example.com"
    assert payload["is_active"] is True
    assert "hashed_password" not in payload
    assert "password" not in payload
    assert "StrongPass123" not in response.text


def test_duplicate_registration_returns_conflict(api_client):
    assert register(api_client).status_code == 201

    response = register(api_client, email="USER@example.com")

    assert response.status_code == 409
    assert response.json() == {"detail": "Email already registered"}


def test_successful_login_returns_bearer_token(api_client):
    register(api_client)

    response = login(api_client[0])

    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert response.json()["access_token"]
    assert "hashed_password" not in response.text


@pytest.mark.parametrize(
    ("email", "password"),
    [
        ("user@example.com", "WrongPass123"),
        ("unknown@example.com", "StrongPass123"),
    ],
)
def test_invalid_login_credentials_return_consistent_401(
    api_client,
    email,
    password,
):
    register(api_client)

    response = login(api_client[0], email=email, password=password)

    assert response.status_code == 401
    assert response.json() == {"detail": "Could not validate credentials"}
    assert response.headers["www-authenticate"] == "Bearer"


def test_valid_jwt_resolves_current_user(api_client):
    register(api_client)
    token_response = login(api_client[0])
    token = token_response.json()["access_token"]

    response = api_client[0].get(
        "/test/current-user",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["email"] == "user@example.com"
    assert "hashed_password" not in response.json()


@pytest.mark.parametrize(
    "token_factory",
    [
        lambda user_id: jwt.encode(
            {
                "sub": str(user_id),
                "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
            },
            TEST_JWT_SECRET,
            algorithm="HS256",
        ),
        lambda user_id: jwt.encode(
            {
                "sub": str(user_id),
                "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
            },
            "different-test-secret-" + ("a" * 40),
            algorithm="HS256",
        ),
        lambda user_id: f"not.a.jwt.{user_id}",
    ],
)
def test_expired_invalid_signature_and_malformed_tokens_return_401(
    api_client,
    token_factory,
):
    register(api_client)
    client, session_factory = api_client
    with session_factory() as db:
        user = db.query(User).filter(User.email == "user@example.com").one()
        token = token_factory(user.id)

    response = client.get(
        "/test/current-user",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Could not validate credentials"}


def test_missing_bearer_token_returns_401(api_client):
    response = api_client[0].get("/test/current-user")

    assert response.status_code == 401
    assert response.json() == {"detail": "Could not validate credentials"}
    assert response.headers["www-authenticate"] == "Bearer"


def test_token_for_nonexistent_user_returns_401(api_client):
    token = jwt.encode(
        {
            "sub": "999999",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        TEST_JWT_SECRET,
        algorithm="HS256",
    )

    response = api_client[0].get(
        "/test/current-user",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Could not validate credentials"}
