"""Focused tests for the user model and password security foundation."""

from datetime import datetime, timedelta, timezone
from time import sleep

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.core.security import hash_password, verify_password
from backend.core.database import Base
from backend.models.user import User
from backend.schemas.user import UserCreate, UserResponse
from backend.services.user_service import (
    create_user,
    get_user_by_email,
    get_user_by_id,
)


@pytest.fixture
def user_db(tmp_path):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'users.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def test_user_model_defaults(user_db):
    user = User(
        email="defaults@example.com",
        hashed_password=hash_password("StrongPass123"),
    )
    user_db.add(user)
    user_db.commit()
    user_db.refresh(user)

    assert user.id is not None
    assert user.email == "defaults@example.com"
    assert user.is_active is True
    assert user.created_at is not None
    assert user.updated_at is not None
    assert user.created_at.tzinfo is not None
    assert user.updated_at.tzinfo is not None
    assert user.created_at.utcoffset() == timedelta(0)
    assert user.updated_at.utcoffset() == timedelta(0)
    assert user.created_at.tzinfo is timezone.utc
    assert user.updated_at.tzinfo is timezone.utc


def test_updated_at_changes_after_record_update(user_db):
    user = create_user(user_db, "updated@example.com", "StrongPass123")
    original_updated_at = user.updated_at

    sleep(0.01)
    user.is_active = False
    user_db.commit()
    user_db.refresh(user)

    assert user.updated_at.tzinfo is timezone.utc
    assert user.updated_at.utcoffset() == timedelta(0)
    assert user.updated_at > original_updated_at


def test_password_hashing_produces_argon2_hash():
    password = "StrongPass123"
    hashed_password = hash_password(password)

    assert hashed_password != password
    assert hashed_password.startswith("$argon2")


def test_password_verification_succeeds():
    hashed_password = hash_password("StrongPass123")

    assert verify_password("StrongPass123", hashed_password) is True


def test_incorrect_password_is_rejected():
    hashed_password = hash_password("StrongPass123")

    assert verify_password("WrongPass123", hashed_password) is False


def test_duplicate_email_registration_is_rejected(user_db):
    create_user(user_db, "User@example.com", "StrongPass123")

    with pytest.raises(ValueError, match="Email already registered"):
        create_user(user_db, "user@example.com", "AnotherPass123")


def test_email_addresses_are_normalized(user_db):
    user = create_user(user_db, "  User.Name+Test@Example.COM  ", "StrongPass123")

    assert user.email == "user.name+test@example.com"
    assert get_user_by_email(user_db, " USER.NAME+TEST@EXAMPLE.COM ") == user
    assert get_user_by_id(user_db, user.id) == user


def test_user_create_validates_email_and_password():
    with pytest.raises(ValueError):
        UserCreate(email="not-an-email", password="StrongPass123")

    with pytest.raises(ValueError):
        UserCreate(email="user@example.com", password="short")

    schema = UserCreate(email="user@example.com", password="StrongPass123")
    assert schema.email == "user@example.com"


def test_user_response_does_not_expose_hashed_password():
    user = User(
        id=7,
        email="response@example.com",
        hashed_password=hash_password("StrongPass123"),
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )
    response = UserResponse.model_validate(user)

    assert "hashed_password" not in response.model_dump()
    assert response.model_dump()["email"] == "response@example.com"
