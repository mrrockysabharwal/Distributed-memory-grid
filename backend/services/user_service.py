"""Persistence operations for application users."""

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.core.security import hash_password
from backend.models.user import User


def normalize_email(email: str) -> str:
    return email.strip().lower()


def get_user_by_email(db: Session, email: str) -> User | None:
    normalized_email = normalize_email(email)
    return db.query(User).filter(User.email == normalized_email).first()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def create_user(db: Session, email: str, password: str) -> User:
    normalized_email = normalize_email(email)

    if get_user_by_email(db, normalized_email) is not None:
        raise ValueError("Email already registered")

    user = User(
        email=normalized_email,
        hashed_password=hash_password(password),
        is_active=True,
    )
    db.add(user)

    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise ValueError("Email already registered") from error

    db.refresh(user)
    return user
