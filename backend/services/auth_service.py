"""Authentication operations for application users."""

from sqlalchemy.orm import Session

from backend.core.security import verify_password
from backend.models.user import User
from backend.services.user_service import get_user_by_email


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(db, email)
    if user is None or not user.is_active:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
