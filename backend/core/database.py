"""SQLAlchemy engine and session configuration."""

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.core.config import settings


_database_url = make_url(settings.database_url)
_connect_args = (
    {"check_same_thread": False}
    if _database_url.get_backend_name() == "sqlite"
    else {}
)

if (
    _database_url.get_backend_name() == "sqlite"
    and _database_url.database
    and _database_url.database != ":memory:"
    and not _database_url.database.startswith("file:")
):
    Path(_database_url.database).expanduser().parent.mkdir(
        parents=True, exist_ok=True
    )

engine = create_engine(settings.database_url, connect_args=_connect_args)
if _database_url.get_backend_name() == "sqlite":
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(
        dbapi_connection,
        _connection_record,
    ) -> None:
        _ = _connection_record
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
