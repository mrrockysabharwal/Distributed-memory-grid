"""Tests for fresh and existing-database migration paths."""

import os
from datetime import datetime, timezone
from pathlib import Path

os.environ["JWT_SECRET_KEY"] = "migration-tests-only-" + ("m" * 40)

from alembic import command
from alembic.config import Config
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.orm import Session

from backend.models.inference_job import InferenceJob
from backend.models.user import User


BASELINE_REVISION = "20261126_0001"
ALEMBIC_CONFIG_PATH = Path(__file__).parents[2] / "alembic.ini"


def _migration_config(engine, operation, revision):
    config = Config(str(ALEMBIC_CONFIG_PATH))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        operation(config, revision)


def _run_upgrade(config, revision):
    command.upgrade(config, revision)


def _run_downgrade(config, revision):
    command.downgrade(config, revision)


def _run_stamp(config, revision):
    command.stamp(config, revision)


def _sqlite_engine(database_path):
    engine = create_engine(f"sqlite:///{database_path}")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        _ = _connection_record
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    return engine


def _create_legacy_schema(engine, *, include_users=True):
    metadata = MetaData()
    if include_users:
        Table(
            "users",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("email", String(320), nullable=False),
            Column("hashed_password", String, nullable=False),
            Column("is_active", Boolean, nullable=False, default=True),
            Column(
                "created_at",
                DateTime,
                nullable=False,
                default=lambda: datetime.now(timezone.utc),
            ),
            Column(
                "updated_at",
                DateTime,
                nullable=False,
                default=lambda: datetime.now(timezone.utc),
                onupdate=lambda: datetime.now(timezone.utc),
            ),
        )
    Table(
        "inference_jobs",
        metadata,
        Column("job_id", String, primary_key=True),
        Column("model_id", String, nullable=False),
        Column("prompt", Text, nullable=False),
        Column("max_new_tokens", Integer, nullable=False),
        Column("temperature", Float, nullable=False),
        Column("status", String, nullable=False, default="queued"),
        Column("output", Text),
        Column("error", Text),
        Column(
            "created_at",
            DateTime(timezone=True),
            nullable=False,
            default=lambda: datetime.now(timezone.utc),
        ),
        Column(
            "updated_at",
            DateTime(timezone=True),
            nullable=False,
            default=lambda: datetime.now(timezone.utc),
            onupdate=lambda: datetime.now(timezone.utc),
        ),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        if include_users:
            connection.execute(
                text("CREATE UNIQUE INDEX ix_users_email ON users (email)")
            )
            connection.execute(
                text(
                    "INSERT INTO users "
                    "(id, email, hashed_password, is_active, created_at, updated_at) "
                    "VALUES (1, 'legacy@example.com', 'test-hash', 1, "
                    "'2026-01-01 00:00:00', '2026-01-01 00:00:00')"
                )
            )
        connection.execute(
            text(
                "INSERT INTO inference_jobs "
                "(job_id, model_id, prompt, max_new_tokens, temperature, status, "
                "created_at, updated_at) VALUES "
                "('legacy-1', 'legacy-model', 'keep me', 5, 0.7, 'queued', "
                "'2026-01-01 00:00:00', '2026-01-01 00:00:00')"
            )
        )


def _table_schema(engine, table_name):
    with engine.connect() as connection:
        inspector = inspect(connection)
        columns = tuple(
            (
                column["name"],
                str(column["type"]),
                column["nullable"],
                column["default"],
            )
            for column in inspector.get_columns(table_name)
        )
        indexes = tuple(
            sorted(
                (
                    index["name"],
                    tuple(index["column_names"]),
                    index["unique"],
                )
                for index in inspector.get_indexes(table_name)
            )
        )
        unique_constraints = tuple(
            sorted(
                (
                    constraint["name"],
                    tuple(constraint["column_names"]),
                )
                for constraint in inspector.get_unique_constraints(table_name)
            )
        )
        primary_key = tuple(
            inspector.get_pk_constraint(table_name)["constrained_columns"]
        )
        return columns, indexes, unique_constraints, primary_key


def test_fresh_database_upgrade_and_downgrade(tmp_path):
    engine = _sqlite_engine(tmp_path / "fresh.db")
    _migration_config(engine, _run_upgrade, "head")

    with Session(engine) as db:
        user = User(email="fresh@example.com", hashed_password="test-hash")
        db.add(user)
        db.flush()
        job = InferenceJob(
            job_id="fresh-job",
            model_id="fresh-model",
            prompt="fresh prompt",
            max_new_tokens=4,
            temperature=0.5,
            user=user,
        )
        db.add(job)
        db.commit()
        assert user.is_active is True
        assert job.status == "queued"
        assert job.user_id == user.id

    with engine.connect() as connection:
        tables = set(inspect(connection).get_table_names())
        assert {"users", "inference_jobs", "alembic_version"} <= tables
        job_columns = {
            column["name"]
            for column in inspect(connection).get_columns("inference_jobs")
        }
        assert "user_id" in job_columns
        assert {
            "ix_users_email",
            "ix_inference_jobs_user_id",
        } <= {
            index["name"]
            for table in ("users", "inference_jobs")
            for index in inspect(connection).get_indexes(table)
        }

    _migration_config(engine, _run_downgrade, BASELINE_REVISION)

    with engine.connect() as connection:
        job_columns = {
            column["name"]
            for column in inspect(connection).get_columns("inference_jobs")
        }
        assert "user_id" not in job_columns
        assert connection.execute(
            text("SELECT job_id FROM inference_jobs")
        ).scalar_one() == "fresh-job"
        assert connection.execute(
            text("SELECT email FROM users")
        ).scalar_one() == "fresh@example.com"

    _migration_config(engine, _run_downgrade, "base")
    with engine.connect() as connection:
        assert not {"users", "inference_jobs"} & set(
            inspect(connection).get_table_names()
        )
    engine.dispose()


def test_legacy_database_stamp_upgrade_and_downgrade_preserves_data(tmp_path):
    engine = _sqlite_engine(tmp_path / "legacy.db")
    _create_legacy_schema(engine)

    _migration_config(engine, _run_stamp, BASELINE_REVISION)
    _migration_config(engine, _run_upgrade, "head")

    with engine.connect() as connection:
        job_columns = {
            column["name"]
            for column in inspect(connection).get_columns("inference_jobs")
        }
        assert "user_id" in job_columns
        foreign_keys = inspect(connection).get_foreign_keys("inference_jobs")
        assert any(
            foreign_key["constrained_columns"] == ["user_id"]
            and foreign_key["referred_table"] == "users"
            and foreign_key["referred_columns"] == ["id"]
            for foreign_key in foreign_keys
        )
        assert connection.execute(
            text("SELECT job_id, user_id, prompt FROM inference_jobs")
        ).one() == ("legacy-1", None, "keep me")
        assert connection.execute(
            text("SELECT email FROM users WHERE id = 1")
        ).scalar_one() == "legacy@example.com"

    _migration_config(engine, _run_downgrade, BASELINE_REVISION)

    with engine.connect() as connection:
        job_columns = {
            column["name"]
            for column in inspect(connection).get_columns("inference_jobs")
        }
        assert "user_id" not in job_columns
        assert connection.execute(
            text("SELECT job_id, prompt FROM inference_jobs")
        ).one() == ("legacy-1", "keep me")
        assert connection.execute(
            text("SELECT email FROM users WHERE id = 1")
        ).scalar_one() == "legacy@example.com"

    engine.dispose()


def test_legacy_jobs_without_users_creates_baseline_users_schema_and_preserves_data(
    tmp_path,
):
    legacy_engine = _sqlite_engine(tmp_path / "legacy_without_users.db")
    _create_legacy_schema(legacy_engine, include_users=False)

    baseline_engine = _sqlite_engine(tmp_path / "baseline_reference.db")
    _migration_config(baseline_engine, _run_upgrade, BASELINE_REVISION)

    _migration_config(legacy_engine, _run_stamp, BASELINE_REVISION)
    _migration_config(legacy_engine, _run_upgrade, "head")

    assert _table_schema(legacy_engine, "users") == _table_schema(
        baseline_engine,
        "users",
    )
    with legacy_engine.connect() as connection:
        job_columns = {
            column["name"]
            for column in inspect(connection).get_columns("inference_jobs")
        }
        assert "user_id" in job_columns
        assert connection.execute(
            text("SELECT job_id, user_id, prompt FROM inference_jobs")
        ).one() == ("legacy-1", None, "keep me")

    with Session(legacy_engine) as db:
        db.add(User(email="created@example.com", hashed_password="test-hash"))
        db.commit()

    _migration_config(legacy_engine, _run_downgrade, BASELINE_REVISION)

    with legacy_engine.connect() as connection:
        job_columns = {
            column["name"]
            for column in inspect(connection).get_columns("inference_jobs")
        }
        assert "user_id" not in job_columns
        assert connection.execute(
            text("SELECT job_id, prompt FROM inference_jobs")
        ).one() == ("legacy-1", "keep me")
        assert connection.execute(
            text("SELECT email FROM users")
        ).scalar_one() == "created@example.com"

    legacy_engine.dispose()
    baseline_engine.dispose()
