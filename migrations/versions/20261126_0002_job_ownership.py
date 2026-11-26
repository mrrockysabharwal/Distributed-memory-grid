"""Add nullable user ownership to inference jobs.

Revision ID: 20261126_0002
Revises: 20261126_0001
Create Date: 2026-11-26
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20261126_0002"
down_revision = "20261126_0001"
branch_labels = None
depends_on = None


def _utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def _create_users_table() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), default=True, nullable=False),
        sa.Column("created_at", sa.DateTime(), default=_utc_now, nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            default=_utc_now,
            onupdate=_utc_now,
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(inspect(bind).get_table_names())
    if "inference_jobs" not in tables:
        raise RuntimeError(
            "The ownership migration requires the baseline inference_jobs table"
        )
    if "users" not in tables:
        _create_users_table()

    columns = {
        column["name"] for column in inspect(bind).get_columns("inference_jobs")
    }
    if "user_id" in columns:
        raise RuntimeError(
            "inference_jobs.user_id already exists; verify the database schema "
            "before applying this migration"
        )

    if bind.dialect.name == "sqlite":
        op.execute(
            "ALTER TABLE inference_jobs "
            "ADD COLUMN user_id INTEGER "
            "CONSTRAINT fk_inference_jobs_user_id_users REFERENCES users (id)"
        )
    else:
        op.add_column(
            "inference_jobs",
            sa.Column(
                "user_id",
                sa.Integer(),
                sa.ForeignKey(
                    "users.id",
                    name="fk_inference_jobs_user_id_users",
                ),
                nullable=True,
            ),
        )
    op.create_index("ix_inference_jobs_user_id", "inference_jobs", ["user_id"])


def downgrade() -> None:
    bind = op.get_bind()
    columns = {
        column["name"] for column in inspect(bind).get_columns("inference_jobs")
    }
    if "user_id" not in columns:
        return

    op.drop_index("ix_inference_jobs_user_id", table_name="inference_jobs")
    if bind.dialect.name == "sqlite":
        op.execute("ALTER TABLE inference_jobs DROP COLUMN user_id")
    else:
        op.drop_constraint(
            "fk_inference_jobs_user_id_users",
            "inference_jobs",
            type_="foreignkey",
        )
        op.drop_column("inference_jobs", "user_id")
