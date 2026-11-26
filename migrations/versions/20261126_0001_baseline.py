"""Create the original users and inference jobs schema.

Revision ID: 20261126_0001
Revises:
Create Date: 2026-11-26
"""

from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa


revision = "20261126_0001"
down_revision = None
branch_labels = None
depends_on = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def upgrade() -> None:
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

    op.create_table(
        "inference_jobs",
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("max_new_tokens", sa.Integer(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("status", sa.String(), default="queued", nullable=False),
        sa.Column("output", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            default=_utc_now,
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            default=_utc_now,
            onupdate=_utc_now,
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("job_id"),
    )


def downgrade() -> None:
    op.drop_table("inference_jobs")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
