"""Add optional return-risk runtime prediction records.

Revision ID: 0005_return_predictions
Revises: 0004_abuse_rings
Create Date: 2026-08-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_return_predictions"
down_revision: str | None = "0004_abuse_rings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "return_predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("order_value", sa.Float(), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("unique_stock_count", sa.Integer(), nullable=False),
        sa.Column("country", sa.String(length=120), nullable=False),
        sa.Column("stock_code", sa.String(length=120), nullable=False),
        sa.Column("prior_order_count", sa.Integer(), nullable=False),
        sa.Column("prior_cancellation_rate", sa.Float(), nullable=False),
        sa.Column("prior_average_order_value", sa.Float(), nullable=False),
        sa.Column("order_hour", sa.Integer(), nullable=False),
        sa.Column("order_day_of_week", sa.Integer(), nullable=False),
        sa.Column("return_risk_probability", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=12), nullable=False),
        sa.Column("model_version", sa.String(length=120), nullable=False),
        sa.Column("automatic_rejection", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "return_risk_probability >= 0", name="ck_return_predictions_probability_low"
        ),
        sa.CheckConstraint(
            "return_risk_probability <= 1", name="ck_return_predictions_probability_high"
        ),
        sa.CheckConstraint(
            "risk_level IN ('LOW', 'MEDIUM', 'HIGH')", name="ck_return_predictions_level"
        ),
    )
    op.create_index("ix_return_predictions_risk_level", "return_predictions", ["risk_level"])
    op.create_index("ix_return_predictions_model_version", "return_predictions", ["model_version"])


def downgrade() -> None:
    op.drop_table("return_predictions")
