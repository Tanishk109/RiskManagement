"""Add Fraud Pulse operational run and alert tables.

Revision ID: 0003_fraud_pulse
Revises: 0002_chargeback_evidence
Create Date: 2026-08-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_fraud_pulse"
down_revision: str | None = "0002_chargeback_evidence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "fraud_pulse_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("detector_method", sa.String(length=32), nullable=False),
        sa.Column("metric", sa.String(length=32), nullable=False),
        sa.Column("window_seconds", sa.Integer(), nullable=False),
        sa.Column("baseline_windows", sa.Integer(), nullable=False),
        sa.Column("sensitivity", sa.Float(), nullable=False),
        sa.Column("ewma_alpha", sa.Float(), nullable=False),
        sa.Column("percent_deviation_threshold", sa.Float(), nullable=False),
        sa.Column("rows_scored", sa.Integer(), nullable=False),
        sa.Column("window_count", sa.Integer(), nullable=False),
        sa.Column("alert_count", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(length=120), nullable=False),
        sa.Column("review_threshold", sa.Float(), nullable=False),
        sa.Column("block_threshold", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("source IN ('VALIDATION_REPLAY', 'MERCHANT_UPLOAD')", name="ck_fraud_pulse_runs_source"),
        sa.CheckConstraint(
            "detector_method IN ('rolling_zscore', 'ewma', 'percent_deviation')",
            name="ck_fraud_pulse_runs_method",
        ),
        sa.CheckConstraint(
            "metric IN ('transaction_count', 'mean_risk_score', 'high_risk_count', 'high_risk_amount')",
            name="ck_fraud_pulse_runs_metric",
        ),
    )
    op.create_index("ix_fraud_pulse_runs_source", "fraud_pulse_runs", ["source"])

    op.create_table(
        "fraud_pulse_alerts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("window_start", sa.BigInteger(), nullable=False),
        sa.Column("window_end", sa.BigInteger(), nullable=False),
        sa.Column("current_value", sa.Float(), nullable=False),
        sa.Column("baseline_value", sa.Float(), nullable=False),
        sa.Column("absolute_change", sa.Float(), nullable=False),
        sa.Column("percent_deviation", sa.Float(), nullable=True),
        sa.Column("detector_score", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["fraud_pulse_runs.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_fraud_pulse_alerts_run_id", "fraud_pulse_alerts", ["run_id"])
    op.create_index("ix_fraud_pulse_alerts_window_start", "fraud_pulse_alerts", ["window_start"])


def downgrade() -> None:
    op.drop_table("fraud_pulse_alerts")
    op.drop_table("fraud_pulse_runs")
