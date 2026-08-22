"""Add Abuse-Ring Sentinel operational run and cluster tables.

Revision ID: 0004_abuse_rings
Revises: 0003_fraud_pulse
Create Date: 2026-08-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_abuse_rings"
down_revision: str | None = "0003_fraud_pulse"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "abuse_graph_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("data_partition", sa.String(length=24), nullable=False),
        sa.Column("min_attribute_degree", sa.Integer(), nullable=False),
        sa.Column("max_attribute_degree", sa.Integer(), nullable=False),
        sa.Column("minimum_cluster_transactions", sa.Integer(), nullable=False),
        sa.Column("minimum_high_risk_transactions", sa.Integer(), nullable=False),
        sa.Column("minimum_high_risk_share", sa.Float(), nullable=False),
        sa.Column("transaction_nodes", sa.Integer(), nullable=False),
        sa.Column("shared_attribute_nodes", sa.Integer(), nullable=False),
        sa.Column("edge_count", sa.Integer(), nullable=False),
        sa.Column("suspicious_cluster_count", sa.Integer(), nullable=False),
        sa.Column("suppressed_attribute_values", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(length=120), nullable=False),
        sa.Column("confirmed_fraud_ring_claimed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("data_partition = 'validation'", name="ck_abuse_graph_runs_partition"),
    )
    op.create_table(
        "abuse_cluster_records",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("cluster_key", sa.String(length=64), nullable=False),
        sa.Column("transaction_count", sa.Integer(), nullable=False),
        sa.Column("shared_attribute_count", sa.Integer(), nullable=False),
        sa.Column("edge_count", sa.Integer(), nullable=False),
        sa.Column("high_risk_count", sa.Integer(), nullable=False),
        sa.Column("high_risk_share", sa.Float(), nullable=False),
        sa.Column("average_risk_score", sa.Float(), nullable=False),
        sa.Column("maximum_risk_score", sa.Float(), nullable=False),
        sa.Column("total_transaction_amount", sa.Float(), nullable=False),
        sa.Column("high_risk_amount", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["abuse_graph_runs.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_abuse_cluster_records_run_id", "abuse_cluster_records", ["run_id"])
    op.create_index("ix_abuse_cluster_records_cluster_key", "abuse_cluster_records", ["cluster_key"])


def downgrade() -> None:
    op.drop_table("abuse_cluster_records")
    op.drop_table("abuse_graph_runs")
