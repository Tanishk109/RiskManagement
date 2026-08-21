"""Create the normalized MerchantShield operational schema.

Revision ID: 0001_operational_schema
Revises:
Create Date: 2026-08-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_operational_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_name", sa.String(length=120), nullable=False),
        sa.Column("model_version", sa.String(length=120), nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_set", sa.String(length=120), nullable=False),
        sa.Column("evaluation_status", sa.String(length=20), nullable=False),
        sa.Column("evaluation_split", sa.String(length=32), nullable=True),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("test_transaction_count", sa.Integer(), nullable=True),
        sa.Column("fraud_count", sa.Integer(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("f1", sa.Float(), nullable=True),
        sa.Column("average_precision", sa.Float(), nullable=True),
        sa.Column("roc_auc", sa.Float(), nullable=True),
        sa.Column("brier_score", sa.Float(), nullable=True),
        sa.Column("true_positives", sa.Integer(), nullable=True),
        sa.Column("false_positives", sa.Integer(), nullable=True),
        sa.Column("true_negatives", sa.Integer(), nullable=True),
        sa.Column("false_negatives", sa.Integer(), nullable=True),
        sa.Column("approve_count", sa.Integer(), nullable=True),
        sa.Column("review_count", sa.Integer(), nullable=True),
        sa.Column("block_count", sa.Integer(), nullable=True),
        sa.Column("false_positive_estimated_cost", sa.Numeric(18, 4), nullable=True),
        sa.Column("false_negative_estimated_cost", sa.Numeric(18, 4), nullable=True),
        sa.Column("review_cost", sa.Numeric(18, 4), nullable=True),
        sa.Column("total_estimated_cost", sa.Numeric(18, 4), nullable=True),
        sa.Column("active_rule_count", sa.Integer(), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "evaluation_status IN ('NOT_EVALUATED', 'COMPLETE')",
            name="ck_model_runs_evaluation_status",
        ),
    )
    op.create_index("ix_model_runs_model_version", "model_runs", ["model_version"], unique=True)
    op.create_index("ix_model_runs_evaluation_status", "model_runs", ["evaluation_status"])

    op.create_table(
        "cost_configs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("config_key", sa.String(length=80), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("fraud_loss_fraction", sa.Float(), nullable=False),
        sa.Column("chargeback_fixed_cost", sa.Numeric(18, 4), nullable=False),
        sa.Column("legitimate_margin_rate", sa.Float(), nullable=False),
        sa.Column("false_positive_fixed_cost", sa.Numeric(18, 4), nullable=False),
        sa.Column("manual_review_cost", sa.Numeric(18, 4), nullable=False),
        sa.Column("review_fraud_catch_rate", sa.Float(), nullable=False),
        sa.Column("review_legitimate_approval_rate", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("fraud_loss_fraction >= 0", name="ck_cost_configs_fraud_loss_fraction"),
        sa.CheckConstraint("chargeback_fixed_cost >= 0", name="ck_cost_configs_chargeback_fixed_cost"),
        sa.CheckConstraint(
            "legitimate_margin_rate BETWEEN 0 AND 1",
            name="ck_cost_configs_legitimate_margin_rate",
        ),
        sa.CheckConstraint(
            "false_positive_fixed_cost >= 0",
            name="ck_cost_configs_false_positive_fixed_cost",
        ),
        sa.CheckConstraint("manual_review_cost >= 0", name="ck_cost_configs_manual_review_cost"),
        sa.CheckConstraint(
            "review_fraud_catch_rate BETWEEN 0 AND 1",
            name="ck_cost_configs_review_fraud_catch_rate",
        ),
        sa.CheckConstraint(
            "review_legitimate_approval_rate BETWEEN 0 AND 1",
            name="ck_cost_configs_review_legitimate_approval_rate",
        ),
    )
    op.create_index("ix_cost_configs_config_key", "cost_configs", ["config_key"], unique=True)

    op.create_table(
        "threshold_configs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("config_key", sa.String(length=120), nullable=False),
        sa.Column("model_run_id", sa.Integer(), nullable=False),
        sa.Column("cost_config_id", sa.Integer(), nullable=True),
        sa.Column("review_threshold", sa.Float(), nullable=False),
        sa.Column("block_threshold", sa.Float(), nullable=False),
        sa.Column("selection_split", sa.String(length=32), nullable=False),
        sa.Column("objective", sa.String(length=180), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "review_threshold >= 0 AND review_threshold <= 1",
            name="ck_threshold_configs_review_range",
        ),
        sa.CheckConstraint(
            "block_threshold >= 0 AND block_threshold <= 1",
            name="ck_threshold_configs_block_range",
        ),
        sa.CheckConstraint("review_threshold < block_threshold", name="ck_threshold_configs_order"),
        sa.ForeignKeyConstraint(["cost_config_id"], ["cost_configs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["model_run_id"], ["model_runs.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_threshold_configs_config_key", "threshold_configs", ["config_key"], unique=True)
    op.create_index("ix_threshold_configs_model_run_id", "threshold_configs", ["model_run_id"])
    op.create_index(
        "ix_threshold_configs_model_active",
        "threshold_configs",
        ["model_run_id", "is_active"],
    )

    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("transaction_id", sa.String(length=80), nullable=False),
        sa.Column("transaction_dt", sa.BigInteger(), nullable=False),
        sa.Column("amount", sa.Numeric(18, 4), nullable=False),
        sa.Column("actual_label", sa.SmallInteger(), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("model_run_id", sa.Integer(), nullable=False),
        sa.Column("threshold_config_id", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "actual_label IS NULL OR actual_label IN (0, 1)",
            name="ck_transactions_actual_label",
        ),
        sa.CheckConstraint("amount >= 0", name="ck_transactions_amount"),
        sa.CheckConstraint(
            "decision IN ('APPROVE', 'REVIEW', 'BLOCK')",
            name="ck_transactions_decision",
        ),
        sa.CheckConstraint("risk_score >= 0 AND risk_score <= 1", name="ck_transactions_risk_score"),
        sa.ForeignKeyConstraint(["model_run_id"], ["model_runs.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["threshold_config_id"],
            ["threshold_configs.id"],
            ondelete="RESTRICT",
        ),
    )
    op.create_index("ix_transactions_transaction_id", "transactions", ["transaction_id"], unique=True)
    op.create_index("ix_transactions_transaction_dt", "transactions", ["transaction_dt"])
    op.create_index("ix_transactions_actual_label", "transactions", ["actual_label"])
    op.create_index("ix_transactions_risk_score", "transactions", ["risk_score"])
    op.create_index("ix_transactions_decision", "transactions", ["decision"])
    op.create_index("ix_transactions_model_run_id", "transactions", ["model_run_id"])
    op.create_index("ix_transactions_threshold_config_id", "transactions", ["threshold_config_id"])
    op.create_index("ix_transactions_decision_risk", "transactions", ["decision", "risk_score"])

    op.create_table(
        "prediction_reasons",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("transaction_id", sa.Integer(), nullable=False),
        sa.Column("rank", sa.SmallInteger(), nullable=False),
        sa.Column("feature_name", sa.String(length=180), nullable=False),
        sa.Column("feature_value", sa.Text(), nullable=True),
        sa.Column("contribution", sa.Float(), nullable=False),
        sa.CheckConstraint("rank > 0", name="ck_prediction_reasons_rank"),
        sa.ForeignKeyConstraint(["transaction_id"], ["transactions.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("transaction_id", "rank", name="uq_prediction_reasons_transaction_rank"),
    )
    op.create_index("ix_prediction_reasons_transaction_id", "prediction_reasons", ["transaction_id"])

    op.create_table(
        "rule_hits",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("transaction_id", sa.Integer(), nullable=False),
        sa.Column("rule_id", sa.String(length=120), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("action IN ('NONE', 'REVIEW', 'BLOCK')", name="ck_rule_hits_action"),
        sa.ForeignKeyConstraint(["transaction_id"], ["transactions.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("transaction_id", "rule_id", name="uq_rule_hits_transaction_rule"),
    )
    op.create_index("ix_rule_hits_transaction_id", "rule_hits", ["transaction_id"])

    op.create_table(
        "review_cases",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("transaction_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("model_decision", sa.String(length=16), nullable=False),
        sa.Column("reviewer_decision", sa.String(length=16), nullable=True),
        sa.Column("reviewer_reason", sa.Text(), nullable=True),
        sa.Column("reviewer_id", sa.String(length=120), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "model_decision IN ('APPROVE', 'REVIEW', 'BLOCK')",
            name="ck_review_cases_model_decision",
        ),
        sa.CheckConstraint(
            "reviewer_decision IS NULL OR reviewer_decision IN ('APPROVE', 'DECLINE')",
            name="ck_review_cases_reviewer_decision",
        ),
        sa.CheckConstraint("status IN ('OPEN', 'DECIDED')", name="ck_review_cases_status"),
        sa.ForeignKeyConstraint(["transaction_id"], ["transactions.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_review_cases_transaction_id", "review_cases", ["transaction_id"], unique=True)
    op.create_index("ix_review_cases_status", "review_cases", ["status"])

    op.create_table(
        "cost_simulations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("simulation_group_id", sa.String(length=36), nullable=False),
        sa.Column("scenario", sa.String(length=16), nullable=False),
        sa.Column("cost_config_id", sa.Integer(), nullable=False),
        sa.Column("threshold_config_id", sa.Integer(), nullable=True),
        sa.Column("evaluation_split", sa.String(length=32), nullable=False),
        sa.Column("transaction_count", sa.Integer(), nullable=False),
        sa.Column("review_threshold", sa.Float(), nullable=False),
        sa.Column("block_threshold", sa.Float(), nullable=False),
        sa.Column("precision", sa.Float(), nullable=False),
        sa.Column("recall", sa.Float(), nullable=False),
        sa.Column("false_positives", sa.Integer(), nullable=False),
        sa.Column("false_negatives", sa.Integer(), nullable=False),
        sa.Column("review_volume", sa.Integer(), nullable=False),
        sa.Column("block_volume", sa.Integer(), nullable=False),
        sa.Column("approve_volume", sa.Integer(), nullable=False),
        sa.Column("fraud_loss", sa.Numeric(18, 4), nullable=False),
        sa.Column("false_positive_cost", sa.Numeric(18, 4), nullable=False),
        sa.Column("review_cost", sa.Numeric(18, 4), nullable=False),
        sa.Column("total_estimated_cost", sa.Numeric(18, 4), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("scenario IN ('CURRENT', 'PROPOSED')", name="ck_cost_simulations_scenario"),
        sa.CheckConstraint(
            "review_threshold < block_threshold",
            name="ck_cost_simulations_threshold_order",
        ),
        sa.ForeignKeyConstraint(["cost_config_id"], ["cost_configs.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["threshold_config_id"],
            ["threshold_configs.id"],
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_cost_simulations_cost_config_id", "cost_simulations", ["cost_config_id"])
    op.create_index(
        "ix_cost_simulations_group",
        "cost_simulations",
        ["simulation_group_id", "scenario"],
    )


def downgrade() -> None:
    op.drop_index("ix_cost_simulations_group", table_name="cost_simulations")
    op.drop_index("ix_cost_simulations_cost_config_id", table_name="cost_simulations")
    op.drop_table("cost_simulations")
    op.drop_index("ix_review_cases_status", table_name="review_cases")
    op.drop_index("ix_review_cases_transaction_id", table_name="review_cases")
    op.drop_table("review_cases")
    op.drop_index("ix_rule_hits_transaction_id", table_name="rule_hits")
    op.drop_table("rule_hits")
    op.drop_index("ix_prediction_reasons_transaction_id", table_name="prediction_reasons")
    op.drop_table("prediction_reasons")
    op.drop_index("ix_transactions_decision_risk", table_name="transactions")
    op.drop_index("ix_transactions_threshold_config_id", table_name="transactions")
    op.drop_index("ix_transactions_model_run_id", table_name="transactions")
    op.drop_index("ix_transactions_decision", table_name="transactions")
    op.drop_index("ix_transactions_risk_score", table_name="transactions")
    op.drop_index("ix_transactions_actual_label", table_name="transactions")
    op.drop_index("ix_transactions_transaction_dt", table_name="transactions")
    op.drop_index("ix_transactions_transaction_id", table_name="transactions")
    op.drop_table("transactions")
    op.drop_index("ix_threshold_configs_model_active", table_name="threshold_configs")
    op.drop_index("ix_threshold_configs_model_run_id", table_name="threshold_configs")
    op.drop_index("ix_threshold_configs_config_key", table_name="threshold_configs")
    op.drop_table("threshold_configs")
    op.drop_index("ix_cost_configs_config_key", table_name="cost_configs")
    op.drop_table("cost_configs")
    op.drop_index("ix_model_runs_evaluation_status", table_name="model_runs")
    op.drop_index("ix_model_runs_model_version", table_name="model_runs")
    op.drop_table("model_runs")
