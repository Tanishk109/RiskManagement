"""Add chargeback evidence responder tables.

Revision ID: 0002_chargeback_evidence
Revises: 0002_review_decision_block
Create Date: 2026-08-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_chargeback_evidence"
down_revision: str | None = "0002_review_decision_block"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chargeback_cases",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dispute_id", sa.String(length=100), nullable=False),
        sa.Column("transaction_id", sa.String(length=100), nullable=False),
        sa.Column("amount", sa.Numeric(18, 4), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("reason", sa.String(length=64), nullable=False),
        sa.Column("deadline", sa.Date(), nullable=False),
        sa.Column("customer_information", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("order_information", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("delivery_information", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("merchant_notes", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("amount > 0", name="ck_chargeback_cases_amount"),
        sa.CheckConstraint(
            "status IN ('DRAFT', 'READY_FOR_HUMAN_REVIEW', 'APPROVED_FOR_EXPORT')",
            name="ck_chargeback_cases_status",
        ),
    )
    op.create_index("ix_chargeback_cases_dispute_id", "chargeback_cases", ["dispute_id"], unique=True)
    op.create_index("ix_chargeback_cases_transaction_id", "chargeback_cases", ["transaction_id"])
    op.create_index("ix_chargeback_cases_reason", "chargeback_cases", ["reason"])
    op.create_index("ix_chargeback_cases_deadline", "chargeback_cases", ["deadline"])
    op.create_index("ix_chargeback_cases_status", "chargeback_cases", ["status"])

    op.create_table(
        "chargeback_evidence",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(length=48), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("storage_key", sa.String(length=255), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("size_bytes > 0", name="ck_chargeback_evidence_size"),
        sa.ForeignKeyConstraint(["case_id"], ["chargeback_cases.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_chargeback_evidence_case_id", "chargeback_evidence", ["case_id"])
    op.create_index("ix_chargeback_evidence_category", "chargeback_evidence", ["category"])
    op.create_index("ix_chargeback_evidence_storage_key", "chargeback_evidence", ["storage_key"], unique=True)

    op.create_table(
        "chargeback_drafts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("draft_text", sa.Text(), nullable=False),
        sa.Column("generation_method", sa.String(length=64), nullable=False),
        sa.Column("evidence_count", sa.Integer(), nullable=False),
        sa.Column("missing_categories", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("human_approved", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["case_id"], ["chargeback_cases.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_chargeback_drafts_case_id", "chargeback_drafts", ["case_id"], unique=True)


def downgrade() -> None:
    op.drop_table("chargeback_drafts")
    op.drop_table("chargeback_evidence")
    op.drop_table("chargeback_cases")
