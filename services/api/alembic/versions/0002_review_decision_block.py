"""Use APPROVE/BLOCK for human review decisions.

Revision ID: 0002_review_decision_block
Revises: 0001_operational_schema
"""

from __future__ import annotations

from alembic import op

revision = "0002_review_decision_block"
down_revision = "0001_operational_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_review_cases_reviewer_decision", "review_cases", type_="check")
    op.execute("UPDATE review_cases SET reviewer_decision = 'BLOCK' WHERE reviewer_decision = 'DECLINE'")
    op.create_check_constraint(
        "ck_review_cases_reviewer_decision",
        "review_cases",
        "reviewer_decision IS NULL OR reviewer_decision IN ('APPROVE', 'BLOCK')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_review_cases_reviewer_decision", "review_cases", type_="check")
    op.execute("UPDATE review_cases SET reviewer_decision = 'DECLINE' WHERE reviewer_decision = 'BLOCK'")
    op.create_check_constraint(
        "ck_review_cases_reviewer_decision",
        "review_cases",
        "reviewer_decision IS NULL OR reviewer_decision IN ('APPROVE', 'DECLINE')",
    )
