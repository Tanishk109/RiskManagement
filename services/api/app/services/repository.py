from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import Select, select
from sqlalchemy.orm import Session, selectinload

from ..models import (
    ModelRun,
    PredictionReason,
    ReviewCase,
    RuleHit,
    ThresholdConfig,
    Transaction,
)
from ..schemas.risk import Factor, ReviewDecisionRequest, ReviewOut, TransactionOut
from .rules_engine import RuleHit as EvaluatedRuleHit


def _model_error(row: Transaction) -> bool | None:
    if row.actual_label is None:
        return None
    predicted_fraud = row.decision == "BLOCK"
    return predicted_fraud != bool(row.actual_label)


def _transaction_load_options():
    return (
        selectinload(Transaction.reasons),
        selectinload(Transaction.rule_hits),
        selectinload(Transaction.model_run),
    )


def transaction_out(row: Transaction) -> TransactionOut:
    return TransactionOut(
        transaction_id=row.transaction_id,
        transaction_dt=row.transaction_dt,
        amount=float(row.amount),
        actual_label=row.actual_label,
        risk_score=row.risk_score,
        model_version=row.model_run.model_version,
        decision=row.decision,
        rules_triggered=[hit.rule_id for hit in row.rule_hits],
        top_factors=[
            Factor(
                feature_name=reason.feature_name,
                feature_value=reason.feature_value,
                contribution=reason.contribution,
            )
            for reason in row.reasons
        ],
        model_error=_model_error(row),
    )


def list_transactions(
    db: Session,
    *,
    decision: str | None,
    actual_label: int | None,
    minimum_risk: float | None,
    maximum_risk: float | None,
    limit: int,
    cursor: int | None,
) -> tuple[list[TransactionOut], int | None]:
    statement: Select[tuple[Transaction]] = (
        select(Transaction)
        .options(*_transaction_load_options())
        .order_by(Transaction.id)
        .limit(limit + 1)
    )
    if decision:
        statement = statement.where(Transaction.decision == decision)
    if actual_label is not None:
        statement = statement.where(Transaction.actual_label == actual_label)
    if minimum_risk is not None:
        statement = statement.where(Transaction.risk_score >= minimum_risk)
    if maximum_risk is not None:
        statement = statement.where(Transaction.risk_score <= maximum_risk)
    if cursor is not None:
        statement = statement.where(Transaction.id > cursor)

    rows = list(db.scalars(statement).unique())
    next_cursor = rows[limit - 1].id if len(rows) > limit else None
    return [transaction_out(row) for row in rows[:limit]], next_cursor


def get_transaction(db: Session, transaction_id: str) -> TransactionOut | None:
    row = db.scalar(
        select(Transaction)
        .where(Transaction.transaction_id == transaction_id)
        .options(*_transaction_load_options())
    )
    return transaction_out(row) if row else None


def review_out(row: ReviewCase) -> ReviewOut:
    return ReviewOut(
        id=row.id,
        transaction_id=row.transaction.transaction_id,
        status=row.status,
        amount=float(row.transaction.amount),
        risk_score=row.transaction.risk_score,
        primary_factors=[reason.feature_name for reason in row.transaction.reasons[:3]],
        reviewer_decision=row.reviewer_decision,
        reviewer_reason=row.reviewer_reason,
        reviewer_id=row.reviewer_id,
        reviewed_at=row.reviewed_at,
    )


def list_reviews(db: Session, status: str | None = "OPEN") -> list[ReviewOut]:
    statement = (
        select(ReviewCase)
        .options(selectinload(ReviewCase.transaction).selectinload(Transaction.reasons))
        .order_by(ReviewCase.id)
    )
    if status:
        statement = statement.where(ReviewCase.status == status)
    return [review_out(row) for row in db.scalars(statement).unique()]


def decide_review(db: Session, review_id: int, payload: ReviewDecisionRequest) -> ReviewOut | None:
    row = db.scalar(
        select(ReviewCase)
        .where(ReviewCase.id == review_id)
        .options(selectinload(ReviewCase.transaction).selectinload(Transaction.reasons))
    )
    if row is None:
        return None
    if row.status != "OPEN":
        raise ValueError("Review case has already been decided")
    row.status = "DECIDED"
    row.reviewer_decision = payload.decision
    row.reviewer_reason = payload.reason
    row.reviewer_id = payload.reviewer_id
    row.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return review_out(row)


def persist_scored_transaction(
    db: Session,
    *,
    transaction_id: str,
    transaction_dt: int,
    amount: float,
    risk_score: float,
    decision: str,
    model_run: ModelRun,
    threshold_config: ThresholdConfig,
    rule_hits: list[EvaluatedRuleHit],
    factors: list[Factor],
) -> Transaction:
    row = Transaction(
        transaction_id=transaction_id,
        transaction_dt=transaction_dt,
        amount=Decimal(str(amount)),
        actual_label=None,
        risk_score=risk_score,
        decision=decision,
        model_run_id=model_run.id,
        threshold_config_id=threshold_config.id,
        source="LIVE_SCORING",
    )
    row.reasons = [
        PredictionReason(
            rank=rank,
            feature_name=factor.feature_name,
            feature_value=None if factor.feature_value is None else str(factor.feature_value),
            contribution=factor.contribution,
        )
        for rank, factor in enumerate(factors, start=1)
    ]
    row.rule_hits = [
        RuleHit(rule_id=hit.rule_id, action=hit.action, reason=hit.reason)
        for hit in rule_hits
    ]
    if decision == "REVIEW":
        row.review_case = ReviewCase(status="OPEN", model_decision="REVIEW")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
