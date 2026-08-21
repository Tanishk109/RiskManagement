from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    transaction_dt: Mapped[int] = mapped_column(Integer, index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    actual_label: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    risk_score: Mapped[float] = mapped_column(Float, index=True)
    model_version: Mapped[str] = mapped_column(String(120))
    decision: Mapped[str] = mapped_column(String(16), index=True)
    rules_triggered: Mapped[list[str]] = mapped_column(JSON, default=list)
    feature_payload: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    reasons: Mapped[list[PredictionReason]] = relationship(back_populates="transaction", cascade="all, delete-orphan")
    review_case: Mapped[ReviewCase | None] = relationship(back_populates="transaction", cascade="all, delete-orphan", uselist=False)


class PredictionReason(Base):
    __tablename__ = "prediction_reasons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id", ondelete="CASCADE"), index=True)
    feature_name: Mapped[str] = mapped_column(String(180))
    feature_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    contribution: Mapped[float] = mapped_column(Float)

    transaction: Mapped[Transaction] = relationship(back_populates="reasons")


class ReviewCase(Base):
    __tablename__ = "review_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id", ondelete="CASCADE"), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(24), default="OPEN", index=True)
    model_decision: Mapped[str] = mapped_column(String(16), default="REVIEW")
    reviewer_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reviewer_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    transaction: Mapped[Transaction] = relationship(back_populates="review_case")


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(120))
    model_version: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    feature_set: Mapped[str] = mapped_column(String(120))
    metrics_json: Mapped[dict[str, object]] = mapped_column(JSON)


class CostConfig(Base):
    __tablename__ = "cost_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    currency: Mapped[str] = mapped_column(String(3), default="INR")
    fraud_loss_fraction: Mapped[float] = mapped_column(Float)
    legitimate_margin_rate: Mapped[float] = mapped_column(Float)
    manual_review_cost: Mapped[float] = mapped_column(Float)
    review_fraud_catch_rate: Mapped[float] = mapped_column(Float)
    review_legitimate_approval_rate: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
