from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base

JSONB_METADATA = JSONB().with_variant(JSON(), "sqlite")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ModelRun(Base):
    __tablename__ = "model_runs"
    __table_args__ = (
        CheckConstraint("evaluation_status IN ('NOT_EVALUATED', 'COMPLETE')", name="ck_model_runs_evaluation_status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(120))
    model_version: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    feature_set: Mapped[str] = mapped_column(String(120))
    evaluation_status: Mapped[str] = mapped_column(String(20), default="NOT_EVALUATED", index=True)
    evaluation_split: Mapped[str | None] = mapped_column(String(32), nullable=True)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    test_transaction_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fraud_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    roc_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    true_positives: Mapped[int | None] = mapped_column(Integer, nullable=True)
    false_positives: Mapped[int | None] = mapped_column(Integer, nullable=True)
    true_negatives: Mapped[int | None] = mapped_column(Integer, nullable=True)
    false_negatives: Mapped[int | None] = mapped_column(Integer, nullable=True)
    approve_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    review_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    block_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    false_positive_estimated_cost: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    false_negative_estimated_cost: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    review_cost: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    total_estimated_cost: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    active_rule_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB_METADATA, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    threshold_configs: Mapped[list[ThresholdConfig]] = relationship(back_populates="model_run")
    transactions: Mapped[list[Transaction]] = relationship(back_populates="model_run")


class CostConfig(Base):
    __tablename__ = "cost_configs"
    __table_args__ = (
        CheckConstraint("fraud_loss_fraction >= 0", name="ck_cost_configs_fraud_loss_fraction"),
        CheckConstraint("chargeback_fixed_cost >= 0", name="ck_cost_configs_chargeback_fixed_cost"),
        CheckConstraint("legitimate_margin_rate BETWEEN 0 AND 1", name="ck_cost_configs_legitimate_margin_rate"),
        CheckConstraint("false_positive_fixed_cost >= 0", name="ck_cost_configs_false_positive_fixed_cost"),
        CheckConstraint("manual_review_cost >= 0", name="ck_cost_configs_manual_review_cost"),
        CheckConstraint("review_fraud_catch_rate BETWEEN 0 AND 1", name="ck_cost_configs_review_fraud_catch_rate"),
        CheckConstraint(
            "review_legitimate_approval_rate BETWEEN 0 AND 1",
            name="ck_cost_configs_review_legitimate_approval_rate",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_key: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    currency: Mapped[str] = mapped_column(String(3), default="INR")
    fraud_loss_fraction: Mapped[float] = mapped_column(Float)
    chargeback_fixed_cost: Mapped[Decimal] = mapped_column(Numeric(18, 4), default=Decimal(0))
    legitimate_margin_rate: Mapped[float] = mapped_column(Float)
    false_positive_fixed_cost: Mapped[Decimal] = mapped_column(Numeric(18, 4), default=Decimal(0))
    manual_review_cost: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    review_fraud_catch_rate: Mapped[float] = mapped_column(Float)
    review_legitimate_approval_rate: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    threshold_configs: Mapped[list[ThresholdConfig]] = relationship(back_populates="cost_config")
    simulations: Mapped[list[CostSimulation]] = relationship(back_populates="cost_config")


class ThresholdConfig(Base):
    __tablename__ = "threshold_configs"
    __table_args__ = (
        CheckConstraint("review_threshold >= 0 AND review_threshold <= 1", name="ck_threshold_configs_review_range"),
        CheckConstraint("block_threshold >= 0 AND block_threshold <= 1", name="ck_threshold_configs_block_range"),
        CheckConstraint("review_threshold < block_threshold", name="ck_threshold_configs_order"),
        Index("ix_threshold_configs_model_active", "model_run_id", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_key: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"), index=True)
    cost_config_id: Mapped[int | None] = mapped_column(ForeignKey("cost_configs.id", ondelete="SET NULL"), nullable=True)
    review_threshold: Mapped[float] = mapped_column(Float)
    block_threshold: Mapped[float] = mapped_column(Float)
    selection_split: Mapped[str] = mapped_column(String(32), default="validation")
    objective: Mapped[str] = mapped_column(String(180), default="lowest estimated cost under configured assumptions")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    model_run: Mapped[ModelRun] = relationship(back_populates="threshold_configs")
    cost_config: Mapped[CostConfig | None] = relationship(back_populates="threshold_configs")
    transactions: Mapped[list[Transaction]] = relationship(back_populates="threshold_config")
    simulations: Mapped[list[CostSimulation]] = relationship(back_populates="threshold_config")


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        CheckConstraint("actual_label IS NULL OR actual_label IN (0, 1)", name="ck_transactions_actual_label"),
        CheckConstraint("risk_score >= 0 AND risk_score <= 1", name="ck_transactions_risk_score"),
        CheckConstraint("decision IN ('APPROVE', 'REVIEW', 'BLOCK')", name="ck_transactions_decision"),
        CheckConstraint("amount >= 0", name="ck_transactions_amount"),
        Index("ix_transactions_decision_risk", "decision", "risk_score"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    transaction_dt: Mapped[int] = mapped_column(BigInteger, index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    actual_label: Mapped[int | None] = mapped_column(SmallInteger, nullable=True, index=True)
    risk_score: Mapped[float] = mapped_column(Float, index=True)
    decision: Mapped[str] = mapped_column(String(16), index=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="RESTRICT"), index=True)
    threshold_config_id: Mapped[int] = mapped_column(ForeignKey("threshold_configs.id", ondelete="RESTRICT"), index=True)
    source: Mapped[str] = mapped_column(String(32), default="LIVE_SCORING")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    model_run: Mapped[ModelRun] = relationship(back_populates="transactions")
    threshold_config: Mapped[ThresholdConfig] = relationship(back_populates="transactions")
    reasons: Mapped[list[PredictionReason]] = relationship(
        back_populates="transaction",
        cascade="all, delete-orphan",
        order_by="PredictionReason.rank",
    )
    rule_hits: Mapped[list[RuleHit]] = relationship(
        back_populates="transaction",
        cascade="all, delete-orphan",
        order_by="RuleHit.id",
    )
    review_case: Mapped[ReviewCase | None] = relationship(
        back_populates="transaction",
        cascade="all, delete-orphan",
        uselist=False,
    )


class PredictionReason(Base):
    __tablename__ = "prediction_reasons"
    __table_args__ = (
        UniqueConstraint("transaction_id", "rank", name="uq_prediction_reasons_transaction_rank"),
        CheckConstraint("rank > 0", name="ck_prediction_reasons_rank"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id", ondelete="CASCADE"), index=True)
    rank: Mapped[int] = mapped_column(SmallInteger)
    feature_name: Mapped[str] = mapped_column(String(180))
    feature_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    contribution: Mapped[float] = mapped_column(Float)

    transaction: Mapped[Transaction] = relationship(back_populates="reasons")


class RuleHit(Base):
    __tablename__ = "rule_hits"
    __table_args__ = (
        UniqueConstraint("transaction_id", "rule_id", name="uq_rule_hits_transaction_rule"),
        CheckConstraint("action IN ('NONE', 'REVIEW', 'BLOCK')", name="ck_rule_hits_action"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id", ondelete="CASCADE"), index=True)
    rule_id: Mapped[str] = mapped_column(String(120))
    action: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    transaction: Mapped[Transaction] = relationship(back_populates="rule_hits")


class ReviewCase(Base):
    __tablename__ = "review_cases"
    __table_args__ = (
        CheckConstraint("status IN ('OPEN', 'DECIDED')", name="ck_review_cases_status"),
        CheckConstraint("model_decision IN ('APPROVE', 'REVIEW', 'BLOCK')", name="ck_review_cases_model_decision"),
        CheckConstraint(
            "reviewer_decision IS NULL OR reviewer_decision IN ('APPROVE', 'DECLINE')",
            name="ck_review_cases_reviewer_decision",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id", ondelete="CASCADE"), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(24), default="OPEN", index=True)
    model_decision: Mapped[str] = mapped_column(String(16), default="REVIEW")
    reviewer_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reviewer_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    transaction: Mapped[Transaction] = relationship(back_populates="review_case")


class CostSimulation(Base):
    __tablename__ = "cost_simulations"
    __table_args__ = (
        CheckConstraint("scenario IN ('CURRENT', 'PROPOSED')", name="ck_cost_simulations_scenario"),
        CheckConstraint("review_threshold < block_threshold", name="ck_cost_simulations_threshold_order"),
        Index("ix_cost_simulations_group", "simulation_group_id", "scenario"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    simulation_group_id: Mapped[str] = mapped_column(String(36))
    scenario: Mapped[str] = mapped_column(String(16))
    cost_config_id: Mapped[int] = mapped_column(ForeignKey("cost_configs.id", ondelete="RESTRICT"), index=True)
    threshold_config_id: Mapped[int | None] = mapped_column(
        ForeignKey("threshold_configs.id", ondelete="SET NULL"),
        nullable=True,
    )
    evaluation_split: Mapped[str] = mapped_column(String(32), default="test")
    transaction_count: Mapped[int] = mapped_column(Integer)
    review_threshold: Mapped[float] = mapped_column(Float)
    block_threshold: Mapped[float] = mapped_column(Float)
    precision: Mapped[float] = mapped_column(Float)
    recall: Mapped[float] = mapped_column(Float)
    false_positives: Mapped[int] = mapped_column(Integer)
    false_negatives: Mapped[int] = mapped_column(Integer)
    review_volume: Mapped[int] = mapped_column(Integer)
    block_volume: Mapped[int] = mapped_column(Integer)
    approve_volume: Mapped[int] = mapped_column(Integer)
    fraud_loss: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    false_positive_cost: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    review_cost: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    total_estimated_cost: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    cost_config: Mapped[CostConfig] = relationship(back_populates="simulations")
    threshold_config: Mapped[ThresholdConfig | None] = relationship(back_populates="simulations")
