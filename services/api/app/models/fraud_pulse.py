from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FraudPulseRun(Base):
    __tablename__ = "fraud_pulse_runs"
    __table_args__ = (
        CheckConstraint("source IN ('VALIDATION_REPLAY', 'MERCHANT_UPLOAD')", name="ck_fraud_pulse_runs_source"),
        CheckConstraint(
            "detector_method IN ('rolling_zscore', 'ewma', 'percent_deviation')",
            name="ck_fraud_pulse_runs_method",
        ),
        CheckConstraint(
            "metric IN ('transaction_count', 'mean_risk_score', 'high_risk_count', 'high_risk_amount')",
            name="ck_fraud_pulse_runs_metric",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    detector_method: Mapped[str] = mapped_column(String(32))
    metric: Mapped[str] = mapped_column(String(32))
    window_seconds: Mapped[int] = mapped_column(Integer)
    baseline_windows: Mapped[int] = mapped_column(Integer)
    sensitivity: Mapped[float] = mapped_column(Float)
    ewma_alpha: Mapped[float] = mapped_column(Float)
    percent_deviation_threshold: Mapped[float] = mapped_column(Float)
    rows_scored: Mapped[int] = mapped_column(Integer)
    window_count: Mapped[int] = mapped_column(Integer)
    alert_count: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(120))
    review_threshold: Mapped[float] = mapped_column(Float)
    block_threshold: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    alerts: Mapped[list[FraudPulseAlert]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="FraudPulseAlert.window_start"
    )


class FraudPulseAlert(Base):
    __tablename__ = "fraud_pulse_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("fraud_pulse_runs.id", ondelete="CASCADE"), index=True)
    window_start: Mapped[int] = mapped_column(BigInteger, index=True)
    window_end: Mapped[int] = mapped_column(BigInteger)
    current_value: Mapped[float] = mapped_column(Float)
    baseline_value: Mapped[float] = mapped_column(Float)
    absolute_change: Mapped[float] = mapped_column(Float)
    percent_deviation: Mapped[float | None] = mapped_column(Float, nullable=True)
    detector_score: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[FraudPulseRun] = relationship(back_populates="alerts")
