from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ReturnPrediction(Base):
    """Optional runtime audit record; ML training rows never belong in PostgreSQL."""

    __tablename__ = "return_predictions"
    __table_args__ = (
        CheckConstraint("return_risk_probability >= 0", name="ck_return_predictions_probability_low"),
        CheckConstraint("return_risk_probability <= 1", name="ck_return_predictions_probability_high"),
        CheckConstraint("risk_level IN ('LOW', 'MEDIUM', 'HIGH')", name="ck_return_predictions_level"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_value: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    unique_stock_count: Mapped[int] = mapped_column(Integer)
    country: Mapped[str] = mapped_column(String(120))
    stock_code: Mapped[str] = mapped_column(String(120))
    prior_order_count: Mapped[int] = mapped_column(Integer)
    prior_cancellation_rate: Mapped[float] = mapped_column(Float)
    prior_average_order_value: Mapped[float] = mapped_column(Float)
    order_hour: Mapped[int] = mapped_column(Integer)
    order_day_of_week: Mapped[int] = mapped_column(Integer)
    return_risk_probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(12), index=True)
    model_version: Mapped[str] = mapped_column(String(120), index=True)
    automatic_rejection: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
