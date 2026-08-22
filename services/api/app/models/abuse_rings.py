from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
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


class AbuseGraphRun(Base):
    __tablename__ = "abuse_graph_runs"
    __table_args__ = (
        CheckConstraint("data_partition = 'validation'", name="ck_abuse_graph_runs_partition"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    data_partition: Mapped[str] = mapped_column(String(24), default="validation")
    min_attribute_degree: Mapped[int] = mapped_column(Integer)
    max_attribute_degree: Mapped[int] = mapped_column(Integer)
    minimum_cluster_transactions: Mapped[int] = mapped_column(Integer)
    minimum_high_risk_transactions: Mapped[int] = mapped_column(Integer)
    minimum_high_risk_share: Mapped[float] = mapped_column(Float)
    transaction_nodes: Mapped[int] = mapped_column(Integer)
    shared_attribute_nodes: Mapped[int] = mapped_column(Integer)
    edge_count: Mapped[int] = mapped_column(Integer)
    suspicious_cluster_count: Mapped[int] = mapped_column(Integer)
    suppressed_attribute_values: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(120))
    confirmed_fraud_ring_claimed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    clusters: Mapped[list[AbuseClusterRecord]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="AbuseClusterRecord.id"
    )


class AbuseClusterRecord(Base):
    __tablename__ = "abuse_cluster_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("abuse_graph_runs.id", ondelete="CASCADE"), index=True)
    cluster_key: Mapped[str] = mapped_column(String(64), index=True)
    transaction_count: Mapped[int] = mapped_column(Integer)
    shared_attribute_count: Mapped[int] = mapped_column(Integer)
    edge_count: Mapped[int] = mapped_column(Integer)
    high_risk_count: Mapped[int] = mapped_column(Integer)
    high_risk_share: Mapped[float] = mapped_column(Float)
    average_risk_score: Mapped[float] = mapped_column(Float)
    maximum_risk_score: Mapped[float] = mapped_column(Float)
    total_transaction_amount: Mapped[float] = mapped_column(Float)
    high_risk_amount: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[AbuseGraphRun] = relationship(back_populates="clusters")
