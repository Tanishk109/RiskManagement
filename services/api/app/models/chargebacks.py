from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base

JSONB_METADATA = JSONB().with_variant(JSON(), "sqlite")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChargebackCase(Base):
    __tablename__ = "chargeback_cases"
    __table_args__ = (
        CheckConstraint("amount > 0", name="ck_chargeback_cases_amount"),
        CheckConstraint(
            "status IN ('DRAFT', 'READY_FOR_HUMAN_REVIEW', 'APPROVED_FOR_EXPORT')",
            name="ck_chargeback_cases_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dispute_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    transaction_id: Mapped[str] = mapped_column(String(100), index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    currency: Mapped[str] = mapped_column(String(3), default="INR")
    reason: Mapped[str] = mapped_column(String(64), index=True)
    deadline: Mapped[date] = mapped_column(Date, index=True)
    customer_information: Mapped[dict[str, Any]] = mapped_column(JSONB_METADATA, default=dict)
    order_information: Mapped[dict[str, Any]] = mapped_column(JSONB_METADATA, default=dict)
    delivery_information: Mapped[dict[str, Any]] = mapped_column(JSONB_METADATA, default=dict)
    merchant_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="DRAFT", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    evidence: Mapped[list[ChargebackEvidence]] = relationship(
        back_populates="case", cascade="all, delete-orphan", order_by="ChargebackEvidence.uploaded_at"
    )
    draft: Mapped[ChargebackDraft | None] = relationship(
        back_populates="case", cascade="all, delete-orphan", uselist=False
    )


class ChargebackEvidence(Base):
    __tablename__ = "chargeback_evidence"
    __table_args__ = (CheckConstraint("size_bytes > 0", name="ck_chargeback_evidence_size"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("chargeback_cases.id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(48), index=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    storage_key: Mapped[str] = mapped_column(String(255), unique=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    case: Mapped[ChargebackCase] = relationship(back_populates="evidence")


class ChargebackDraft(Base):
    __tablename__ = "chargeback_drafts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(
        ForeignKey("chargeback_cases.id", ondelete="CASCADE"), unique=True, index=True
    )
    draft_text: Mapped[str] = mapped_column(Text)
    generation_method: Mapped[str] = mapped_column(String(64), default="DETERMINISTIC_EVIDENCE_TEMPLATE")
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    missing_categories: Mapped[list[str]] = mapped_column(JSONB_METADATA, default=list)
    human_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    case: Mapped[ChargebackCase] = relationship(back_populates="draft")
