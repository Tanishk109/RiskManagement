from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DisputeReason = Literal[
    "item_not_received",
    "duplicate",
    "refund_not_received",
    "cancelled_recurring",
    "not_as_described",
    "other",
]
EvidenceCategory = Literal[
    "invoice",
    "proof_of_delivery",
    "tracking",
    "customer_communication",
    "refund_evidence",
    "merchant_policy",
    "other",
]


class ChargebackCaseCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispute_id: str = Field(min_length=2, max_length=100, pattern=r"^[A-Za-z0-9._:-]+$")
    transaction_id: str = Field(min_length=1, max_length=100)
    amount: float = Field(gt=0, le=1_000_000_000)
    currency: str = Field(default="INR", min_length=3, max_length=3, pattern=r"^[A-Za-z]{3}$")
    reason: DisputeReason
    deadline: date
    customer_information: dict[str, str] = Field(default_factory=dict)
    order_information: dict[str, str] = Field(default_factory=dict)
    delivery_information: dict[str, str] = Field(default_factory=dict)
    merchant_notes: str | None = Field(default=None, max_length=5000)

    @field_validator("customer_information", "order_information", "delivery_information")
    @classmethod
    def validate_metadata(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 20:
            raise ValueError("information groups accept at most 20 fields")
        for key, item in value.items():
            if not key.strip() or len(key) > 80 or len(item) > 1000:
                raise ValueError("information keys must be 1-80 characters and values at most 1000")
        return {key.strip(): item.strip() for key, item in value.items() if item.strip()}


class EvidenceUpload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: EvidenceCategory
    filename: str = Field(min_length=1, max_length=255)
    content_type: Literal["application/pdf", "image/png", "image/jpeg"]
    base64_content: str = Field(min_length=4)


class EvidenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    category: EvidenceCategory
    original_filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime


class CompletenessOut(BaseModel):
    present: list[EvidenceCategory]
    expected: list[EvidenceCategory]
    missing: list[EvidenceCategory]
    present_count: int
    expected_count: int
    ratio: float = Field(ge=0, le=1)
    checklist_basis: str


class DraftOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    draft_text: str
    generation_method: str
    evidence_count: int
    missing_categories: list[EvidenceCategory]
    human_approved: bool
    updated_at: datetime


class ChargebackCaseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dispute_id: str
    transaction_id: str
    amount: float
    currency: str
    reason: DisputeReason
    deadline: date
    customer_information: dict[str, str]
    order_information: dict[str, str]
    delivery_information: dict[str, str]
    merchant_notes: str | None
    status: str
    created_at: datetime
    updated_at: datetime
    evidence: list[EvidenceOut]
    draft: DraftOut | None
    completeness: CompletenessOut


class DraftSave(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_text: str = Field(min_length=20, max_length=20000)
    human_approved: bool = False


class ChargebackStatus(BaseModel):
    module: Literal["Chargeback Evidence Responder"]
    data_source: str
    evaluation_status: Literal["Not evaluated yet"]
    checklist_basis: str
    limitations: list[str]
    accepted_file_types: list[str]
    maximum_file_size_bytes: int
    automatic_submission: Literal[False]
