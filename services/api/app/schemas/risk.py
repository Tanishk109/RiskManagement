from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Decision = Literal["APPROVE", "REVIEW", "BLOCK"]
ReviewerDecision = Literal["APPROVE", "BLOCK"]


class ModelInfo(BaseModel):
    available: bool
    name: str | None = None
    version: str | None = None
    trained_at: datetime | None = None
    feature_set: str | None = None
    thresholds: dict[str, float] | None = None


class Factor(BaseModel):
    feature_name: str
    feature_value: str | float | int | None = None
    contribution: float


class TransactionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    transaction_id: str
    transaction_dt: int
    amount: float
    actual_label: int | None
    risk_score: float
    model_version: str
    decision: Decision
    rules_triggered: list[str] = Field(default_factory=list)
    top_factors: list[Factor] = Field(default_factory=list)
    model_error: bool | None = None


class TransactionList(BaseModel):
    items: list[TransactionOut]
    next_cursor: int | None


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str | None = Field(default=None, max_length=80)
    features: dict[str, Any]
    persist: bool = False

    @model_validator(mode="after")
    def reject_label_and_future_fields(self):
        forbidden = {"isFraud", "actual_label", "future_chargeback_outcome", "future_fraud_label"}
        used = sorted(forbidden.intersection(self.features))
        if used:
            raise ValueError(f"Scoring features contain label/future fields: {', '.join(used)}")
        return self


class ScoreResponse(BaseModel):
    fraud_probability: float = Field(ge=0, le=1)
    risk_score: float
    decision: Decision
    rules_triggered: list[str]
    top_factors: list[Factor]
    model_version: str
    threshold_config_id: str
    threshold_configuration: dict[str, str | float | bool]
    feature_schema: list[str]
    held_out_test_accessed: Literal[False]


class BatchScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv_content: str = Field(min_length=1)


class BatchScoreItem(BaseModel):
    row: int
    transaction_id: str
    fraud_probability: float = Field(ge=0, le=1)
    decision: Decision


class BatchInvalidRow(BaseModel):
    row: int
    transaction_id: str | None = None
    errors: list[str]


class BatchScoreSummary(BaseModel):
    rows_received: int
    rows_processed: int
    approved: int
    reviewed: int
    blocked: int
    invalid_rows: int


class BatchScoreResponse(BaseModel):
    summary: BatchScoreSummary
    results: list[BatchScoreItem]
    invalid_rows: list[BatchInvalidRow]
    model_version: str
    threshold_configuration: dict[str, str | float | bool]
    feature_schema: list[str]
    upload_persisted: Literal[False]
    held_out_test_accessed: Literal[False]


class ReviewOut(BaseModel):
    id: int
    transaction_id: str
    status: str
    amount: float
    risk_score: float
    primary_factors: list[str]
    reviewer_decision: str | None = None
    reviewer_reason: str | None = None
    reviewer_id: str | None = None
    reviewed_at: datetime | None = None


class ReviewDecisionRequest(BaseModel):
    decision: ReviewerDecision
    reason: str | None = Field(default=None, max_length=1000)
    reviewer_id: str | None = Field(default=None, min_length=2, max_length=120)


class CostAssumptions(BaseModel):
    currency: str = Field(default="INR", min_length=3, max_length=3)
    fraud_loss_fraction: float = Field(default=1.0, ge=0, le=1)
    chargeback_fixed_cost: float = Field(default=0, ge=0)
    legitimate_margin_rate: float = Field(default=0.2, ge=0, le=1)
    false_positive_fixed_cost: float = Field(default=0, ge=0)
    manual_review_cost: float = Field(default=150, ge=0)
    review_fraud_catch_rate: float = Field(default=0.9, ge=0, le=1)
    review_legitimate_approval_rate: float = Field(default=0.98, ge=0, le=1)


class CostSimulationRequest(BaseModel):
    review_threshold: float = Field(ge=0, le=1)
    block_threshold: float = Field(ge=0, le=1)
    assumptions: CostAssumptions = Field(default_factory=CostAssumptions)

    @model_validator(mode="after")
    def validate_threshold_order(self):
        if self.review_threshold >= self.block_threshold:
            raise ValueError("review_threshold must be less than block_threshold")
        return self


class CostOutcome(BaseModel):
    precision: float
    recall: float
    false_positives: int
    false_negatives: int
    review_volume: int
    block_volume: int
    fraud_loss: float
    false_positive_cost: float
    review_cost: float
    total_estimated_cost: float


class CostSimulationResponse(BaseModel):
    evaluated: bool
    provenance: str
    current: CostOutcome | None
    proposed: CostOutcome | None
    simulation_group_id: str | None = None


class MetricsResponse(BaseModel):
    evaluated: bool
    provenance: str
    generated_at: datetime | None = None
    metrics: dict[str, float | int] | None = None


class BootstrapResponse(BaseModel):
    status: str
    evaluated: bool
    generated_at: datetime | None
    dataset: dict[str, str | bool]
    model: ModelInfo
    metrics: dict[str, float | int] | None
    decision_distribution: dict[str, dict[str, float | int]] | None
    confusion_matrix: dict[str, int] | None
    transactions: list[TransactionOut]
    reviews: list[ReviewOut]
    rules: dict[str, str | int]
    provenance: str
