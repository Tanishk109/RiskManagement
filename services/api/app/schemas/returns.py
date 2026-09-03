from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReturnOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_value: float = Field(ge=0, le=100_000_000)
    quantity: float = Field(gt=0, le=10_000_000)
    unique_stock_count: int = Field(ge=1, le=100_000)
    country: str = Field(min_length=1, max_length=120)
    stock_code: str = Field(min_length=1, max_length=120)
    prior_order_count: int = Field(ge=0, le=100_000_000)
    prior_cancellation_rate: float = Field(ge=0, le=1)
    prior_average_order_value: float = Field(ge=0, le=100_000_000)
    order_hour: int = Field(ge=0, le=23)
    order_day_of_week: int = Field(ge=0, le=6)

    @field_validator("country", "stock_code")
    @classmethod
    def strip_category(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("categorical values cannot be blank")
        return normalized


class ReturnThresholds(BaseModel):
    medium: float
    high: float


class ReturnScoreResult(BaseModel):
    row: int | None = None
    return_risk_probability: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    model_version: str
    thresholds: ReturnThresholds
    automatic_rejection: Literal[False] = False


class ReturnBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[ReturnOrderInput] = Field(min_length=1, max_length=1_000)


class ReturnBatchResponse(BaseModel):
    rows_received: int
    rows_scored: int
    results: list[ReturnScoreResult]
    model_version: str
    thresholds: ReturnThresholds
    uploaded_file_persisted: Literal[False] = False
    automatic_rejection: Literal[False] = False


class ReturnMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    average_precision: float
    threshold: float
    positives: int
    negatives: int
    predicted_positive: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


class ReturnStatus(BaseModel):
    module: Literal["Return-Risk Scorer"] = "Return-Risk Scorer"
    data_source: str
    dataset_id: int
    model_version: str
    evaluation_status: Literal["Evaluated on chronological UCI test partition"]
    proxy_disclosure: str
    feature_schema: list[str]
    categorical_features: list[str]
    test_metrics: ReturnMetrics
    thresholds: ReturnThresholds
    automatic_rejection: Literal[False] = False
    ieee_cis_model_modified: Literal[False] = False
    ieee_cis_held_out_test_accessed: Literal[False] = False
    limitations: list[str]
