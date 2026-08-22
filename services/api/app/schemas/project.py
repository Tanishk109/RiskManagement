from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DatasetStatus(BaseModel):
    status: str
    name: str
    transactions: int
    fraud_transactions: int
    legitimate_transactions: int
    fraud_prevalence: float
    identity_rows: int
    identity_coverage: float


class SplitStatus(BaseModel):
    status: str
    strategy: str
    train_rows: int
    validation_rows: int
    test_rows: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    train_transaction_dt_min: int
    train_transaction_dt_max: int
    validation_transaction_dt_min: int
    validation_transaction_dt_max: int
    test_transaction_dt_min: int
    test_transaction_dt_max: int
    test_status: str


class StatusValue(BaseModel):
    status: str
    test_status: str | None = None
    name: str | None = None
    experiment_id: str | None = None


class ProjectStatusResponse(BaseModel):
    dataset: DatasetStatus
    split: SplitStatus
    baseline: StatusValue
    candidate_model: StatusValue
    threshold_analysis: StatusValue
    rules: StatusValue
    operational_thresholds: StatusValue
    final_test: StatusValue


class ValidationMetrics(BaseModel):
    average_precision: float
    roc_auc: float
    precision_at_0_5: float
    recall_at_0_5: float
    f1_at_0_5: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    threshold: float


class ComparedModel(BaseModel):
    name: str
    experiment_id: str
    metrics: ValidationMetrics


class CandidateDetails(BaseModel):
    status: str
    feature_count: int
    identity_fields_included: bool
    class_weight: str
    identity_ap_loss: float
    selection_reason: str


class FailureSlice(BaseModel):
    slice: str
    fraud_support: int
    logistic_recall: float
    catboost_recall: float
    absolute_improvement: float


class FalseNegativeAmounts(BaseModel):
    count: int
    transaction_amount_total: float
    transaction_amount_max: float


class FailureAnalysis(BaseModel):
    label: str
    slices: list[FailureSlice]
    false_negatives: FalseNegativeAmounts


class PrecisionRecallPoint(BaseModel):
    recall: float
    precision: float


class PrecisionRecallSeries(BaseModel):
    model: str
    points: list[PrecisionRecallPoint]


class ModelComparisonResponse(BaseModel):
    status: str
    split: str
    held_out_test_status: str
    threshold: float
    logistic_regression: ComparedModel
    catboost: ComparedModel
    average_precision_relative_improvement: float
    candidate_details: CandidateDetails
    failure_analysis: FailureAnalysis
    precision_recall_curves: list[PrecisionRecallSeries]
    provenance: str


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    status: str
    model: str
    items: list[FeatureImportanceItem]
    note: str


class ValidationTransaction(BaseModel):
    transaction_id: str
    transaction_dt: int
    transaction_amount: float
    actual_label: Literal[0, 1]
    fraud_probability: float
    predicted_label_at_0_5: Literal[0, 1]
    outcome: Literal["TRUE_POSITIVE", "FALSE_POSITIVE", "FALSE_NEGATIVE", "TRUE_NEGATIVE"]
    model_error: bool
    features: dict[str, str | float | int | None]


class ValidationTransactionPage(BaseModel):
    status: str
    split: str
    threshold: float
    filter: str
    page: int
    page_size: int
    total: int
    page_count: int
    items: list[ValidationTransaction]


class InterestingCase(ValidationTransaction):
    case_type: str


class InterestingCasesResponse(BaseModel):
    status: str
    split: str
    cases: list[InterestingCase]
