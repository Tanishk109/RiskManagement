from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas.risk import BootstrapResponse, MetricsResponse, ModelInfo
from ..services.artifacts import ArtifactUnavailable, load_metrics, load_model_metadata
from ..services.repository import list_reviews, list_transactions

router = APIRouter(prefix="/api/v1", tags=["evidence"])


def _model_info() -> ModelInfo:
    try:
        metadata = load_model_metadata()
    except ArtifactUnavailable:
        return ModelInfo(available=False)
    return ModelInfo(
        available=True,
        name=metadata.get("model_name"),
        version=metadata.get("model_version"),
        trained_at=metadata.get("trained_at"),
        feature_set=metadata.get("feature_set"),
        thresholds=metadata.get("thresholds"),
    )


@router.get("/model", response_model=ModelInfo)
def model_info() -> ModelInfo:
    return _model_info()


@router.get("/metrics/summary", response_model=MetricsResponse)
def metrics_summary() -> MetricsResponse:
    try:
        artifact = load_metrics()
    except ArtifactUnavailable:
        return MetricsResponse(evaluated=False, provenance="Not evaluated yet")
    metric_keys = {
        "test_transaction_count",
        "fraud_count",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
        "brier_score",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "approve_count",
        "review_count",
        "block_count",
        "false_positive_estimated_cost",
        "false_negative_estimated_cost",
        "review_cost",
        "total_estimated_cost",
        "review_threshold",
        "block_threshold",
    }
    public_metrics = {key: artifact[key] for key in metric_keys if key in artifact}
    return MetricsResponse(
        evaluated=True,
        provenance="Calculated from the held-out temporal test set.",
        generated_at=artifact.get("generated_at"),
        metrics=public_metrics,
    )


@router.get("/bootstrap", response_model=BootstrapResponse)
def bootstrap(db: Annotated[Session, Depends(get_db)]) -> BootstrapResponse:
    model = _model_info()
    try:
        artifact = load_metrics()
    except ArtifactUnavailable:
        artifact = None

    transactions, _ = list_transactions(
        db,
        decision=None,
        actual_label=None,
        minimum_risk=None,
        maximum_risk=None,
        limit=25,
        cursor=None,
    )
    reviews = list_reviews(db)
    if artifact is None:
        return BootstrapResponse(
            status="ok",
            evaluated=False,
            generated_at=None,
            dataset={"name": "IEEE-CIS Fraud Detection", "available": False, "validation_status": "Not evaluated yet"},
            model=model,
            metrics=None,
            decision_distribution=None,
            confusion_matrix=None,
            transactions=transactions,
            reviews=reviews,
            rules={"active_count": 0, "evidence_status": "Awaiting validation error analysis"},
            provenance="Not evaluated yet",
        )

    return BootstrapResponse(
        status="ok",
        evaluated=True,
        generated_at=artifact.get("generated_at"),
        dataset={"name": "IEEE-CIS Fraud Detection", "available": True, "validation_status": "VALID"},
        model=model,
        metrics={
            "transactions_evaluated": artifact["test_transaction_count"],
            "fraud_cases": artifact["fraud_count"],
            "precision": artifact["precision"],
            "recall": artifact["recall"],
            "f1": artifact["f1"],
            "average_precision": artifact["average_precision"],
            "false_positives": artifact["false_positives"],
            "false_negatives": artifact["false_negatives"],
            "estimated_total_cost": artifact["total_estimated_cost"],
        },
        decision_distribution=artifact.get("decision_distribution"),
        confusion_matrix=artifact.get("confusion_matrix"),
        transactions=transactions,
        reviews=reviews,
        rules={"active_count": int(artifact.get("active_rule_count", 0)), "evidence_status": "Validation-derived rules only"},
        provenance="Calculated from the held-out temporal test set.",
    )
