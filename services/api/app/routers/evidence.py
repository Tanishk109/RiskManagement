from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import ModelRun
from ..schemas.risk import BootstrapResponse, MetricsResponse, ModelInfo
from ..services.evidence_store import active_threshold, latest_model_run
from ..services.repository import list_reviews, list_transactions

router = APIRouter(prefix="/api/v1", tags=["evidence"])


def _model_info(model_run: ModelRun | None) -> ModelInfo:
    if model_run is None:
        return ModelInfo(available=False)
    threshold = active_threshold(model_run)
    return ModelInfo(
        available=True,
        name=model_run.model_name,
        version=model_run.model_version,
        trained_at=model_run.trained_at,
        feature_set=model_run.feature_set,
        thresholds=(
            {"review": threshold.review_threshold, "block": threshold.block_threshold}
            if threshold
            else None
        ),
    )


def _public_metrics(model_run: ModelRun) -> dict[str, float | int]:
    fields = (
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
    )
    output: dict[str, float | int] = {}
    for field in fields:
        value = getattr(model_run, field)
        if value is not None:
            output[field] = float(value) if field.endswith("cost") else value
    threshold = active_threshold(model_run)
    if threshold:
        output["review_threshold"] = threshold.review_threshold
        output["block_threshold"] = threshold.block_threshold
    return output


@router.get("/model", response_model=ModelInfo)
def model_info(db: Annotated[Session, Depends(get_db)]) -> ModelInfo:
    return _model_info(latest_model_run(db))


@router.get("/metrics/summary", response_model=MetricsResponse)
def metrics_summary(db: Annotated[Session, Depends(get_db)]) -> MetricsResponse:
    model_run = latest_model_run(db)
    if model_run is None or model_run.evaluation_status != "COMPLETE":
        return MetricsResponse(evaluated=False, provenance="Not evaluated yet")
    return MetricsResponse(
        evaluated=True,
        provenance="Calculated from the held-out temporal test set and stored in PostgreSQL.",
        generated_at=model_run.evaluated_at,
        metrics=_public_metrics(model_run),
    )


@router.get("/bootstrap", response_model=BootstrapResponse)
def bootstrap(db: Annotated[Session, Depends(get_db)]) -> BootstrapResponse:
    model_run = latest_model_run(db)
    model = _model_info(model_run)
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
    if model_run is None or model_run.evaluation_status != "COMPLETE":
        return BootstrapResponse(
            status="ok",
            evaluated=False,
            generated_at=None,
            dataset={
                "name": "IEEE-CIS Fraud Detection",
                "available": False,
                "validation_status": "Not evaluated yet",
            },
            model=model,
            metrics=None,
            decision_distribution=None,
            confusion_matrix=None,
            transactions=transactions,
            reviews=reviews,
            rules={"active_count": 0, "evidence_status": "Awaiting validation error analysis"},
            provenance="Not evaluated yet",
        )

    total = model_run.test_transaction_count or 0
    decision_counts = {
        "approve": model_run.approve_count or 0,
        "review": model_run.review_count or 0,
        "block": model_run.block_count or 0,
    }
    distribution = {
        name: {"count": count, "share": count / total if total else 0.0}
        for name, count in decision_counts.items()
    }
    public = _public_metrics(model_run)
    return BootstrapResponse(
        status="ok",
        evaluated=True,
        generated_at=model_run.evaluated_at,
        dataset={"name": "IEEE-CIS Fraud Detection", "available": True, "validation_status": "VALID"},
        model=model,
        metrics={
            "transactions_evaluated": public["test_transaction_count"],
            "fraud_cases": public["fraud_count"],
            "precision": public["precision"],
            "recall": public["recall"],
            "f1": public["f1"],
            "average_precision": public["average_precision"],
            "false_positives": public["false_positives"],
            "false_negatives": public["false_negatives"],
            "estimated_total_cost": public["total_estimated_cost"],
        },
        decision_distribution=distribution,
        confusion_matrix={
            "true_positives": model_run.true_positives or 0,
            "false_positives": model_run.false_positives or 0,
            "true_negatives": model_run.true_negatives or 0,
            "false_negatives": model_run.false_negatives or 0,
        },
        transactions=transactions,
        reviews=reviews,
        rules={
            "active_count": model_run.active_rule_count,
            "evidence_status": "Validation-derived rules only",
        },
        provenance="Calculated from the held-out temporal test set and stored in PostgreSQL.",
    )
