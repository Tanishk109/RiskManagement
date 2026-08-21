from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..models import CostConfig, ModelRun, ThresholdConfig
from .artifacts import ArtifactUnavailable, load_metrics, load_model_metadata

COST_FIELDS = (
    "currency",
    "fraud_loss_fraction",
    "chargeback_fixed_cost",
    "legitimate_margin_rate",
    "false_positive_fixed_cost",
    "manual_review_cost",
    "review_fraud_catch_rate",
    "review_legitimate_approval_rate",
)
METRIC_FIELDS = (
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
    "active_rule_count",
)
DECIMAL_METRICS = {
    "false_positive_estimated_cost",
    "false_negative_estimated_cost",
    "review_cost",
    "total_estimated_cost",
}


def parse_datetime(value: Any, *, fallback: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return fallback or datetime.now(timezone.utc)


def _cost_payload(values: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "currency": "INR",
        "fraud_loss_fraction": 1.0,
        "chargeback_fixed_cost": 0.0,
        "legitimate_margin_rate": 0.2,
        "false_positive_fixed_cost": 0.0,
        "manual_review_cost": 150.0,
        "review_fraud_catch_rate": 0.9,
        "review_legitimate_approval_rate": 0.98,
    }
    return {field: values.get(field, defaults[field]) for field in COST_FIELDS}


def upsert_cost_config(db: Session, values: dict[str, Any], *, name_prefix: str) -> CostConfig:
    payload = _cost_payload(values)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    config_key = f"cost-{digest}"
    row = db.scalar(select(CostConfig).where(CostConfig.config_key == config_key))
    if row is not None:
        return row
    row = CostConfig(
        config_key=config_key,
        name=f"{name_prefix}-{digest[:8]}",
        currency=str(payload["currency"]).upper(),
        fraud_loss_fraction=float(payload["fraud_loss_fraction"]),
        chargeback_fixed_cost=Decimal(str(payload["chargeback_fixed_cost"])),
        legitimate_margin_rate=float(payload["legitimate_margin_rate"]),
        false_positive_fixed_cost=Decimal(str(payload["false_positive_fixed_cost"])),
        manual_review_cost=Decimal(str(payload["manual_review_cost"])),
        review_fraud_catch_rate=float(payload["review_fraud_catch_rate"]),
        review_legitimate_approval_rate=float(payload["review_legitimate_approval_rate"]),
    )
    db.add(row)
    db.flush()
    return row


def upsert_runtime_evidence(
    db: Session,
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any] | None,
) -> tuple[ModelRun, ThresholdConfig]:
    version = str(metadata["model_version"])
    model_run = db.scalar(select(ModelRun).where(ModelRun.model_version == version))
    if model_run is None:
        model_run = ModelRun(
            model_name=str(metadata["model_name"]),
            model_version=version,
            trained_at=parse_datetime(metadata.get("trained_at")),
            feature_set=str(metadata["feature_set"]),
            evaluation_status="NOT_EVALUATED",
            active_rule_count=0,
            metadata_json={},
        )
        db.add(model_run)
        db.flush()

    model_run.model_name = str(metadata["model_name"])
    model_run.trained_at = parse_datetime(metadata.get("trained_at"), fallback=model_run.trained_at)
    model_run.feature_set = str(metadata["feature_set"])
    model_run.metadata_json = {
        key: metadata[key]
        for key in (
            "feature_names",
            "training_split",
            "selection_split",
            "evaluation_split",
            "calibration",
        )
        if key in metadata
    }

    if metrics is not None and metrics.get("evaluation_status") == "complete":
        model_run.evaluation_status = "COMPLETE"
        model_run.evaluation_split = str(metrics.get("split", "test"))
        model_run.evaluated_at = parse_datetime(metrics.get("generated_at"))
        for field in METRIC_FIELDS:
            value = metrics.get(field)
            if value is not None:
                setattr(model_run, field, Decimal(str(value)) if field in DECIMAL_METRICS else value)
        if "metric_definition" in metrics:
            model_run.metadata_json["metric_definition"] = metrics["metric_definition"]

    assumptions = dict(metrics.get("business_assumptions") or {}) if metrics else {}
    cost_config = upsert_cost_config(db, assumptions, name_prefix="evaluation") if assumptions else None
    thresholds = dict(metadata.get("thresholds") or {})
    if "review" not in thresholds or "block" not in thresholds:
        raise ValueError("Model metadata does not contain frozen review and block thresholds")
    config_key = str(metadata.get("threshold_config_id") or f"{version}-thresholds")
    threshold = db.scalar(select(ThresholdConfig).where(ThresholdConfig.config_key == config_key))
    if threshold is None:
        threshold = ThresholdConfig(
            config_key=config_key,
            model_run_id=model_run.id,
            review_threshold=float(thresholds["review"]),
            block_threshold=float(thresholds["block"]),
            selection_split=str(metadata.get("selection_split", "validation")),
            objective="lowest estimated cost under configured assumptions",
            is_active=True,
        )
        db.add(threshold)
    threshold.model_run_id = model_run.id
    threshold.cost_config_id = cost_config.id if cost_config else threshold.cost_config_id
    threshold.review_threshold = float(thresholds["review"])
    threshold.block_threshold = float(thresholds["block"])
    threshold.is_active = True
    for other in db.scalars(
        select(ThresholdConfig).where(
            ThresholdConfig.model_run_id == model_run.id,
            ThresholdConfig.config_key != config_key,
        )
    ):
        other.is_active = False
    db.flush()
    return model_run, threshold


def sync_evidence_artifacts(db: Session, *, require_metrics: bool = False) -> tuple[ModelRun, ThresholdConfig]:
    metadata = load_model_metadata()
    try:
        metrics = load_metrics()
    except ArtifactUnavailable:
        if require_metrics:
            raise
        metrics = None
    result = upsert_runtime_evidence(db, metadata=metadata, metrics=metrics)
    db.commit()
    return result


def latest_model_run(db: Session) -> ModelRun | None:
    return db.scalar(
        select(ModelRun)
        .options(selectinload(ModelRun.threshold_configs))
        .order_by(ModelRun.trained_at.desc(), ModelRun.id.desc())
        .limit(1)
    )


def active_threshold(model_run: ModelRun) -> ThresholdConfig | None:
    active = [config for config in model_run.threshold_configs if config.is_active]
    if active:
        return max(active, key=lambda config: config.id)
    return max(model_run.threshold_configs, key=lambda config: config.id) if model_run.threshold_configs else None
