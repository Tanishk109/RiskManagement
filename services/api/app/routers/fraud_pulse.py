from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ..schemas.fraud_pulse import (
    PulseReplayRequest,
    PulseResponse,
    PulseStatus,
    PulseUploadRequest,
)
from ..services.artifacts import ArtifactUnavailable
from ..services.fraud_pulse import FraudPulseService, get_fraud_pulse_service
from ..services.validation_scoring import FEATURE_SCHEMA

router = APIRouter(prefix="/api/v1/fraud-pulse", tags=["fraud-pulse"])
PulseServiceDependency = Annotated[FraudPulseService, Depends(get_fraud_pulse_service)]


@router.get("/status", response_model=PulseStatus)
def pulse_status(service: PulseServiceDependency) -> PulseStatus:
    try:
        config = service.scoring.operating_config
        return PulseStatus(
            module="Fraud-Spike Detector",
            data_source="Frozen CatBoost validation probabilities joined to chronological IEEE-CIS validation transaction fields",
            evaluation_status="Not evaluated yet",
            detector_is_classifier=False,
            model_version=str(service.scoring.metadata["model_version"]),
            review_threshold=float(config["review_threshold"]),
            block_threshold=float(config["block_threshold"]),
            methods=["rolling_zscore", "ewma", "percent_deviation"],
            metrics=["transaction_count", "mean_risk_score", "high_risk_count", "high_risk_amount"],
            upload_required_columns=["EventTime", *FEATURE_SCHEMA],
            limitations=[
                "No new classifier is trained; this detector monitors frozen fraud scores.",
                "Alerts are changes from a configured baseline, not confirmed fraud incidents.",
                "Held-out test data is not used.",
            ],
        )
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/replay", response_model=PulseResponse)
def replay_validation(request: PulseReplayRequest, service: PulseServiceDependency) -> PulseResponse:
    try:
        return service.replay_validation(request.config)
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/upload", response_model=PulseResponse)
def analyze_upload(request: PulseUploadRequest, service: PulseServiceDependency) -> PulseResponse:
    try:
        return service.analyze_upload(request.csv_content, request.config)
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
