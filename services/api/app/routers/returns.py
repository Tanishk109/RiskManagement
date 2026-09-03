from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ..schemas.returns import (
    ReturnBatchRequest,
    ReturnBatchResponse,
    ReturnOrderInput,
    ReturnScoreResult,
    ReturnStatus,
)
from ..services.artifacts import ArtifactUnavailable
from ..services.returns_scoring import ReturnRiskService, get_return_risk_service

router = APIRouter(prefix="/api/v1/returns", tags=["returns"])
ReturnServiceDependency = Annotated[ReturnRiskService, Depends(get_return_risk_service)]


@router.get("/status", response_model=ReturnStatus)
def return_status(service: ReturnServiceDependency) -> ReturnStatus:
    try:
        return service.status()
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/score", response_model=ReturnScoreResult)
def score_return_order(
    request: ReturnOrderInput,
    service: ReturnServiceDependency,
) -> ReturnScoreResult:
    try:
        return service.score_one(request)
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/score/batch", response_model=ReturnBatchResponse)
def score_return_batch(
    request: ReturnBatchRequest,
    service: ReturnServiceDependency,
) -> ReturnBatchResponse:
    try:
        return service.score_many(request.rows)
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
