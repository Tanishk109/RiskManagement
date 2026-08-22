from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ..schemas.risk import (
    BatchScoreRequest,
    BatchScoreResponse,
    ScoreRequest,
    ScoreResponse,
)
from ..services.artifacts import ArtifactUnavailable
from ..services.validation_scoring import (
    ValidationScoringService,
    get_validation_scoring_service,
)

router = APIRouter(prefix="/api/v1/score", tags=["scoring"])
ScoringServiceDependency = Annotated[
    ValidationScoringService, Depends(get_validation_scoring_service)
]


@router.post("", response_model=ScoreResponse)
def score(payload: ScoreRequest, service: ScoringServiceDependency) -> ScoreResponse:
    try:
        return service.score(payload)
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/batch", response_model=BatchScoreResponse)
def score_batch(
    payload: BatchScoreRequest, service: ScoringServiceDependency
) -> BatchScoreResponse:
    try:
        return service.score_batch(payload)
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
