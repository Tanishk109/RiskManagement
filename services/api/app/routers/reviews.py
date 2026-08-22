from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas.risk import ReviewDecisionRequest, ReviewOut
from ..services.artifacts import ArtifactUnavailable
from ..services.project_artifacts import (
    ProjectArtifactService,
    get_project_artifact_service,
)
from ..services.repository import decide_review, list_reviews
from ..services.validation_cost import (
    ValidationCostService,
    get_validation_cost_service,
)
from ..services.validation_reviews import ReviewOrder, ValidationReviewService

router = APIRouter(prefix="/api/v1/reviews", tags=["reviews"])

ProjectDependency = Annotated[ProjectArtifactService, Depends(get_project_artifact_service)]
CostDependency = Annotated[ValidationCostService, Depends(get_validation_cost_service)]


@router.get("", response_model=list[ReviewOut])
def reviews(db: Annotated[Session, Depends(get_db)], status: str | None = "OPEN") -> list[ReviewOut]:
    return list_reviews(db, status=status)


@router.post("/{review_id}/decision", response_model=ReviewOut)
def submit_review(review_id: int, payload: ReviewDecisionRequest, db: Annotated[Session, Depends(get_db)]) -> ReviewOut:
    try:
        result = decide_review(db, review_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Review case not found")
    return result


def _validation_service(
    project: ProjectArtifactService, cost: ValidationCostService
) -> ValidationReviewService:
    return ValidationReviewService(project, cost)


@router.get("/validation")
def validation_reviews(
    db: Annotated[Session, Depends(get_db)],
    project: ProjectDependency,
    cost: CostDependency,
    order: Annotated[ReviewOrder, Query()] = "highest_amount",
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
) -> dict[str, object]:
    try:
        return _validation_service(project, cost).list_reviews(
            db, order=order, page=page, page_size=page_size
        )
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/validation/{transaction_id}/ground-truth")
def reveal_validation_ground_truth(
    transaction_id: str,
    db: Annotated[Session, Depends(get_db)],
    project: ProjectDependency,
    cost: CostDependency,
) -> dict[str, object]:
    try:
        return _validation_service(project, cost).reveal_ground_truth(db, transaction_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Validation review case not found") from exc
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/validation/{transaction_id}/decision")
def decide_validation_review(
    transaction_id: str,
    payload: ReviewDecisionRequest,
    db: Annotated[Session, Depends(get_db)],
    project: ProjectDependency,
    cost: CostDependency,
) -> dict[str, object]:
    try:
        return _validation_service(project, cost).decide(
            db, transaction_id=transaction_id, payload=payload
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Validation review case not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
