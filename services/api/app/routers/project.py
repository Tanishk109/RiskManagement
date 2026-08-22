from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from ..schemas.project import (
    FeatureImportanceResponse,
    InterestingCasesResponse,
    ModelComparisonResponse,
    ProjectStatusResponse,
    ValidationTransactionPage,
)
from ..services.artifacts import ArtifactUnavailable
from ..services.project_artifacts import (
    ProjectArtifactService,
    ValidationFilter,
    get_project_artifact_service,
)
from ..services.validation_cost import (
    ValidationCostService,
    get_validation_cost_service,
)

router = APIRouter(prefix="/api/v1", tags=["project evidence"])

ArtifactServiceDependency = Annotated[ProjectArtifactService, Depends(get_project_artifact_service)]
CostServiceDependency = Annotated[ValidationCostService, Depends(get_validation_cost_service)]


def _unavailable(exc: ArtifactUnavailable) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


@router.get("/project/status", response_model=ProjectStatusResponse)
def project_status(service: ArtifactServiceDependency) -> dict[str, object]:
    try:
        return service.project_status()
    except ArtifactUnavailable as exc:
        raise _unavailable(exc) from exc


@router.get("/model-comparison", response_model=ModelComparisonResponse)
def model_comparison(service: ArtifactServiceDependency) -> dict[str, object]:
    try:
        return service.model_comparison()
    except ArtifactUnavailable as exc:
        raise _unavailable(exc) from exc


@router.get("/model/feature-importance", response_model=FeatureImportanceResponse)
def feature_importance(
    service: ArtifactServiceDependency,
    limit: Annotated[int, Query(ge=1, le=15)] = 13,
) -> dict[str, object]:
    try:
        return service.feature_importance(limit)
    except ArtifactUnavailable as exc:
        raise _unavailable(exc) from exc


@router.get("/validation/transactions", response_model=ValidationTransactionPage)
def validation_transactions(
    service: ArtifactServiceDependency,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
    filter_name: Annotated[
        ValidationFilter,
        Query(alias="filter"),
    ] = "all",
    search: Annotated[str | None, Query(min_length=1, max_length=80)] = None,
) -> dict[str, object]:
    try:
        return service.validation_transactions(
            page=page,
            page_size=page_size,
            filter_name=filter_name,
            search=search,
        )
    except ArtifactUnavailable as exc:
        raise _unavailable(exc) from exc


@router.get("/validation/interesting-cases", response_model=InterestingCasesResponse)
def interesting_cases(service: ArtifactServiceDependency) -> dict[str, object]:
    try:
        return service.interesting_cases()
    except ArtifactUnavailable as exc:
        raise _unavailable(exc) from exc


@router.get("/validation/residual-risk")
def validation_residual_risk(service: CostServiceDependency) -> dict[str, object]:
    try:
        return service.residual_risk()
    except ArtifactUnavailable as exc:
        raise _unavailable(exc) from exc
