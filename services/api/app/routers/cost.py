from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from ..schemas.project import (
    CostScenariosResponse,
    ValidationCostResponse,
    ValidationCostSimulationRequest,
)
from ..services.artifacts import ArtifactUnavailable
from ..services.validation_cost import (
    ValidationCostService,
    get_validation_cost_service,
)

router = APIRouter(prefix="/api/v1/cost", tags=["validation cost"])

CostServiceDependency = Annotated[ValidationCostService, Depends(get_validation_cost_service)]


def _unavailable(exc: ArtifactUnavailable) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


@router.get("/scenarios", response_model=CostScenariosResponse)
def scenarios(service: CostServiceDependency) -> dict[str, Any]:
    try:
        return service.list_scenarios()
    except ArtifactUnavailable as exc:
        raise _unavailable(exc) from exc


@router.get("/validation-summary")
def validation_summary(service: CostServiceDependency) -> dict[str, Any]:
    try:
        return service.summary()
    except ArtifactUnavailable as exc:
        raise _unavailable(exc) from exc


@router.post("/simulate", response_model=ValidationCostResponse)
def simulate(
    payload: ValidationCostSimulationRequest,
    service: CostServiceDependency,
) -> dict[str, Any]:
    try:
        return service.simulate(
            scenario_id=payload.scenario_id,
            review_threshold=payload.review_threshold,
            block_threshold=payload.block_threshold,
            review_capacity=payload.review_capacity,
        )
    except ArtifactUnavailable as exc:
        raise _unavailable(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/residual-risk")
def residual_risk(service: CostServiceDependency) -> dict[str, Any]:
    try:
        return service.residual_risk()
    except ArtifactUnavailable as exc:
        raise _unavailable(exc) from exc
