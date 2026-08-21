from __future__ import annotations

from fastapi import APIRouter

from ..schemas.risk import CostSimulationRequest, CostSimulationResponse
from ..services.artifacts import ArtifactUnavailable
from ..services.cost_service import simulate_from_held_out

router = APIRouter(prefix="/api/v1/cost", tags=["cost"])


@router.post("/simulate", response_model=CostSimulationResponse)
def simulate(payload: CostSimulationRequest) -> CostSimulationResponse:
    try:
        return simulate_from_held_out(payload)
    except ArtifactUnavailable as exc:
        return CostSimulationResponse(
            evaluated=False,
            provenance=f"Not evaluated yet — {exc}",
            current=None,
            proposed=None,
        )
