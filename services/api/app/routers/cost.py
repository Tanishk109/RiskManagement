from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas.risk import CostSimulationRequest, CostSimulationResponse
from ..services.artifacts import ArtifactUnavailable
from ..services.cost_service import simulate_from_held_out

router = APIRouter(prefix="/api/v1/cost", tags=["cost"])


@router.post("/simulate", response_model=CostSimulationResponse)
def simulate(
    payload: CostSimulationRequest,
    db: Annotated[Session, Depends(get_db)],
) -> CostSimulationResponse:
    try:
        return simulate_from_held_out(payload, db)
    except ArtifactUnavailable as exc:
        return CostSimulationResponse(
            evaluated=False,
            provenance=f"Not evaluated yet — {exc}",
            current=None,
            proposed=None,
        )
