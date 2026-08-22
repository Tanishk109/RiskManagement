from __future__ import annotations

from sqlalchemy.orm import Session

from ..schemas.risk import ScoreRequest, ScoreResponse
from .validation_scoring import get_validation_scoring_service


def score_transaction(payload: ScoreRequest, _db: Session) -> ScoreResponse:
    """Compatibility entry point for inference-only validation scoring."""

    return get_validation_scoring_service().score(payload)
