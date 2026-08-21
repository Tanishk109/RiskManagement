from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas.risk import ScoreRequest, ScoreResponse
from ..services.artifacts import ArtifactUnavailable
from ..services.scoring import score_transaction

router = APIRouter(prefix="/api/v1/score", tags=["scoring"])


@router.post("", response_model=ScoreResponse)
def score(payload: ScoreRequest, db: Annotated[Session, Depends(get_db)]) -> ScoreResponse:
    try:
        return score_transaction(payload, db)
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="transaction_id already exists") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
