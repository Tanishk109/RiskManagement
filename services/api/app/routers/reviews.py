from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas.risk import ReviewDecisionRequest, ReviewOut
from ..services.repository import decide_review, list_reviews

router = APIRouter(prefix="/api/v1/reviews", tags=["reviews"])


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
