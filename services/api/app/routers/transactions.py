from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas.risk import TransactionList, TransactionOut
from ..services.repository import get_transaction, list_transactions

router = APIRouter(prefix="/api/v1/transactions", tags=["transactions"])


@router.get("", response_model=TransactionList)
def transactions(
    db: Annotated[Session, Depends(get_db)],
    decision: Literal["APPROVE", "REVIEW", "BLOCK"] | None = None,
    actual_label: Literal[0, 1] | None = None,
    minimum_risk: float | None = Query(default=None, ge=0, le=1),
    maximum_risk: float | None = Query(default=None, ge=0, le=1),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: int | None = Query(default=None, ge=1),
) -> TransactionList:
    if minimum_risk is not None and maximum_risk is not None and minimum_risk > maximum_risk:
        raise HTTPException(status_code=422, detail="minimum_risk cannot exceed maximum_risk")
    items, next_cursor = list_transactions(
        db,
        decision=decision,
        actual_label=actual_label,
        minimum_risk=minimum_risk,
        maximum_risk=maximum_risk,
        limit=limit,
        cursor=cursor,
    )
    return TransactionList(items=items, next_cursor=next_cursor)


@router.get("/{transaction_id}", response_model=TransactionOut)
def transaction_detail(transaction_id: str, db: Annotated[Session, Depends(get_db)]) -> TransactionOut:
    result = get_transaction(db, transaction_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return result
