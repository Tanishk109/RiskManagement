from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..database import get_db

router = APIRouter(tags=["health"])
DatabaseDependency = Annotated[Session, Depends(get_db)]


@router.get("/health/db")
def database_health(db: DatabaseDependency) -> dict[str, str]:
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="Operational PostgreSQL is unavailable.",
        ) from exc
    return {"status": "ok", "database": "postgresql"}
