from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import models  # noqa: F401
from .config import get_settings
from .routers import (
    chargebacks,
    cost,
    evidence,
    fraud_pulse,
    project,
    reviews,
    scoring,
    transactions,
)

settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Defense-only, cost-aware fraud decisions with honest evidence provenance.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(evidence.router)
app.include_router(project.router)
app.include_router(transactions.router)
app.include_router(reviews.router)
app.include_router(scoring.router)
app.include_router(cost.router)
app.include_router(chargebacks.router)
app.include_router(fraud_pulse.router)
