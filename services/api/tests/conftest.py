from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal

import pytest

# Keep API tests deterministic even when a developer's local .env is configured
# for production Render/Neon deployment.
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = (
    "postgresql+psycopg://merchantshield_test:merchantshield_test@localhost:5432/merchantshield_test"
)

from app.database import Base, get_db
from app.main import app
from app.models import (
    ModelRun,
    PredictionReason,
    ReviewCase,
    ThresholdConfig,
    Transaction,
)
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

TEST_ENGINE = create_engine(
    "sqlite+pysqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSession = sessionmaker(bind=TEST_ENGINE, expire_on_commit=False)


def override_get_db():
    with TestingSession() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def reset_database():
    Base.metadata.drop_all(TEST_ENGINE)
    Base.metadata.create_all(TEST_ENGINE)
    yield


@pytest.fixture()
def db() -> Session:
    with TestingSession() as session:
        yield session


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def seeded_review(db: Session) -> int:
    model_run = ModelRun(
        model_name="Fixture classifier",
        model_version="fixture-software-test-only",
        trained_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        feature_set="fixture",
        evaluation_status="NOT_EVALUATED",
        active_rule_count=0,
        metadata_json={"fixture": True},
    )
    db.add(model_run)
    db.flush()
    threshold = ThresholdConfig(
        config_key="fixture-thresholds",
        model_run_id=model_run.id,
        review_threshold=0.4,
        block_threshold=0.8,
        selection_split="validation",
        objective="software fixture only",
        is_active=True,
    )
    db.add(threshold)
    db.flush()
    transaction = Transaction(
        transaction_id="fixture-review-1",
        transaction_dt=120,
        amount=Decimal("2500.00"),
        actual_label=1,
        risk_score=0.62,
        decision="REVIEW",
        model_run_id=model_run.id,
        threshold_config_id=threshold.id,
        source="TEST_FIXTURE",
    )
    transaction.reasons = [
        PredictionReason(rank=1, feature_name="V17", feature_value=None, contribution=0.42)
    ]
    transaction.review_case = ReviewCase(status="OPEN", model_decision="REVIEW")
    db.add(transaction)
    db.commit()
    return transaction.review_case.id
