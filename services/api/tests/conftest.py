from __future__ import annotations

from decimal import Decimal

import pytest
from app.database import Base, get_db
from app.main import app
from app.models import PredictionReason, ReviewCase, Transaction
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
    transaction = Transaction(
        transaction_id="fixture-review-1",
        transaction_dt=120,
        amount=Decimal("2500.00"),
        actual_label=1,
        risk_score=0.62,
        model_version="fixture-software-test-only",
        decision="REVIEW",
        rules_triggered=[],
        feature_payload={},
    )
    transaction.reasons = [PredictionReason(feature_name="V17", feature_value=None, contribution=0.42)]
    transaction.review_case = ReviewCase(status="OPEN", model_decision="REVIEW")
    db.add(transaction)
    db.commit()
    return transaction.review_case.id
