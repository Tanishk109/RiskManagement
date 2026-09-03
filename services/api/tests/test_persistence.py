from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from app.config import (
    operational_database_url,
    required_database_url,
    validate_database_transport,
)
from app.models import CostConfig, CostSimulation, ReviewCase, RuleHit, Transaction
from app.schemas.risk import CostSimulationRequest, Factor, ReviewDecisionRequest
from app.services.cost_service import simulate_from_held_out
from app.services.evidence_store import upsert_runtime_evidence
from app.services.repository import decide_review, persist_scored_transaction
from app.services.rules_engine import RuleHit as EvaluatedRuleHit
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError


def test_operational_database_is_postgresql_only():
    assert operational_database_url("postgres://user:pass@example/db") == (
        "postgresql+psycopg://user:pass@example/db"
    )
    assert operational_database_url("postgresql://user:pass@example/db") == (
        "postgresql+psycopg://user:pass@example/db"
    )
    with pytest.raises(ValueError, match="PostgreSQL"):
        operational_database_url("sqlite:///merchantshield.db")


def test_database_url_is_required_and_has_no_python_fallback(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL is required"):
        required_database_url()


def test_production_database_url_requires_tls():
    with pytest.raises(ValueError, match="must require TLS"):
        validate_database_transport(
            "postgresql://user:password@provider.example/merchantshield",
            "production",
        )
    validate_database_transport(
        "postgresql://user:password@provider.example/merchantshield?sslmode=require",
        "production",
    )


def test_migration_history_has_one_current_head():
    config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
    script = ScriptDirectory.from_config(config)
    assert script.get_heads() == ["0005_return_predictions"]


def _evidence(db):
    metadata = {
        "model_name": "XGBoost",
        "model_version": "merchantshield-test-v1",
        "trained_at": "2026-08-20T10:00:00+00:00",
        "feature_set": "expanded",
        "feature_names": ["TransactionAmt", "V17"],
        "training_split": "first 70% by TransactionDT",
        "selection_split": "validation",
        "evaluation_split": "test",
        "thresholds": {"review": 0.4, "block": 0.8},
        "threshold_config_id": "validation-cost-test-v1",
    }
    metrics = {
        "evaluation_status": "complete",
        "generated_at": "2026-08-20T12:00:00+00:00",
        "split": "test",
        "test_transaction_count": 4,
        "fraud_count": 2,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
        "average_precision": 0.8,
        "roc_auc": 0.75,
        "brier_score": 0.2,
        "true_positives": 1,
        "false_positives": 0,
        "true_negatives": 2,
        "false_negatives": 1,
        "approve_count": 2,
        "review_count": 1,
        "block_count": 1,
        "false_positive_estimated_cost": 0,
        "false_negative_estimated_cost": 500,
        "review_cost": 150,
        "total_estimated_cost": 650,
        "active_rule_count": 1,
        "business_assumptions": {
            "currency": "INR",
            "fraud_loss_fraction": 1.0,
            "chargeback_fixed_cost": 0,
            "legitimate_margin_rate": 0.2,
            "false_positive_fixed_cost": 0,
            "manual_review_cost": 150,
            "review_fraud_catch_rate": 0.9,
            "review_legitimate_approval_rate": 0.98,
        },
    }
    return upsert_runtime_evidence(db, metadata=metadata, metrics=metrics)


def test_model_metrics_thresholds_and_costs_are_relational(db):
    model_run, threshold = _evidence(db)
    db.commit()

    assert model_run.evaluation_status == "COMPLETE"
    assert model_run.false_positives == 0
    assert model_run.metadata_json["feature_names"] == ["TransactionAmt", "V17"]
    assert threshold.review_threshold == 0.4
    assert threshold.cost_config_id is not None
    assert db.scalar(select(func.count()).select_from(CostConfig)) == 1


def test_prediction_reasons_and_rule_hits_are_separate_rows(db):
    model_run, threshold = _evidence(db)
    transaction = persist_scored_transaction(
        db,
        transaction_id="runtime-1",
        transaction_dt=100,
        amount=2500,
        risk_score=0.61,
        decision="REVIEW",
        model_run=model_run,
        threshold_config=threshold,
        rule_hits=[EvaluatedRuleHit(rule_id="velocity-review", action="REVIEW", reason="Validation rule")],
        factors=[Factor(feature_name="V17", feature_value=2.5, contribution=0.42)],
    )

    assert transaction.review_case is not None
    assert db.scalar(select(func.count()).select_from(RuleHit)) == 1
    assert db.scalar(select(func.count()).select_from(Transaction)) == 1
    assert transaction.reasons[0].rank == 1


def test_cost_simulation_history_is_persisted(db, tmp_path, monkeypatch):
    _evidence(db)
    db.commit()
    predictions = tmp_path / "held-out.csv"
    pd.DataFrame(
        {
            "TransactionAmt": [100.0, 200.0, 300.0, 400.0],
            "isFraud": [0, 1, 0, 1],
            "risk_score": [0.1, 0.5, 0.7, 0.9],
        }
    ).to_csv(predictions, index=False)
    monkeypatch.setattr(
        "app.services.cost_service.get_settings",
        lambda: SimpleNamespace(predictions_path=predictions),
    )

    result = simulate_from_held_out(
        CostSimulationRequest(review_threshold=0.3, block_threshold=0.75),
        db,
    )

    assert result.evaluated is True
    assert result.simulation_group_id
    assert db.scalar(select(func.count()).select_from(CostSimulation)) == 2


def test_review_write_failure_rolls_back_without_changing_decision(
    db, seeded_review, monkeypatch
):
    rollback_calls = 0
    real_rollback = db.rollback

    def fail_commit():
        raise SQLAlchemyError("simulated PostgreSQL write failure")

    def tracked_rollback():
        nonlocal rollback_calls
        rollback_calls += 1
        real_rollback()

    monkeypatch.setattr(db, "commit", fail_commit)
    monkeypatch.setattr(db, "rollback", tracked_rollback)

    with pytest.raises(SQLAlchemyError, match="simulated PostgreSQL write failure"):
        decide_review(
            db,
            seeded_review,
            ReviewDecisionRequest(
                decision="APPROVE",
                reason="This write must roll back.",
            ),
        )

    db.expire_all()
    review = db.get(ReviewCase, seeded_review)
    assert rollback_calls == 1
    assert review is not None
    assert review.status == "OPEN"
    assert review.reviewer_decision is None


def test_operational_schema_contains_no_training_dataset_tables():
    table_names = set(Transaction.metadata.tables)
    forbidden_fragments = ("ieee", "online_retail", "training_dataset", "train_transaction")
    assert not any(fragment in table for fragment in forbidden_fragments for table in table_names)
