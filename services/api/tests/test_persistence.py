from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from app.config import operational_database_url
from app.models import CostConfig, CostSimulation, RuleHit, Transaction
from app.schemas.risk import CostSimulationRequest, Factor
from app.services.cost_service import simulate_from_held_out
from app.services.evidence_store import upsert_runtime_evidence
from app.services.repository import persist_scored_transaction
from app.services.rules_engine import RuleHit as EvaluatedRuleHit
from sqlalchemy import func, select


def test_operational_database_is_postgresql_only():
    assert operational_database_url("postgres://user:pass@example/db") == (
        "postgresql+psycopg://user:pass@example/db"
    )
    assert operational_database_url("postgresql://user:pass@example/db") == (
        "postgresql+psycopg://user:pass@example/db"
    )
    with pytest.raises(ValueError, match="PostgreSQL"):
        operational_database_url("sqlite:///merchantshield.db")


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
