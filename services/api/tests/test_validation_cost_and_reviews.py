from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from app.main import app
from app.models import CostConfig, ModelRun, ReviewCase, ThresholdConfig, Transaction
from app.services.project_artifacts import get_project_artifact_service
from app.services.validation_cost import (
    ValidationCostService,
    get_validation_cost_service,
)
from merchantshield_ml.cost import CostAssumptions, simulate_cost
from sqlalchemy import select


@pytest.fixture()
def validation_cost_service(tmp_path) -> ValidationCostService:
    labels = np.array([0, 1, 0, 1])
    amounts = np.array([10.0, 50.0, 20.0, 900.0])
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    frame = pd.DataFrame(
        {
            "TransactionID": [101, 102, 103, 104],
            "TransactionDT": [7, 7, 8, 8],
            "actual_label": labels,
            "TransactionAmt": amounts,
            "fraud_probability": scores,
            "ProductCD": ["W", "C", "S", "W"],
            "card4": ["visa", "visa", "discover", "mastercard"],
            "card6": ["debit", "credit", "credit", "debit"],
            "C1": [1, 2, 3, 4],
            "C2": [1, 2, 3, 4],
            "C3": [1, 2, 3, 4],
            "C4": [1, 2, 3, 4],
            "C5": [1, 2, 3, 4],
            "D1": [1, 2, 3, 4],
            "D2": [None, 2, 3, 4],
            "D3": [1, 2, 3, None],
        }
    )
    assumptions = {
        "currency": "INR",
        "fraud_loss_fraction": 0.85,
        "chargeback_fixed_cost": 50.0,
        "legitimate_margin_rate": 0.18,
        "false_positive_fixed_cost": 20.0,
        "manual_review_cost": 25.0,
        "review_fraud_catch_rate": 0.9,
        "review_legitimate_approval_rate": 0.98,
    }
    cost_assumptions = CostAssumptions(**assumptions)
    operating_result = simulate_cost(
        labels=labels,
        amounts=amounts,
        risk_scores=scores,
        review_threshold=0.15,
        block_threshold=0.85,
        assumptions=cost_assumptions,
    )
    capacity_result = simulate_cost(
        labels=labels,
        amounts=amounts,
        risk_scores=scores,
        review_threshold=0.15,
        block_threshold=0.5,
        assumptions=cost_assumptions,
    )
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "assumption_status": "ILLUSTRATIVE MERCHANT ASSUMPTIONS",
                "cost_output_label": "ESTIMATED BUSINESS COST UNDER USER-SUPPLIED ASSUMPTIONS",
                "scenarios": {
                    "moderate": {
                        "name": "Scenario B — Moderate merchant",
                        "description": "Software-test scenario.",
                        "assumptions": assumptions,
                    }
                },
                "review_capacity_rates": [None, 0.25, 0.5],
            }
        ),
        encoding="utf-8",
    )
    operating_path = tmp_path / "operating.json"
    operating_path.write_text(
        json.dumps(
            {
                "status": "provisional_validation_config",
                "not_final": True,
                "generated_at": "2026-08-22T07:19:55+00:00",
                "model_name": "CatBoostClassifier",
                "model_version": "cost-review-test-v1",
                "experiment_id": "cb-test",
                "feature_set": "test-features",
                "selection_split": "validation",
                "held_out_test_accessed": False,
                "scenario": "moderate",
                "scenario_name": "Scenario B — Moderate merchant",
                "review_threshold": 0.15,
                "block_threshold": 0.85,
                "review_capacity_limit": 0.5,
                "selection_reason": "software test only",
                "cost_assumptions": assumptions,
                "validation_metrics": operating_result,
                "limitations": ["Software fixture; not merchant evidence."],
            }
        ),
        encoding="utf-8",
    )
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "status": "provisional_validation_analysis",
                "selection_split": "validation",
                "held_out_test_accessed": False,
                "model": {"feature_names": ["TransactionAmt"]},
                "threshold_grid": {"total_stored_rows": 2},
                "scenarios": {
                    "moderate": {
                        "assumptions": assumptions,
                        "evaluated_configuration_count": 2,
                        "lowest_estimated_cost": min(
                            [operating_result | {"review_threshold": 0.15, "block_threshold": 0.85},
                             capacity_result | {"review_threshold": 0.15, "block_threshold": 0.5}],
                            key=lambda item: item["total_estimated_cost"],
                        ),
                    }
                },
                "sensitivity_analysis": [],
                "failure_slices": {},
                "high_value_fraud": {
                    "fraud_rows": 2,
                    "approve": {"count": 1},
                    "review": {"count": 1},
                    "block": {"count": 0},
                    "highest_value_approved_fraud_examples": [],
                },
            }
        ),
        encoding="utf-8",
    )
    grid_path = tmp_path / "grid.parquet"
    pd.DataFrame(
        [
            {"analysis_type": "merchant_scenario", "scenario_id": "moderate", **operating_result,
             "review_threshold": 0.15, "block_threshold": 0.85},
            {"analysis_type": "merchant_scenario", "scenario_id": "moderate", **capacity_result,
             "review_threshold": 0.15, "block_threshold": 0.5},
        ]
    ).to_parquet(grid_path, index=False)
    project = SimpleNamespace(
        paths=SimpleNamespace(
            operating_config=operating_path,
            threshold_analysis=analysis_path,
            merchant_scenarios=scenario_path,
            threshold_grid=grid_path,
        ),
        validation_frame=frame,
        classification_threshold=0.5,
    )
    return ValidationCostService(project)


@pytest.fixture()
def validation_client(client, validation_cost_service):
    app.dependency_overrides[get_validation_cost_service] = lambda: validation_cost_service
    app.dependency_overrides[get_project_artifact_service] = lambda: validation_cost_service.project
    yield client
    app.dependency_overrides.pop(get_validation_cost_service, None)
    app.dependency_overrides.pop(get_project_artifact_service, None)


def test_scenario_loading_and_default_operating_configuration(validation_client):
    response = validation_client.get("/api/v1/cost/scenarios")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_scenario_id"] == "moderate"
    assert payload["default_review_threshold"] == 0.15
    assert payload["default_block_threshold"] == 0.85
    assert payload["held_out_test_status"] == "sealed_not_evaluated"


def test_simulation_policy_comparison_and_capacity_restore(validation_client):
    response = validation_client.post(
        "/api/v1/cost/simulate",
        json={
            "scenario_id": "moderate",
            "review_threshold": 0.15,
            "block_threshold": 0.85,
            "review_capacity": 0.25,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["transaction_count"] == 4
    assert payload["metrics"]["review_count"] == 2
    assert payload["capacity_met"] is False
    assert payload["lowest_cost_feasible"]["block_threshold"] == 0.5
    assert [item["policy"] for item in payload["policy_comparison"]] == [
        "Approve all",
        "Binary model @ 0.50",
        "Three-way policy",
    ]


def test_validation_summary_matches_saved_config(validation_client):
    response = validation_client.get("/api/v1/cost/validation-summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["review_threshold"] == 0.15
    assert payload["block_threshold"] == 0.85
    assert payload["provisional"] is True
    assert "validation predictions only" in payload["provenance"]
    residual = validation_client.get("/api/v1/validation/residual-risk")
    assert residual.status_code == 200
    assert residual.json()["held_out_test_status"] == "sealed_not_evaluated"


def test_review_band_hides_labels_and_decision_persists(validation_client, db):
    queue = validation_client.get("/api/v1/reviews/validation?order=highest_amount")
    assert queue.status_code == 200
    assert queue.json()["total"] == 2
    item = queue.json()["items"][0]
    assert item["transaction_id"] == "104"
    assert item["ground_truth"] is None
    assert "actual_label" not in item
    assert item["features"]["ProductCD"] is not None

    decision = validation_client.post(
        "/api/v1/reviews/validation/104/decision",
        json={"decision": "BLOCK", "reason": "Manual validation review."},
    )
    assert decision.status_code == 200
    assert decision.json()["reviewer_decision"] == "BLOCK"
    assert db.scalar(select(ReviewCase)).status == "DECIDED"
    assert db.scalar(select(Transaction)).source == "VALIDATION_REVIEW_DEMO"
    model_run = db.scalar(select(ModelRun))
    assert model_run.evaluation_status == "NOT_EVALUATED"
    assert model_run.average_precision is None
    assert db.scalar(select(CostConfig)) is not None
    assert db.scalar(select(ThresholdConfig)).selection_split == "validation"

    truth = validation_client.get("/api/v1/reviews/validation/104/ground-truth")
    assert truth.status_code == 200
    assert truth.json()["ground_truth"] == "FRAUD"
    assert truth.json()["reviewer_correct"] is True
    repeat = validation_client.post(
        "/api/v1/reviews/validation/104/decision", json={"decision": "APPROVE"}
    )
    assert repeat.status_code == 409


def test_missing_cost_artifact_returns_503(validation_client, validation_cost_service):
    validation_cost_service.project.paths.operating_config.unlink()
    response = validation_client.get("/api/v1/cost/validation-summary")
    assert response.status_code == 503
    assert "not available" in response.json()["detail"]
