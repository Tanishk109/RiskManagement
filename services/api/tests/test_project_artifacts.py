from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from app.main import app
from app.services.artifacts import ArtifactUnavailable
from app.services.project_artifacts import (
    ArtifactPaths,
    ProjectArtifactService,
    get_project_artifact_service,
)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def artifact_service(tmp_path: Path) -> ProjectArtifactService:
    eda = tmp_path / "eda.json"
    split = tmp_path / "split.json"
    baseline = tmp_path / "baseline.json"
    catboost = tmp_path / "catboost.json"
    metadata = tmp_path / "metadata.json"
    experiments = tmp_path / "experiments.csv"
    importance = tmp_path / "importance.csv"
    baseline_predictions = tmp_path / "baseline_predictions.parquet"
    selected_predictions = tmp_path / "selected_predictions.parquet"
    validation = tmp_path / "validation.parquet"
    threshold = tmp_path / "threshold.json"
    final = tmp_path / "final.json"

    logistic_record = {
        "experiment_id": "lr-selected",
        "model": "Logistic Regression",
        "average_precision": 0.25,
        "roc_auc": 0.75,
        "precision": 0.5,
        "recall": 0.25,
        "f1": 1 / 3,
        "fp": 1,
        "fn": 3,
        "tp": 1,
        "tn": 3,
        "default_threshold": 0.5,
    }
    catboost_record = {
        "experiment_id": "cb-selected",
        "model": "CatBoostClassifier",
        "average_precision": 0.5,
        "roc_auc": 0.85,
        "precision": 0.75,
        "recall": 0.5,
        "f1": 0.6,
        "fp": 1,
        "fn": 1,
        "tp": 1,
        "tn": 1,
        "default_threshold": 0.5,
        "validation_rows": 4,
    }
    write_json(
        eda,
        {
            "dataset_validation": {
                "transaction_rows": 10,
                "identity_rows": 4,
                "fraud_rows": 2,
                "legitimate_rows": 8,
                "fraud_percentage": 20,
                "identity_coverage": 40,
            }
        },
    )
    write_json(
        split,
        {
            "strategy": "chronological",
            "train_rows": 6,
            "validation_rows": 2,
            "test_rows": 2,
            "train_fraction_actual": 0.6,
            "validation_fraction_actual": 0.2,
            "test_fraction_actual": 0.2,
            "train_transaction_dt_min": 1,
            "train_transaction_dt_max": 6,
            "validation_transaction_dt_min": 7,
            "validation_transaction_dt_max": 8,
            "test_transaction_dt_min": 9,
            "test_transaction_dt_max": 10,
        },
    )
    write_json(baseline, {"best_experiment": logistic_record})
    write_json(
        catboost,
        {
            "selected_candidate": catboost_record,
            "failure_slice_comparison": [
                {
                    "slice": name,
                    "fraud_support": 2,
                    "logistic_recall": 0,
                    "catboost_recall": 0.5,
                    "absolute_improvement": 0.5,
                }
                for name in (
                    "ProductCD=W",
                    "TransactionAmt>=500",
                    "card4=discover",
                    "ProductCD=S",
                )
            ],
            "selected_false_negative_amounts": {"count": 1, "total": 900, "max": 900},
        },
    )
    write_json(
        metadata,
        {
            "status": "validation_candidate",
            "model_name": "CatBoostClassifier",
            "feature_names": ["TransactionAmt", "ProductCD", "card4"],
            "class_weight": "none",
            "identity_feature_decision": {
                "selected_identity_features": False,
                "ap_loss_without_identity": 0.001,
                "reason": "Fixture identity ablation passed.",
            },
        },
    )
    pd.DataFrame(
        [
            {key: logistic_record[key] for key in ("experiment_id", "average_precision", "roc_auc", "precision", "recall", "f1")},
            {key: catboost_record[key] for key in ("experiment_id", "average_precision", "roc_auc", "precision", "recall", "f1")},
        ]
    ).to_csv(experiments, index=False)
    pd.DataFrame(
        [{"feature": "C1", "importance": 60.0}, {"feature": "D1", "importance": 40.0}]
    ).to_csv(importance, index=False)

    prediction_rows = pd.DataFrame(
        {
            "TransactionID": [101, 102, 103, 104],
            "actual_label": [0, 1, 0, 1],
            "fraud_probability": [0.1, 0.9, 0.8, 0.2],
            "predicted_label_at_0_5": [0, 1, 1, 0],
            "experiment_id": ["cb-selected"] * 4,
            "model_version": ["fixture"] * 4,
        }
    )
    prediction_rows.to_parquet(selected_predictions, index=False)
    prediction_rows.assign(experiment_id="lr-selected").to_parquet(
        baseline_predictions, index=False
    )
    validation_rows = pd.DataFrame(
        {
            "TransactionID": [101, 102, 103, 104],
            "TransactionDT": [7, 7, 8, 8],
            "isFraud": [0, 1, 0, 1],
            "TransactionAmt": [10, 50, 20, 900],
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
    validation_rows.to_parquet(validation, index=False)
    return ProjectArtifactService(
        ArtifactPaths(
            eda_summary=eda,
            split_metadata=split,
            baseline_metrics=baseline,
            catboost_metrics=catboost,
            catboost_metadata=metadata,
            experiments=experiments,
            feature_importance=importance,
            baseline_predictions=baseline_predictions,
            selected_predictions=selected_predictions,
            validation_data=validation,
            threshold_analysis=threshold,
            final_metrics=final,
        )
    )


@pytest.fixture()
def artifact_client(client, artifact_service: ProjectArtifactService):
    app.dependency_overrides[get_project_artifact_service] = lambda: artifact_service
    yield client
    app.dependency_overrides.pop(get_project_artifact_service, None)


def test_artifact_parsing_and_not_evaluated_states(artifact_service: ProjectArtifactService):
    status = artifact_service.project_status()
    assert status["dataset"]["transactions"] == 10
    assert status["dataset"]["fraud_prevalence"] == 0.2
    assert status["split"]["strategy"] == "chronological"
    assert status["threshold_analysis"]["status"] == "not_evaluated"
    assert status["operational_thresholds"]["status"] == "locked"
    assert status["final_test"] == {"status": "not_evaluated", "test_status": "sealed"}


def test_model_comparison_is_calculated_from_artifacts(artifact_service: ProjectArtifactService):
    comparison = artifact_service.model_comparison()
    assert comparison["logistic_regression"]["metrics"]["average_precision"] == 0.25
    assert comparison["catboost"]["metrics"]["average_precision"] == 0.5
    assert comparison["average_precision_relative_improvement"] == 1.0
    assert comparison["held_out_test_status"] == "sealed"
    assert len(comparison["precision_recall_curves"]) == 2


def test_feature_importance_uses_generated_order(artifact_service: ProjectArtifactService):
    result = artifact_service.feature_importance(1)
    assert result["items"] == [{"feature": "C1", "importance": 60.0}]
    assert "not causation" in result["note"]


def test_validation_transaction_pagination_and_filters(artifact_service: ProjectArtifactService):
    first = artifact_service.validation_transactions(
        page=1, page_size=2, filter_name="all", search=None
    )
    assert first["total"] == 4
    assert first["page_count"] == 2
    assert [row["transaction_id"] for row in first["items"]] == ["101", "102"]
    false_negatives = artifact_service.validation_transactions(
        page=1, page_size=25, filter_name="false_negative", search=None
    )
    assert false_negatives["total"] == 1
    assert false_negatives["items"][0]["transaction_amount"] == 900
    assert false_negatives["items"][0]["model_error"] is True


def test_interesting_cases_are_selected_programmatically(artifact_service: ProjectArtifactService):
    cases = {item["case_type"]: item for item in artifact_service.interesting_cases()["cases"]}
    assert cases["highest_value_false_negative"]["transaction_id"] == "104"
    assert cases["highest_confidence_false_positive"]["transaction_id"] == "103"
    assert cases["highest_confidence_true_fraud"]["transaction_id"] == "102"
    assert cases["highest_confidence_legitimate"]["transaction_id"] == "101"


def test_missing_artifact_is_reported(tmp_path: Path, artifact_service: ProjectArtifactService):
    missing = ProjectArtifactService(
        ArtifactPaths(**{**artifact_service.paths.__dict__, "eda_summary": tmp_path / "missing.json"})
    )
    with pytest.raises(ArtifactUnavailable, match="not available"):
        missing.project_status()


def test_missing_artifact_endpoint_returns_503(
    client, tmp_path: Path, artifact_service: ProjectArtifactService
):
    missing = ProjectArtifactService(
        ArtifactPaths(**{**artifact_service.paths.__dict__, "eda_summary": tmp_path / "missing.json"})
    )
    app.dependency_overrides[get_project_artifact_service] = lambda: missing
    try:
        response = client.get("/api/v1/project/status")
    finally:
        app.dependency_overrides.pop(get_project_artifact_service, None)
    assert response.status_code == 503
    assert "not available" in response.json()["detail"]


def test_project_evidence_endpoints(artifact_client):
    status = artifact_client.get("/api/v1/project/status")
    comparison = artifact_client.get("/api/v1/model-comparison")
    importance = artifact_client.get("/api/v1/model/feature-importance", params={"limit": 1})
    transactions = artifact_client.get(
        "/api/v1/validation/transactions", params={"filter": "false_negative"}
    )
    interesting = artifact_client.get("/api/v1/validation/interesting-cases")
    assert status.status_code == comparison.status_code == importance.status_code == 200
    assert transactions.status_code == interesting.status_code == 200
    assert status.json()["final_test"]["test_status"] == "sealed"
    assert comparison.json()["status"] == "validation_results"
    assert transactions.json()["items"][0]["outcome"] == "FALSE_NEGATIVE"
