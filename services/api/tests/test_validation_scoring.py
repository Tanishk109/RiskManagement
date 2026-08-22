from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from app.main import app
from app.schemas.risk import BatchScoreRequest, ScoreRequest
from app.services.artifacts import ArtifactUnavailable
from app.services.validation_scoring import (
    CATEGORICAL_FEATURES,
    FEATURE_SCHEMA,
    ValidationScoringPaths,
    ValidationScoringService,
    configured_validation_scoring_paths,
    get_validation_scoring_service,
)


class DeterministicFixtureModel:
    feature_names_: ClassVar[list[str]] = list(FEATURE_SCHEMA)
    tree_count_ = 990

    def __init__(self):
        self.frames = []

    def predict_proba(self, frame):
        self.frames.append(frame.copy())
        probabilities = frame["TransactionAmt"].to_numpy(dtype=float)
        return np.column_stack([1 - probabilities, probabilities])


def feature_payload(**updates):
    payload = {
        "TransactionAmt": 0.2,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "C1": 1,
        "C2": 2,
        "C3": 0,
        "C4": 0,
        "C5": 1,
        "D1": 10,
        "D2": None,
        "D3": 3,
    }
    payload.update(updates)
    return payload


@pytest.fixture()
def scoring_service(tmp_path: Path) -> ValidationScoringService:
    metadata = tmp_path / "candidate_metadata.json"
    operating = tmp_path / "validation_operating_config.json"
    metadata.write_text(
        json.dumps(
            {
                "status": "validation_candidate",
                "model_version": "fixture-validation-v1",
                "feature_names": list(FEATURE_SCHEMA),
                "categorical_feature_names": list(CATEGORICAL_FEATURES),
                "held_out_test_accessed": False,
                "actual_tree_count": 990,
            }
        ),
        encoding="utf-8",
    )
    operating.write_text(
        json.dumps(
            {
                "status": "provisional_validation_config",
                "not_final": True,
                "model_version": "fixture-validation-v1",
                "selection_split": "validation",
                "held_out_test_accessed": False,
                "scenario": "fixture",
                "review_threshold": 0.4,
                "block_threshold": 0.8,
            }
        ),
        encoding="utf-8",
    )
    service = ValidationScoringService(
        ValidationScoringPaths(
            model=tmp_path / "candidate.cbm",
            metadata=metadata,
            operating_config=operating,
        )
    )
    service.__dict__["model"] = DeterministicFixtureModel()
    return service


@pytest.fixture()
def scoring_client(client, scoring_service: ValidationScoringService):
    app.dependency_overrides[get_validation_scoring_service] = lambda: scoring_service
    yield client
    app.dependency_overrides.pop(get_validation_scoring_service, None)


def test_single_valid_scoring(scoring_client):
    response = scoring_client.post("/api/v1/score", json={"features": feature_payload()})
    assert response.status_code == 200
    assert response.json()["fraud_probability"] == pytest.approx(0.2)
    assert response.json()["decision"] == "APPROVE"
    assert response.json()["held_out_test_accessed"] is False


def test_missing_feature_handling_uses_training_normalization(
    scoring_service: ValidationScoringService,
):
    payload = feature_payload(P_emaildomain=None, D2=None)
    scoring_service.score(ScoreRequest(features=payload))
    prepared = scoring_service.model.frames[-1]
    assert prepared.loc[0, "P_emaildomain"] == "__MISSING__"
    assert np.isnan(prepared.loc[0, "D2"])


def test_missing_feature_key_is_rejected(scoring_client):
    payload = feature_payload()
    payload.pop("D3")
    response = scoring_client.post("/api/v1/score", json={"features": payload})
    assert response.status_code == 422
    assert "missing fields: D3" in response.json()["detail"]


def test_invalid_numeric_value_is_rejected(scoring_client):
    response = scoring_client.post(
        "/api/v1/score", json={"features": feature_payload(C2="not-a-number")}
    )
    assert response.status_code == 422
    assert "C2 must be numeric" in response.json()["detail"]


def test_unknown_categories_are_scored_by_catboost(scoring_client):
    response = scoring_client.post(
        "/api/v1/score",
        json={"features": feature_payload(ProductCD="NEVER_SEEN_FIXTURE_CATEGORY")},
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        ({"features": feature_payload(isFraud=1)}, 422),
        ({"features": feature_payload(), "isFraud": 1}, 422),
    ],
)
def test_forbidden_label_input(scoring_client, payload, expected_status):
    assert scoring_client.post("/api/v1/score", json=payload).status_code == expected_status


@pytest.mark.parametrize(
    ("probability", "decision"),
    [(0.399, "APPROVE"), (0.4, "REVIEW"), (0.799, "REVIEW"), (0.8, "BLOCK")],
)
def test_decision_boundary_behavior(
    scoring_service: ValidationScoringService, probability: float, decision: str
):
    result = scoring_service.score(
        ScoreRequest(features=feature_payload(TransactionAmt=probability))
    )
    assert result.decision == decision


def test_batch_scoring_counts_valid_and_invalid_rows(scoring_client):
    header = ",".join(["TransactionID", *FEATURE_SCHEMA])
    rows = [
        "tx-approve,0.2,W,visa,debit,gmail.com,1,2,0,0,1,10,,3",
        "tx-review,0.4,W,visa,debit,gmail.com,1,2,0,0,1,10,,3",
        "tx-block,0.8,W,visa,debit,gmail.com,1,2,0,0,1,10,,3",
        "tx-invalid,nope,W,visa,debit,gmail.com,1,2,0,0,1,10,,3",
    ]
    response = scoring_client.post(
        "/api/v1/score/batch", json={"csv_content": "\n".join([header, *rows])}
    )
    assert response.status_code == 200
    assert response.json()["summary"] == {
        "rows_received": 4,
        "rows_processed": 3,
        "approved": 1,
        "reviewed": 1,
        "blocked": 1,
        "invalid_rows": 1,
    }
    assert response.json()["upload_persisted"] is False


def test_invalid_csv_schema_and_label_are_rejected(scoring_client):
    missing = scoring_client.post(
        "/api/v1/score/batch", json={"csv_content": "TransactionAmt,ProductCD\n1,W"}
    )
    assert missing.status_code == 422
    forbidden = scoring_client.post(
        "/api/v1/score/batch",
        json={"csv_content": ",".join([*FEATURE_SCHEMA, "isFraud"]) + "\n"},
    )
    assert forbidden.status_code == 422
    assert "forbidden" in forbidden.json()["detail"]


def test_model_schema_order_is_exact(scoring_service: ValidationScoringService):
    scoring_service.score(ScoreRequest(features=feature_payload()))
    assert list(scoring_service.model.frames[-1].columns) == list(FEATURE_SCHEMA)


def test_saved_model_inference_is_deterministic_when_artifact_is_available():
    paths = configured_validation_scoring_paths()
    if not paths.model.is_file():
        pytest.skip("Local frozen CatBoost candidate is not available")
    service = ValidationScoringService(paths)
    request = ScoreRequest(features=feature_payload(TransactionAmt=100.0))
    first = service.score(request)
    second = service.score(request)
    assert first.fraud_probability == second.fraud_probability
    assert first.decision == second.decision


def test_held_out_test_access_guard_runs_before_file_access(tmp_path: Path):
    service = ValidationScoringService(
        ValidationScoringPaths(
            model=tmp_path / "sealed_test_model.cbm",
            metadata=tmp_path / "metadata.json",
            operating_config=tmp_path / "operating.json",
        )
    )
    with pytest.raises(ArtifactUnavailable, match="held-out test"):
        _ = service.model


def test_batch_request_object_never_persists_upload(scoring_service: ValidationScoringService):
    header = ",".join(FEATURE_SCHEMA)
    row = "0.2,W,visa,debit,gmail.com,1,2,0,0,1,10,,3"
    result = scoring_service.score_batch(BatchScoreRequest(csv_content=f"{header}\n{row}"))
    assert result.upload_persisted is False
