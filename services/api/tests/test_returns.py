from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from app.database import Base
from app.schemas.returns import ReturnOrderInput
from app.services.artifacts import ArtifactUnavailable
from app.services.returns_scoring import (
    ReturnRiskPaths,
    ReturnRiskService,
    get_return_risk_service,
)
from merchantshield_ml.returns import RETURN_FEATURES


def return_payload(**overrides):
    payload = {
        "order_value": 185.50,
        "quantity": 12,
        "unique_stock_count": 3,
        "country": "United Kingdom",
        "stock_code": "85123A",
        "prior_order_count": 4,
        "prior_cancellation_rate": 0.25,
        "prior_average_order_value": 142.75,
        "order_hour": 14,
        "order_day_of_week": 2,
    }
    payload.update(overrides)
    return payload


def test_return_status_has_separate_measured_uci_metrics(client):
    response = client.get("/api/v1/returns/status")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["dataset_id"] == 502
    assert result["model_version"] == "returns-catboost-uci-v1"
    assert result["evaluation_status"] == "Evaluated on chronological UCI test partition"
    assert result["test_metrics"]["average_precision"] == pytest.approx(0.8561512057564747)
    assert result["ieee_cis_model_modified"] is False
    assert result["ieee_cis_held_out_test_accessed"] is False
    assert "not a verified physical-return label" in " ".join(result["limitations"])


def test_single_return_scoring_uses_saved_candidate(client):
    response = client.post("/api/v1/returns/score", json=return_payload())
    assert response.status_code == 200, response.text
    result = response.json()
    assert 0 <= result["return_risk_probability"] <= 1
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["model_version"] == "returns-catboost-uci-v1"
    assert result["automatic_rejection"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {key: value for key, value in return_payload().items() if key != "quantity"},
        return_payload(quantity="not-a-number"),
        return_payload(order_hour=24),
        return_payload(prior_cancellation_rate=1.1),
    ],
)
def test_return_input_validation(client, payload):
    assert client.post("/api/v1/returns/score", json=payload).status_code == 422


def test_return_label_inputs_are_forbidden(client):
    for label in ("is_cancellation_proxy", "isFraud", "actual_label"):
        response = client.post("/api/v1/returns/score", json=return_payload(**{label: 1}))
        assert response.status_code == 422


def test_unknown_categories_are_scored_without_schema_substitution(client):
    response = client.post(
        "/api/v1/returns/score",
        json=return_payload(country="Neverland", stock_code="NEW-SKU-2026"),
    )
    assert response.status_code == 200, response.text
    assert np.isfinite(response.json()["return_risk_probability"])


def test_return_batch_scores_every_valid_row_without_persistence(client):
    response = client.post(
        "/api/v1/returns/score/batch",
        json={"rows": [return_payload(), return_payload(stock_code="22423", order_value=42)]},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["rows_received"] == result["rows_scored"] == 2
    assert [item["row"] for item in result["results"]] == [1, 2]
    assert result["uploaded_file_persisted"] is False
    assert result["automatic_rejection"] is False


def test_return_model_schema_order_is_exact():
    service = get_return_risk_service()
    assert service.metadata["feature_schema"] == list(RETURN_FEATURES)


def test_saved_return_model_inference_is_deterministic():
    service = get_return_risk_service()
    order = ReturnOrderInput(**return_payload())
    first = service.score_one(order).return_risk_probability
    second = service.score_one(order).return_risk_probability
    assert first == pytest.approx(second, rel=0, abs=0)


def test_return_service_refuses_ieee_held_out_artifacts():
    with pytest.raises(ArtifactUnavailable, match="IEEE-CIS held-out"):
        ReturnRiskService(
            ReturnRiskPaths(
                model=Path("artifacts/models/return.cbm"),
                metadata=Path("artifacts/models/return.json"),
                metrics=Path("data/processed/ieee-cis/test.parquet"),
            )
        )


def test_return_predictions_schema_is_normalized():
    columns = Base.metadata.tables["return_predictions"].columns
    assert all(feature in columns for feature in RETURN_FEATURES)
    assert "training_rows" not in columns
    assert "raw_order" not in columns
