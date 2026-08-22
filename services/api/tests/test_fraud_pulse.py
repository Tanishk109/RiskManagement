from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd
import pytest
from app.database import Base
from app.schemas.fraud_pulse import PulseDetectorConfig
from app.services.artifacts import ArtifactUnavailable
from app.services.fraud_pulse import (
    FraudPulsePaths,
    FraudPulseService,
    aggregate_pulse,
)
from app.services.validation_scoring import (
    FEATURE_SCHEMA,
    get_validation_scoring_service,
)


def upload_csv(*, include_label: bool = False) -> str:
    headers = ["EventTime", "TransactionID", *FEATURE_SCHEMA]
    if include_label:
        headers.append("isFraud")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=headers)
    writer.writeheader()
    for index in range(5):
        row = {
            "EventTime": 1_700_000_000 + index * 900,
            "TransactionID": f"merchant-{index}",
            "TransactionAmt": 100 + index,
            "ProductCD": "W",
            "card4": "visa",
            "card6": "credit",
            "P_emaildomain": "gmail.com",
            "C1": 1,
            "C2": 1,
            "C3": 0,
            "C4": 0,
            "C5": 0,
            "D1": 1,
            "D2": "",
            "D3": "",
        }
        if include_label:
            row["isFraud"] = 1
        writer.writerow(row)
    return buffer.getvalue()


def test_pulse_status_is_not_a_classifier_and_not_evaluated(client):
    response = client.get("/api/v1/fraud-pulse/status")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["detector_is_classifier"] is False
    assert result["evaluation_status"] == "Not evaluated yet"
    assert result["review_threshold"] == 0.175
    assert result["block_threshold"] == 0.25


def test_real_chronological_validation_replay_uses_only_validation_scores(client):
    response = client.post(
        "/api/v1/fraud-pulse/replay",
        json={"config": {"window_seconds": 21600, "baseline_windows": 8}},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["source"] == "IEEE-CIS chronological validation replay"
    assert result["data_partition"] == "validation"
    assert result["rows_scored"] == 88581
    assert result["held_out_test_accessed"] is False
    assert result["evaluation_status"] == "Not evaluated yet"
    assert result["windows"][0]["baseline_state"] == "WARMING_UP"
    assert result["windows"][8]["baseline_state"] == "READY"


def test_upload_is_scored_by_frozen_candidate_before_aggregation(client):
    response = client.post(
        "/api/v1/fraud-pulse/upload",
        json={
            "csv_content": upload_csv(),
            "config": {
                "method": "percent_deviation",
                "metric": "transaction_count",
                "window_seconds": 900,
                "baseline_windows": 3,
                "percent_deviation_threshold": 0.5,
            },
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["source"] == "Merchant CSV scored by frozen CatBoost candidate"
    assert result["model_version"] == "catboost-validation-v1"
    assert result["rows_received"] == 5
    assert result["rows_scored"] == 5
    assert len(result["windows"]) == 5


def test_upload_forbids_ground_truth(client):
    response = client.post(
        "/api/v1/fraud-pulse/upload",
        json={"csv_content": upload_csv(include_label=True)},
    )
    assert response.status_code == 422
    assert "forbidden label" in response.json()["detail"]


def test_transparent_detector_marks_simulated_test_signal_only():
    # SIMULATED TEST SIGNAL: software-only detector boundary test, never product evidence.
    records = []
    for window, high_risk_rows in enumerate([1, 1, 1, 5]):
        records.extend(
            {"event_time": window * 900 + offset, "amount": 100.0, "risk_score": 0.3}
            for offset in range(high_risk_rows)
        )
    windows, alerts = aggregate_pulse(
        pd.DataFrame(records),
        config=PulseDetectorConfig(
            method="percent_deviation",
            metric="high_risk_count",
            window_seconds=900,
            baseline_windows=3,
            percent_deviation_threshold=0.5,
        ),
        review_threshold=0.175,
        block_threshold=0.25,
    )
    assert [window.high_risk_count for window in windows] == [1, 1, 1, 5]
    assert len(alerts) == 1
    assert alerts[0].label == "SPIKE ALERT"
    assert alerts[0].current_value == 5
    assert alerts[0].baseline_value == 1


def test_held_out_artifact_paths_are_rejected_before_read():
    service = FraudPulseService(
        FraudPulsePaths(
            validation_predictions=Path("artifacts/predictions/catboost_test_predictions.parquet"),
            validation_data=Path("data/processed/ieee-cis/validation.parquet"),
        ),
        get_validation_scoring_service(),
    )
    with pytest.raises(ArtifactUnavailable, match="held-out test"):
        _ = service.validation_frame


def test_pulse_operational_schema_is_normalized():
    run_columns = Base.metadata.tables["fraud_pulse_runs"].columns
    alert_columns = Base.metadata.tables["fraud_pulse_alerts"].columns
    assert "csv_content" not in run_columns
    assert "run_id" in alert_columns
    assert "detector_score" in alert_columns
