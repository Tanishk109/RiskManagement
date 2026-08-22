from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import get_settings
from ..schemas.fraud_pulse import (
    PulseAlert,
    PulseDetectorConfig,
    PulseInvalidRow,
    PulseResponse,
    PulseWindow,
)
from ..schemas.risk import BatchScoreRequest
from .artifacts import ArtifactUnavailable
from .validation_scoring import (
    FEATURE_SCHEMA,
    ValidationScoringService,
    get_validation_scoring_service,
)


@dataclass(frozen=True)
class FraudPulsePaths:
    validation_predictions: Path
    validation_data: Path


def configured_fraud_pulse_paths() -> FraudPulsePaths:
    settings = get_settings()
    return FraudPulsePaths(
        validation_predictions=settings.catboost_validation_predictions_path,
        validation_data=settings.validation_data_path,
    )


def _assert_validation_path(path: Path, purpose: str) -> None:
    normalized = path.stem.lower().replace("-", "_")
    tokens = normalized.split("_")
    if "test" in tokens or "heldout" in tokens or "held" in tokens:
        raise ArtifactUnavailable(f"{purpose} must not access a held-out test artifact")
    if "validation" not in tokens:
        raise ArtifactUnavailable(f"{purpose} must be explicitly validation-scoped")


def _parse_event_time(value: Any) -> int:
    if value is None or not str(value).strip():
        raise ValueError("EventTime is required")
    text = str(value).strip()
    try:
        numeric = float(text)
        if math.isfinite(numeric) and numeric >= 0:
            return int(numeric)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("EventTime must be Unix seconds or an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def _metric_value(window: dict[str, Any], metric: str) -> float:
    return float(window[metric])


def aggregate_pulse(
    frame: pd.DataFrame,
    *,
    config: PulseDetectorConfig,
    review_threshold: float,
    block_threshold: float,
) -> tuple[list[PulseWindow], list[PulseAlert]]:
    if frame.empty:
        return [], []
    required = {"event_time", "amount", "risk_score"}
    if not required.issubset(frame.columns):
        raise ValueError("Pulse aggregation requires event_time, amount, and risk_score")
    ordered = frame.loc[:, ["event_time", "amount", "risk_score"]].copy()
    ordered = ordered.sort_values("event_time", kind="stable")
    start = int(ordered["event_time"].min())
    ordered["window_index"] = ((ordered["event_time"] - start) // config.window_seconds).astype(int)

    raw_windows: list[dict[str, Any]] = []
    last_index = int(ordered["window_index"].max())
    for index in range(last_index + 1):
        group = ordered.loc[ordered["window_index"] == index]
        review_mask = (group["risk_score"] >= review_threshold) & (group["risk_score"] < block_threshold)
        block_mask = group["risk_score"] >= block_threshold
        high_mask = group["risk_score"] >= review_threshold
        raw_windows.append(
            {
                "window_index": index,
                "window_start": start + index * config.window_seconds,
                "window_end": start + (index + 1) * config.window_seconds,
                "transaction_count": len(group),
                "mean_risk_score": float(group["risk_score"].mean()) if len(group) else 0.0,
                "high_risk_count": int(high_mask.sum()),
                "review_count": int(review_mask.sum()),
                "block_count": int(block_mask.sum()),
                "high_risk_amount": float(group.loc[high_mask, "amount"].sum()) if len(group) else 0.0,
            }
        )

    windows: list[PulseWindow] = []
    alerts: list[PulseAlert] = []
    prior_values: list[float] = []
    for raw in raw_windows:
        current = _metric_value(raw, config.metric)
        ready = len(prior_values) >= config.baseline_windows
        baseline_value: float | None = None
        absolute_change: float | None = None
        percent_deviation: float | None = None
        detector_score: float | None = None
        active = False
        if ready:
            baseline_slice = prior_values[-config.baseline_windows :]
            if config.method == "ewma":
                baseline_value = baseline_slice[0]
                for value in baseline_slice[1:]:
                    baseline_value = config.ewma_alpha * value + (1 - config.ewma_alpha) * baseline_value
                detector_score = (current - baseline_value) / max(abs(baseline_value), 1.0)
                active = detector_score >= config.sensitivity
            else:
                baseline_value = float(np.mean(baseline_slice))
                if config.method == "rolling_zscore":
                    deviation = float(np.std(baseline_slice, ddof=0))
                    detector_score = (current - baseline_value) / max(deviation, 1e-9)
                    active = detector_score >= config.sensitivity
                else:
                    detector_score = (current - baseline_value) / max(abs(baseline_value), 1.0)
                    active = detector_score >= config.percent_deviation_threshold
            absolute_change = current - baseline_value
            percent_deviation = absolute_change / abs(baseline_value) if baseline_value else None

        window = PulseWindow(
            **raw,
            monitored_value=current,
            baseline_state="READY" if ready else "WARMING_UP",
            baseline_value=baseline_value,
            absolute_change=absolute_change,
            percent_deviation=percent_deviation,
            detector_score=detector_score,
            alert_active=active,
        )
        windows.append(window)
        if active and baseline_value is not None and absolute_change is not None and detector_score is not None:
            alerts.append(
                PulseAlert(
                    window_index=window.window_index,
                    window_start=window.window_start,
                    window_end=window.window_end,
                    metric=config.metric,
                    current_value=current,
                    baseline_value=baseline_value,
                    absolute_change=absolute_change,
                    percent_deviation=percent_deviation,
                    detector_score=detector_score,
                    label="SPIKE ALERT",
                )
            )
        prior_values.append(current)
    return windows, alerts


class FraudPulseService:
    """Transparent monitoring over frozen model scores; this service is not a classifier."""

    def __init__(self, paths: FraudPulsePaths, scoring: ValidationScoringService):
        self.paths = paths
        self.scoring = scoring

    @cached_property
    def validation_frame(self) -> pd.DataFrame:
        _assert_validation_path(self.paths.validation_predictions, "Pulse predictions")
        _assert_validation_path(self.paths.validation_data, "Pulse transactions")
        if not self.paths.validation_predictions.is_file() or not self.paths.validation_data.is_file():
            raise ArtifactUnavailable("Validation replay artifacts are unavailable")
        predictions = pd.read_parquet(
            self.paths.validation_predictions,
            columns=["TransactionID", "fraud_probability", "model_version"],
        )
        transactions = pd.read_parquet(
            self.paths.validation_data,
            columns=["TransactionID", "TransactionDT", "TransactionAmt"],
        )
        if predictions["TransactionID"].duplicated().any() or transactions["TransactionID"].duplicated().any():
            raise ArtifactUnavailable("Validation replay requires unique TransactionID values")
        joined = transactions.merge(predictions, on="TransactionID", how="inner", validate="one_to_one")
        if len(joined) != len(transactions) or len(joined) != len(predictions):
            raise ArtifactUnavailable("Validation replay score join is incomplete")
        versions = joined["model_version"].dropna().astype(str).unique().tolist()
        if versions != [str(self.scoring.metadata["model_version"])]:
            raise ArtifactUnavailable("Validation replay scores do not match the frozen model version")
        return joined.rename(
            columns={"TransactionDT": "event_time", "TransactionAmt": "amount", "fraud_probability": "risk_score"}
        ).loc[:, ["event_time", "amount", "risk_score"]]

    def _response(
        self,
        *,
        frame: pd.DataFrame,
        config: PulseDetectorConfig,
        source: str,
        partition: str,
        rows_received: int,
        invalid_rows: list[PulseInvalidRow],
    ) -> PulseResponse:
        thresholds = self.scoring.operating_config
        review = float(thresholds["review_threshold"])
        block = float(thresholds["block_threshold"])
        windows, alerts = aggregate_pulse(
            frame,
            config=config,
            review_threshold=review,
            block_threshold=block,
        )
        return PulseResponse(
            source=source,
            data_partition=partition,
            evaluation_status="Not evaluated yet",
            detector_is_classifier=False,
            config=config,
            model_version=str(self.scoring.metadata["model_version"]),
            review_threshold=review,
            block_threshold=block,
            rows_received=rows_received,
            rows_scored=len(frame),
            invalid_rows=invalid_rows,
            windows=windows,
            alerts=alerts,
            held_out_test_accessed=False,
            limitations=[
                "Alerts identify score-volume changes, not confirmed fraud attacks.",
                "Validation replay is development evidence and is not detector performance evaluation.",
                "Baselines depend on the selected window, metric, and sensitivity settings.",
            ],
        )

    def replay_validation(self, config: PulseDetectorConfig) -> PulseResponse:
        frame = self.validation_frame
        return self._response(
            frame=frame,
            config=config,
            source="IEEE-CIS chronological validation replay",
            partition="validation",
            rows_received=len(frame),
            invalid_rows=[],
        )

    def analyze_upload(self, csv_content: str, config: PulseDetectorConfig) -> PulseResponse:
        if len(csv_content.encode("utf-8")) > 1_000_000:
            raise ValueError("CSV exceeds the 1 MB pulse upload limit")
        reader = csv.DictReader(io.StringIO(csv_content, newline=""))
        headers = [header.removeprefix("\ufeff").strip() for header in (reader.fieldnames or [])]
        expected = {"EventTime", *FEATURE_SCHEMA}
        optional = {"TransactionID"}
        forbidden = {"isFraud", "actual_label"}.intersection(headers)
        missing = expected.difference(headers)
        unexpected = set(headers).difference(expected).difference(optional)
        if forbidden:
            raise ValueError(f"CSV contains forbidden label columns: {', '.join(sorted(forbidden))}")
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing columns: {', '.join(sorted(missing))}")
            if unexpected:
                details.append(f"unexpected columns: {', '.join(sorted(unexpected))}")
            raise ValueError("Invalid pulse CSV schema (" + "; ".join(details) + ")")
        rows = list(reader)
        if len(rows) > 1_000:
            raise ValueError("CSV exceeds the 1,000 row pulse upload limit")

        scoring_headers = (["TransactionID"] if "TransactionID" in headers else []) + list(FEATURE_SCHEMA)
        scoring_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(scoring_buffer, fieldnames=scoring_headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        scored = self.scoring.score_batch(BatchScoreRequest(csv_content=scoring_buffer.getvalue()))
        by_row = {index + 2: row for index, row in enumerate(rows)}
        pulse_records: list[dict[str, float | int]] = []
        invalid_rows = [PulseInvalidRow.model_validate(item) for item in scored.invalid_rows]
        for item in scored.results:
            raw = by_row[item.row]
            try:
                event_time = _parse_event_time(raw.get("EventTime"))
            except ValueError as exc:
                invalid_rows.append(
                    PulseInvalidRow(row=item.row, transaction_id=item.transaction_id, errors=[str(exc)])
                )
                continue
            pulse_records.append(
                {
                    "event_time": event_time,
                    "amount": float(raw["TransactionAmt"]),
                    "risk_score": item.fraud_probability,
                }
            )
        frame = pd.DataFrame(pulse_records, columns=["event_time", "amount", "risk_score"])
        return self._response(
            frame=frame,
            config=config,
            source="Merchant CSV scored by frozen CatBoost candidate",
            partition="merchant upload",
            rows_received=len(rows),
            invalid_rows=sorted(invalid_rows, key=lambda item: item.row),
        )


@lru_cache(maxsize=1)
def get_fraud_pulse_service() -> FraudPulseService:
    return FraudPulseService(configured_fraud_pulse_paths(), get_validation_scoring_service())
