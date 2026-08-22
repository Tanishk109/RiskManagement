from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DetectorMethod = Literal["rolling_zscore", "ewma", "percent_deviation"]
PulseMetric = Literal["transaction_count", "mean_risk_score", "high_risk_count", "high_risk_amount"]


class PulseDetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: DetectorMethod = "rolling_zscore"
    metric: PulseMetric = "high_risk_count"
    window_seconds: int = Field(default=21_600, ge=900, le=604_800)
    baseline_windows: int = Field(default=8, ge=3, le=48)
    sensitivity: float = Field(default=3.0, gt=0, le=20)
    ewma_alpha: float = Field(default=0.3, gt=0, le=1)
    percent_deviation_threshold: float = Field(default=0.5, gt=0, le=10)


class PulseReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: PulseDetectorConfig = Field(default_factory=PulseDetectorConfig)


class PulseUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv_content: str = Field(min_length=1, max_length=2_000_000)
    config: PulseDetectorConfig = Field(default_factory=PulseDetectorConfig)


class PulseWindow(BaseModel):
    window_index: int
    window_start: int
    window_end: int
    transaction_count: int
    mean_risk_score: float
    high_risk_count: int
    review_count: int
    block_count: int
    high_risk_amount: float
    monitored_value: float
    baseline_state: Literal["WARMING_UP", "READY"]
    baseline_value: float | None
    absolute_change: float | None
    percent_deviation: float | None
    detector_score: float | None
    alert_active: bool


class PulseAlert(BaseModel):
    window_index: int
    window_start: int
    window_end: int
    metric: PulseMetric
    current_value: float
    baseline_value: float
    absolute_change: float
    percent_deviation: float | None
    detector_score: float
    label: Literal["SPIKE ALERT"]


class PulseInvalidRow(BaseModel):
    row: int
    transaction_id: str | None
    errors: list[str]


class PulseResponse(BaseModel):
    source: Literal["IEEE-CIS chronological validation replay", "Merchant CSV scored by frozen CatBoost candidate"]
    data_partition: Literal["validation", "merchant upload"]
    evaluation_status: Literal["Not evaluated yet"]
    detector_is_classifier: Literal[False]
    config: PulseDetectorConfig
    model_version: str
    review_threshold: float
    block_threshold: float
    rows_received: int
    rows_scored: int
    invalid_rows: list[PulseInvalidRow]
    windows: list[PulseWindow]
    alerts: list[PulseAlert]
    held_out_test_accessed: Literal[False]
    limitations: list[str]


class PulseStatus(BaseModel):
    module: Literal["Fraud-Spike Detector"]
    data_source: str
    evaluation_status: Literal["Not evaluated yet"]
    detector_is_classifier: Literal[False]
    model_version: str
    review_threshold: float
    block_threshold: float
    methods: list[DetectorMethod]
    metrics: list[PulseMetric]
    upload_required_columns: list[str]
    limitations: list[str]
