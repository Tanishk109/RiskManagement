from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from merchantshield_ml.returns import (
    RETURN_CATEGORICAL_FEATURES,
    RETURN_FEATURES,
    RETURN_HIGH_THRESHOLD,
    RETURN_MEDIUM_THRESHOLD,
    normalize_return_features,
    return_risk_level,
)

from ..config import get_settings
from ..schemas.returns import (
    ReturnBatchResponse,
    ReturnOrderInput,
    ReturnScoreResult,
    ReturnStatus,
    ReturnThresholds,
)
from .artifacts import ArtifactUnavailable


@dataclass(frozen=True)
class ReturnRiskPaths:
    model: Path
    metadata: Path
    metrics: Path


class ReturnRiskService:
    def __init__(self, paths: ReturnRiskPaths) -> None:
        self.paths = paths
        self._assert_dataset_isolation()

    def _assert_dataset_isolation(self) -> None:
        forbidden = ("ieee-cis/test", "final_test", "ieee_test", "heldout")
        for path in (self.paths.model, self.paths.metadata, self.paths.metrics):
            normalized = str(path).lower().replace("\\", "/")
            if any(token in normalized for token in forbidden):
                raise ArtifactUnavailable("Return scoring refuses IEEE-CIS held-out test artifacts")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise ArtifactUnavailable(f"Return-risk artifact is unavailable: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactUnavailable(f"Return-risk artifact is invalid: {path}") from exc
        if not isinstance(payload, dict):
            raise ArtifactUnavailable(f"Return-risk artifact must be an object: {path}")
        return payload

    @cached_property
    def metadata(self) -> dict[str, Any]:
        payload = self._read_json(self.paths.metadata)
        if payload.get("feature_schema") != list(RETURN_FEATURES):
            raise ArtifactUnavailable("Saved return model feature order does not match runtime schema")
        if payload.get("categorical_features") != list(RETURN_CATEGORICAL_FEATURES):
            raise ArtifactUnavailable("Saved return model categorical schema does not match runtime schema")
        return payload

    @cached_property
    def evaluation(self) -> dict[str, Any]:
        payload = self._read_json(self.paths.metrics)
        if payload.get("data_source") != "UCI Online Retail II (dataset 502)":
            raise ArtifactUnavailable("Return evaluation has unexpected dataset provenance")
        if payload.get("ieee_cis_held_out_test_accessed") is not False:
            raise ArtifactUnavailable("Return evaluation does not preserve IEEE-CIS isolation")
        return payload

    @cached_property
    def model(self) -> CatBoostClassifier:
        if not self.paths.model.is_file():
            raise ArtifactUnavailable(f"Return-risk model is unavailable: {self.paths.model}")
        model = CatBoostClassifier()
        try:
            model.load_model(self.paths.model)
        except Exception as exc:
            raise ArtifactUnavailable("Saved return-risk CatBoost model could not be loaded") from exc
        return model

    @property
    def thresholds(self) -> ReturnThresholds:
        return ReturnThresholds(medium=RETURN_MEDIUM_THRESHOLD, high=RETURN_HIGH_THRESHOLD)

    def score_many(self, orders: list[ReturnOrderInput]) -> ReturnBatchResponse:
        if not orders:
            raise ValueError("At least one order is required")
        if len(orders) > 1_000:
            raise ValueError("Batch scoring is limited to 1,000 rows")
        frame = pd.DataFrame([order.model_dump() for order in orders], columns=RETURN_FEATURES)
        normalized = normalize_return_features(frame)
        probabilities = np.asarray(self.model.predict_proba(normalized)[:, 1], dtype=float)
        version = str(self.metadata["model_version"])
        results = [
            ReturnScoreResult(
                row=index + 1,
                return_risk_probability=float(probability),
                risk_level=return_risk_level(float(probability)),
                model_version=version,
                thresholds=self.thresholds,
            )
            for index, probability in enumerate(probabilities)
        ]
        return ReturnBatchResponse(
            rows_received=len(orders),
            rows_scored=len(results),
            results=results,
            model_version=version,
            thresholds=self.thresholds,
        )

    def score_one(self, order: ReturnOrderInput) -> ReturnScoreResult:
        result = self.score_many([order]).results[0]
        return result.model_copy(update={"row": None})

    def status(self) -> ReturnStatus:
        metrics = self.evaluation["models"]["catboost"]["test"]
        return ReturnStatus(
            data_source=str(self.evaluation["data_source"]),
            dataset_id=int(self.evaluation["uci_dataset_id"]),
            model_version=str(self.metadata["model_version"]),
            evaluation_status="Evaluated on chronological UCI test partition",
            proxy_disclosure=str(self.evaluation["proxy_disclosure"]),
            feature_schema=list(self.metadata["feature_schema"]),
            categorical_features=list(self.metadata["categorical_features"]),
            test_metrics=metrics,
            thresholds=self.thresholds,
            limitations=[
                "The target is a cancellation/reversal proxy, not a verified physical-return label.",
                "Metrics apply only to the separate Online Retail II chronological test partition.",
                "Customer history fields must contain prior orders only at merchant integration time.",
                "Risk labels support review prioritization and never automatically reject an order.",
            ],
        )


@lru_cache(maxsize=1)
def get_return_risk_service() -> ReturnRiskService:
    settings = get_settings()
    return ReturnRiskService(
        ReturnRiskPaths(
            model=settings.return_model_path,
            metadata=settings.return_model_metadata_path,
            metrics=settings.return_metrics_path,
        )
    )
