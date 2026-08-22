from __future__ import annotations

import csv
import io
import math
from collections import Counter
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from merchantshield_ml.catboost_candidate import normalize_catboost_features

from ..config import get_settings
from ..schemas.risk import (
    BatchScoreRequest,
    BatchScoreResponse,
    ScoreRequest,
    ScoreResponse,
)
from .artifacts import ArtifactUnavailable, read_json
from .decision_engine import decision_from_score

FEATURE_SCHEMA = (
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "D1",
    "D2",
    "D3",
)
CATEGORICAL_FEATURES = ("ProductCD", "card4", "card6", "P_emaildomain")
NUMERIC_FEATURES = tuple(name for name in FEATURE_SCHEMA if name not in CATEGORICAL_FEATURES)
FORBIDDEN_FIELDS = {
    "isFraud",
    "actual_label",
    "future_chargeback_outcome",
    "future_fraud_label",
}
OPTIONAL_BATCH_COLUMNS = {"TransactionID"}
MAX_BATCH_ROWS = 1_000
MAX_BATCH_BYTES = 1_000_000


@dataclass(frozen=True)
class ValidationScoringPaths:
    model: Path
    metadata: Path
    operating_config: Path


def configured_validation_scoring_paths() -> ValidationScoringPaths:
    settings = get_settings()
    return ValidationScoringPaths(
        model=settings.catboost_model_path,
        metadata=settings.catboost_metadata_path,
        operating_config=settings.validation_operating_config_path,
    )


def _assert_not_held_out(path: Path, purpose: str) -> None:
    tokens = path.stem.lower().replace("-", "_").split("_")
    if "test" in tokens or "heldout" in tokens or "held" in tokens:
        raise ArtifactUnavailable(f"{purpose} must not reference a held-out test artifact")


class ValidationScoringService:
    """Inference-only access to the frozen validation candidate and provisional thresholds."""

    def __init__(self, paths: ValidationScoringPaths):
        self.paths = paths

    @cached_property
    def metadata(self) -> dict[str, Any]:
        _assert_not_held_out(self.paths.metadata, "Model metadata")
        payload = read_json(self.paths.metadata)
        if payload.get("status") != "validation_candidate":
            raise ArtifactUnavailable("CatBoost metadata is not a validation candidate")
        if payload.get("held_out_test_accessed") is not False:
            raise ArtifactUnavailable("CatBoost metadata failed the held-out access guard")
        if payload.get("feature_names") != list(FEATURE_SCHEMA):
            raise ArtifactUnavailable("CatBoost metadata feature order does not match Risk Check")
        if payload.get("categorical_feature_names") != list(CATEGORICAL_FEATURES):
            raise ArtifactUnavailable("CatBoost categorical schema does not match training")
        return payload

    @cached_property
    def operating_config(self) -> dict[str, Any]:
        _assert_not_held_out(self.paths.operating_config, "Threshold configuration")
        payload = read_json(self.paths.operating_config)
        if (
            payload.get("status") != "provisional_validation_config"
            or payload.get("selection_split") != "validation"
            or payload.get("held_out_test_accessed") is not False
            or payload.get("not_final") is not True
        ):
            raise ArtifactUnavailable("Provisional threshold configuration failed provenance checks")
        if payload.get("model_version") != self.metadata.get("model_version"):
            raise ArtifactUnavailable("Model and threshold versions do not match")
        review = float(payload["review_threshold"])
        block = float(payload["block_threshold"])
        if not 0 <= review < block <= 1:
            raise ArtifactUnavailable("Provisional thresholds are invalid")
        return payload

    @cached_property
    def model(self) -> CatBoostClassifier:
        _assert_not_held_out(self.paths.model, "Model")
        if not self.paths.model.is_file():
            raise ArtifactUnavailable("Frozen CatBoost candidate is not available")
        model = CatBoostClassifier()
        try:
            model.load_model(self.paths.model)
        except Exception as exc:  # CatBoost raises several native wrapper exception types.
            raise ArtifactUnavailable("Frozen CatBoost candidate could not be loaded") from exc
        if list(model.feature_names_) != list(FEATURE_SCHEMA):
            raise ArtifactUnavailable("Saved CatBoost feature order does not match Risk Check")
        if int(model.tree_count_) != int(self.metadata.get("actual_tree_count", -1)):
            raise ArtifactUnavailable("Saved CatBoost tree count does not match its metadata")
        return model

    @property
    def threshold_configuration(self) -> dict[str, str | float | bool]:
        config = self.operating_config
        return {
            "id": f"{config['model_version']}-{config['scenario']}-provisional-validation",
            "status": str(config["status"]),
            "selection_split": "validation",
            "scenario": str(config["scenario"]),
            "review_threshold": float(config["review_threshold"]),
            "block_threshold": float(config["block_threshold"]),
            "provisional": True,
        }

    def _normalize_payload(self, features: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        errors: list[str] = []
        used_forbidden = sorted(FORBIDDEN_FIELDS.intersection(features))
        if used_forbidden:
            errors.append(f"forbidden scoring fields: {', '.join(used_forbidden)}")
        missing = sorted(set(FEATURE_SCHEMA).difference(features))
        unexpected = sorted(set(features).difference(FEATURE_SCHEMA).difference(FORBIDDEN_FIELDS))
        if missing:
            errors.append(f"missing fields: {', '.join(missing)}")
        if unexpected:
            errors.append(f"unexpected fields: {', '.join(unexpected)}")
        if errors:
            return {}, errors

        normalized: dict[str, Any] = {}
        for feature in NUMERIC_FEATURES:
            value = features[feature]
            if value is None or (isinstance(value, str) and not value.strip()):
                if feature == "TransactionAmt":
                    errors.append("TransactionAmt is required")
                normalized[feature] = np.nan
                continue
            if isinstance(value, bool):
                errors.append(f"{feature} must be numeric")
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                errors.append(f"{feature} must be numeric")
                continue
            if not math.isfinite(parsed):
                errors.append(f"{feature} must be a finite number")
                continue
            if feature == "TransactionAmt" and parsed < 0:
                errors.append("TransactionAmt must be greater than or equal to 0")
                continue
            normalized[feature] = parsed

        for feature in CATEGORICAL_FEATURES:
            value = features[feature]
            if value is None or (isinstance(value, str) and not value.strip()):
                normalized[feature] = None
                continue
            if not isinstance(value, str):
                errors.append(f"{feature} must be text or null")
                continue
            cleaned = value.strip()
            if len(cleaned) > 255:
                errors.append(f"{feature} must be at most 255 characters")
                continue
            normalized[feature] = cleaned
        return normalized, errors

    def _prepared_frame(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows, columns=FEATURE_SCHEMA)
        for feature in NUMERIC_FEATURES:
            frame[feature] = pd.to_numeric(frame[feature], errors="raise").astype(float)
        return normalize_catboost_features(
            frame,
            list(FEATURE_SCHEMA),
            list(CATEGORICAL_FEATURES),
        )

    def _probabilities(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.array([], dtype=float)
        frame = self._prepared_frame(rows)
        probabilities = np.asarray(self.model.predict_proba(frame)[:, 1], dtype=float)
        if len(probabilities) != len(rows) or not np.isfinite(probabilities).all():
            raise ArtifactUnavailable("Saved CatBoost candidate returned invalid probabilities")
        return probabilities

    def score(self, payload: ScoreRequest) -> ScoreResponse:
        if payload.persist:
            raise ValueError("Risk Check scoring does not persist submitted feature payloads")
        normalized, errors = self._normalize_payload(payload.features)
        if errors:
            raise ValueError("; ".join(errors))
        probability = float(self._probabilities([normalized])[0])
        config = self.operating_config
        decision = decision_from_score(
            probability,
            float(config["review_threshold"]),
            float(config["block_threshold"]),
        )
        threshold_configuration = self.threshold_configuration
        return ScoreResponse(
            fraud_probability=probability,
            risk_score=probability,
            decision=decision,
            rules_triggered=[],
            top_factors=[],
            model_version=str(self.metadata["model_version"]),
            threshold_config_id=str(threshold_configuration["id"]),
            threshold_configuration=threshold_configuration,
            feature_schema=list(FEATURE_SCHEMA),
            held_out_test_accessed=False,
        )

    def score_batch(self, payload: BatchScoreRequest) -> BatchScoreResponse:
        encoded_size = len(payload.csv_content.encode("utf-8"))
        if encoded_size > MAX_BATCH_BYTES:
            raise ValueError(f"CSV exceeds the {MAX_BATCH_BYTES // 1_000_000} MB upload limit")
        if "\x00" in payload.csv_content:
            raise ValueError("CSV contains a null byte")
        try:
            reader = csv.DictReader(io.StringIO(payload.csv_content, newline=""))
            headers = [header.removeprefix("\ufeff").strip() for header in (reader.fieldnames or [])]
        except csv.Error as exc:
            raise ValueError("CSV could not be parsed") from exc
        if not headers:
            raise ValueError("CSV must contain a header row")
        if len(headers) != len(set(headers)):
            raise ValueError("CSV contains duplicate columns")
        forbidden = sorted(FORBIDDEN_FIELDS.intersection(headers))
        if forbidden:
            raise ValueError(f"CSV contains forbidden scoring columns: {', '.join(forbidden)}")
        missing = sorted(set(FEATURE_SCHEMA).difference(headers))
        unexpected = sorted(set(headers).difference(FEATURE_SCHEMA).difference(OPTIONAL_BATCH_COLUMNS))
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing columns: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected columns: {', '.join(unexpected)}")
            raise ValueError("Invalid CSV schema (" + "; ".join(details) + ")")

        valid_rows: list[dict[str, Any]] = []
        valid_meta: list[tuple[int, str]] = []
        invalid_rows: list[dict[str, Any]] = []
        rows_received = 0
        try:
            for row_number, raw in enumerate(reader, start=2):
                rows_received += 1
                if rows_received > MAX_BATCH_ROWS:
                    raise ValueError(f"CSV exceeds the {MAX_BATCH_ROWS} row limit")
                normalized_raw = {
                    key.removeprefix("\ufeff").strip() if key is not None else key: value
                    for key, value in raw.items()
                }
                transaction_id_value = normalized_raw.get("TransactionID")
                transaction_id = (
                    str(transaction_id_value).strip()
                    if transaction_id_value is not None and str(transaction_id_value).strip()
                    else str(row_number - 1)
                )
                features = {name: normalized_raw.get(name) for name in FEATURE_SCHEMA}
                normalized, errors = self._normalize_payload(features)
                if len(transaction_id) > 80:
                    errors.append("TransactionID must be at most 80 characters")
                if errors:
                    invalid_rows.append(
                        {"row": row_number, "transaction_id": transaction_id, "errors": errors}
                    )
                    continue
                valid_rows.append(normalized)
                valid_meta.append((row_number, transaction_id))
        except csv.Error as exc:
            raise ValueError("CSV could not be parsed") from exc

        probabilities = self._probabilities(valid_rows)
        config = self.operating_config
        results = []
        decisions: Counter[str] = Counter()
        for (row_number, transaction_id), probability in zip(
            valid_meta, probabilities, strict=True
        ):
            decision = decision_from_score(
                float(probability),
                float(config["review_threshold"]),
                float(config["block_threshold"]),
            )
            decisions[decision] += 1
            results.append(
                {
                    "row": row_number,
                    "transaction_id": transaction_id,
                    "fraud_probability": float(probability),
                    "decision": decision,
                }
            )
        return BatchScoreResponse(
            summary={
                "rows_received": rows_received,
                "rows_processed": len(results),
                "approved": decisions["APPROVE"],
                "reviewed": decisions["REVIEW"],
                "blocked": decisions["BLOCK"],
                "invalid_rows": len(invalid_rows),
            },
            results=results,
            invalid_rows=invalid_rows,
            model_version=str(self.metadata["model_version"]),
            threshold_configuration=self.threshold_configuration,
            feature_schema=list(FEATURE_SCHEMA),
            upload_persisted=False,
            held_out_test_accessed=False,
        )


@lru_cache(maxsize=1)
def get_validation_scoring_service() -> ValidationScoringService:
    return ValidationScoringService(configured_validation_scoring_paths())
