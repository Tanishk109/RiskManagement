from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from ..config import get_settings
from .artifacts import ArtifactUnavailable
from .validation_scoring import FEATURE_SCHEMA

ValidationFilter = Literal[
    "all",
    "true_fraud",
    "true_legitimate",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "high_risk",
    "high_value",
]

HIGH_VALUE_THRESHOLD = 500.0


@dataclass(frozen=True)
class ArtifactPaths:
    eda_summary: Path
    split_metadata: Path
    baseline_metrics: Path
    catboost_metrics: Path
    catboost_metadata: Path
    experiments: Path
    feature_importance: Path
    baseline_predictions: Path
    selected_predictions: Path
    validation_data: Path
    threshold_analysis: Path
    final_metrics: Path
    threshold_grid: Path | None = None
    operating_config: Path | None = None
    merchant_scenarios: Path | None = None


def configured_artifact_paths() -> ArtifactPaths:
    settings = get_settings()
    return ArtifactPaths(
        eda_summary=settings.eda_summary_path,
        split_metadata=settings.split_metadata_path,
        baseline_metrics=settings.baseline_metrics_path,
        catboost_metrics=settings.catboost_metrics_path,
        catboost_metadata=settings.catboost_metadata_path,
        experiments=settings.experiments_path,
        feature_importance=settings.feature_importance_path,
        baseline_predictions=settings.baseline_validation_predictions_path,
        selected_predictions=settings.catboost_validation_predictions_path,
        validation_data=settings.validation_data_path,
        threshold_analysis=settings.threshold_analysis_path,
        final_metrics=settings.metrics_path,
        threshold_grid=settings.threshold_grid_path,
        operating_config=settings.validation_operating_config_path,
        merchant_scenarios=settings.merchant_scenarios_path,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactUnavailable(f"Required project artifact is not available: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactUnavailable(f"Project artifact is unreadable: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ArtifactUnavailable(f"Project artifact must contain an object: {path.name}")
    return payload


def _required_mapping(payload: dict[str, Any], key: str, artifact_name: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ArtifactUnavailable(f"{artifact_name} does not contain a valid {key} object")
    return value


def _native(value: Any) -> str | float | int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def _metric_view(record: dict[str, Any]) -> dict[str, float | int]:
    return {
        "average_precision": float(record["average_precision"]),
        "roc_auc": float(record["roc_auc"]),
        "precision_at_0_5": float(record["precision"]),
        "recall_at_0_5": float(record["recall"]),
        "f1_at_0_5": float(record["f1"]),
        "false_positives": int(record["fp"]),
        "false_negatives": int(record["fn"]),
        "true_positives": int(record["tp"]),
        "true_negatives": int(record["tn"]),
        "threshold": float(record.get("default_threshold", 0.5)),
    }


class ProjectArtifactService:
    """Cached, read-only projection of generated validation artifacts for the product API."""

    def __init__(self, paths: ArtifactPaths):
        self.paths = paths

    @cached_property
    def eda(self) -> dict[str, Any]:
        return _read_json(self.paths.eda_summary)

    @cached_property
    def split(self) -> dict[str, Any]:
        return _read_json(self.paths.split_metadata)

    @cached_property
    def baseline(self) -> dict[str, Any]:
        return _read_json(self.paths.baseline_metrics)

    @cached_property
    def catboost(self) -> dict[str, Any]:
        return _read_json(self.paths.catboost_metrics)

    @cached_property
    def catboost_metadata(self) -> dict[str, Any]:
        return _read_json(self.paths.catboost_metadata)

    @cached_property
    def selected_candidate(self) -> dict[str, Any]:
        return _required_mapping(
            self.catboost, "selected_candidate", "catboost_validation.json"
        )

    @cached_property
    def classification_threshold(self) -> float:
        threshold = float(self.selected_candidate["default_threshold"])
        if not 0 <= threshold <= 1:
            raise ArtifactUnavailable("Selected candidate threshold is outside [0, 1]")
        return threshold

    @cached_property
    def operating_config(self) -> dict[str, Any] | None:
        if self.paths.operating_config is None or not self.paths.operating_config.is_file():
            return None
        config = _read_json(self.paths.operating_config)
        if (
            config.get("selection_split") != "validation"
            or config.get("held_out_test_accessed") is not False
            or config.get("not_final") is not True
        ):
            raise ArtifactUnavailable("Validation operating configuration failed provenance checks")
        return config

    @cached_property
    def experiments(self) -> pd.DataFrame:
        if not self.paths.experiments.is_file():
            raise ArtifactUnavailable("Experiment registry is not available")
        try:
            frame = pd.read_csv(self.paths.experiments)
        except (OSError, ValueError) as exc:
            raise ArtifactUnavailable("Experiment registry is unreadable") from exc
        required = {"experiment_id", "average_precision", "roc_auc", "precision", "recall", "f1"}
        if not required.issubset(frame.columns):
            raise ArtifactUnavailable("Experiment registry has an invalid schema")
        return frame

    @cached_property
    def validation_frame(self) -> pd.DataFrame:
        for path in (self.paths.selected_predictions, self.paths.validation_data):
            if not path.is_file():
                raise ArtifactUnavailable(
                    f"Validation transaction artifact is not available: {path.name}"
                )
        prediction_columns = [
            "TransactionID",
            "actual_label",
            "fraud_probability",
            "predicted_label_at_0_5",
            "experiment_id",
            "model_version",
        ]
        validation_columns = [
            "TransactionID",
            "TransactionDT",
            "isFraud",
            *FEATURE_SCHEMA,
        ]
        try:
            predictions = pd.read_parquet(self.paths.selected_predictions, columns=prediction_columns)
            validation = pd.read_parquet(self.paths.validation_data, columns=validation_columns)
        except (OSError, ValueError, KeyError) as exc:
            raise ArtifactUnavailable("Validation transaction artifacts are unreadable") from exc
        selected = self.selected_candidate
        expected_rows = int(selected["validation_rows"])
        expected_experiment = str(selected["experiment_id"])
        if len(predictions) != expected_rows or predictions["TransactionID"].nunique() != expected_rows:
            raise ArtifactUnavailable("Selected CatBoost validation predictions failed row validation")
        if set(predictions["experiment_id"].astype(str).unique()) != {expected_experiment}:
            raise ArtifactUnavailable("Selected CatBoost validation predictions have the wrong experiment")
        joined = predictions.merge(validation, on="TransactionID", how="left", validate="one_to_one")
        if len(joined) != expected_rows or joined["TransactionAmt"].isna().any():
            raise ArtifactUnavailable("Validation transaction feature join is incomplete")
        if not (joined["actual_label"].to_numpy() == joined["isFraud"].to_numpy()).all():
            raise ArtifactUnavailable("Validation transaction labels do not match predictions")
        joined["outcome"] = np.select(
            [
                (joined["actual_label"] == 1) & (joined["predicted_label_at_0_5"] == 1),
                (joined["actual_label"] == 0) & (joined["predicted_label_at_0_5"] == 1),
                (joined["actual_label"] == 1) & (joined["predicted_label_at_0_5"] == 0),
            ],
            ["TRUE_POSITIVE", "FALSE_POSITIVE", "FALSE_NEGATIVE"],
            default="TRUE_NEGATIVE",
        )
        return joined

    def project_status(self) -> dict[str, Any]:
        dataset = _required_mapping(self.eda, "dataset_validation", "eda_summary.json")
        baseline = _required_mapping(self.baseline, "best_experiment", "baseline_validation.json")
        candidate = self.selected_candidate
        threshold_status = "not_evaluated"
        if self.paths.threshold_analysis.is_file():
            threshold = _read_json(self.paths.threshold_analysis)
            if (
                threshold.get("selection_split") == "validation"
                and threshold.get("held_out_test_accessed") is False
            ):
                threshold_status = "validation_analysis_ready"
        final_status = "not_evaluated"
        if self.paths.final_metrics.is_file():
            final = _read_json(self.paths.final_metrics)
            if final.get("split") == "test" and final.get("evaluation_status") == "complete":
                final_status = "complete"
        return {
            "dataset": {
                "status": "ready",
                "name": "IEEE-CIS Fraud Detection",
                "transactions": int(dataset["transaction_rows"]),
                "fraud_transactions": int(dataset["fraud_rows"]),
                "legitimate_transactions": int(dataset["legitimate_rows"]),
                "fraud_prevalence": float(dataset["fraud_percentage"]) / 100,
                "identity_rows": int(dataset["identity_rows"]),
                "identity_coverage": float(dataset["identity_coverage"]) / 100,
            },
            "split": {
                "status": "ready",
                "strategy": str(self.split["strategy"]),
                "train_rows": int(self.split["train_rows"]),
                "validation_rows": int(self.split["validation_rows"]),
                "test_rows": int(self.split["test_rows"]),
                "train_fraction": float(self.split["train_fraction_actual"]),
                "validation_fraction": float(self.split["validation_fraction_actual"]),
                "test_fraction": float(self.split["test_fraction_actual"]),
                "train_transaction_dt_min": int(self.split["train_transaction_dt_min"]),
                "train_transaction_dt_max": int(self.split["train_transaction_dt_max"]),
                "validation_transaction_dt_min": int(self.split["validation_transaction_dt_min"]),
                "validation_transaction_dt_max": int(self.split["validation_transaction_dt_max"]),
                "test_transaction_dt_min": int(self.split["test_transaction_dt_min"]),
                "test_transaction_dt_max": int(self.split["test_transaction_dt_max"]),
                "test_status": "sealed",
            },
            "baseline": {
                "status": "evaluated_on_validation",
                "experiment_id": str(baseline["experiment_id"]),
            },
            "candidate_model": {
                "status": "validation_candidate",
                "name": str(candidate["model"]),
                "experiment_id": str(candidate["experiment_id"]),
            },
            "threshold_analysis": {"status": threshold_status},
            "rules": {"status": "pending"},
            "operational_thresholds": {
                "status": (
                    "provisional_validation_config"
                    if self.operating_config is not None
                    else "not_evaluated"
                )
            },
            "final_test": {"status": final_status, "test_status": "sealed"},
        }

    def model_comparison(self) -> dict[str, Any]:
        baseline = _required_mapping(self.baseline, "best_experiment", "baseline_validation.json")
        candidate = self.selected_candidate
        self._verify_experiment_registry(baseline)
        self._verify_experiment_registry(candidate)
        logistic_metrics = _metric_view(baseline)
        catboost_metrics = _metric_view(candidate)
        ap_improvement = (
            float(catboost_metrics["average_precision"])
            - float(logistic_metrics["average_precision"])
        ) / float(logistic_metrics["average_precision"])
        identity = _required_mapping(
            self.catboost_metadata, "identity_feature_decision", "catboost_candidate_metadata.json"
        )
        failure_rows = self.catboost.get("failure_slice_comparison")
        if not isinstance(failure_rows, list):
            raise ArtifactUnavailable("catboost_validation.json has no failure slice comparison")
        requested_slices = {
            "ProductCD=W",
            "TransactionAmt>=500",
            "card4=discover",
            "ProductCD=S",
        }
        failure_slices = [row for row in failure_rows if row.get("slice") in requested_slices]
        fn_amounts = _required_mapping(
            self.catboost, "selected_false_negative_amounts", "catboost_validation.json"
        )
        return {
            "status": "validation_results",
            "split": "validation",
            "held_out_test_status": "sealed",
            "threshold": self.classification_threshold,
            "logistic_regression": {
                "name": str(baseline["model"]),
                "experiment_id": str(baseline["experiment_id"]),
                "metrics": logistic_metrics,
            },
            "catboost": {
                "name": str(candidate["model"]),
                "experiment_id": str(candidate["experiment_id"]),
                "metrics": catboost_metrics,
            },
            "average_precision_relative_improvement": ap_improvement,
            "candidate_details": {
                "status": str(self.catboost_metadata["status"]),
                "feature_count": len(self.catboost_metadata["feature_names"]),
                "identity_fields_included": bool(identity["selected_identity_features"]),
                "class_weight": str(self.catboost_metadata["class_weight"]),
                "identity_ap_loss": float(identity["ap_loss_without_identity"]),
                "selection_reason": str(identity["reason"]),
            },
            "failure_analysis": {
                "label": "Validation analysis @ threshold 0.50",
                "slices": failure_slices,
                "false_negatives": {
                    "count": int(fn_amounts["count"]),
                    "transaction_amount_total": float(fn_amounts["total"]),
                    "transaction_amount_max": float(fn_amounts["max"]),
                },
            },
            "precision_recall_curves": self._precision_recall_curves(baseline, candidate),
            "provenance": (
                "Measured on the chronological validation partition. "
                "The held-out test set remains sealed."
            ),
        }

    def _verify_experiment_registry(self, record: dict[str, Any]) -> None:
        matched = self.experiments.loc[
            self.experiments["experiment_id"].astype(str) == str(record["experiment_id"])
        ]
        if len(matched) != 1:
            raise ArtifactUnavailable(
                f"Experiment registry does not contain exactly one {record['experiment_id']} row"
            )
        experiment = matched.iloc[0]
        for field in ("average_precision", "roc_auc", "precision", "recall", "f1"):
            if not np.isclose(float(experiment[field]), float(record[field]), rtol=0, atol=1e-12):
                raise ArtifactUnavailable(
                    f"Experiment registry metric mismatch for {record['experiment_id']}: {field}"
                )

    def _precision_recall_curves(
        self, baseline: dict[str, Any], candidate: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if not self.paths.baseline_predictions.is_file():
            raise ArtifactUnavailable("Logistic validation predictions are not available")
        try:
            logistic = pd.read_parquet(
                self.paths.baseline_predictions,
                columns=["actual_label", "fraud_probability", "experiment_id"],
            )
        except (OSError, ValueError, KeyError) as exc:
            raise ArtifactUnavailable("Logistic validation predictions are unreadable") from exc
        logistic = logistic.loc[logistic["experiment_id"] == baseline["experiment_id"]]
        candidate_frame = self.validation_frame
        sources = (
            (str(baseline["model"]), logistic),
            (str(candidate["model"]), candidate_frame),
        )
        curves: list[dict[str, Any]] = []
        for name, frame in sources:
            if len(frame) != int(candidate["validation_rows"]):
                raise ArtifactUnavailable(f"{name} validation predictions have an unexpected row count")
            precision, recall, _ = precision_recall_curve(
                frame["actual_label"].to_numpy(dtype=int),
                frame["fraud_probability"].to_numpy(dtype=float),
            )
            curve = pd.DataFrame({"recall": recall, "precision": precision})
            curve = curve.groupby("recall", as_index=False)["precision"].max().sort_values("recall")
            recall_grid = np.linspace(0, 1, 101)
            interpolated = np.interp(
                recall_grid,
                curve["recall"].to_numpy(dtype=float),
                curve["precision"].to_numpy(dtype=float),
            )
            points = [
                {"recall": float(recall_value), "precision": float(precision_value)}
                for recall_value, precision_value in zip(recall_grid, interpolated, strict=True)
            ]
            curves.append({"model": name, "points": points})
        return curves

    def feature_importance(self, limit: int) -> dict[str, Any]:
        if not self.paths.feature_importance.is_file():
            raise ArtifactUnavailable("Feature importance artifact is not available")
        try:
            frame = pd.read_csv(self.paths.feature_importance)
        except (OSError, ValueError) as exc:
            raise ArtifactUnavailable("Feature importance artifact is unreadable") from exc
        if not {"feature", "importance"}.issubset(frame.columns):
            raise ArtifactUnavailable("Feature importance artifact has an invalid schema")
        ordered = frame.sort_values(["importance", "feature"], ascending=[False, True]).head(limit)
        return {
            "status": "validation_candidate",
            "model": str(self.catboost_metadata["model_name"]),
            "items": [
                {"feature": str(row.feature), "importance": float(row.importance)}
                for row in ordered.itertuples(index=False)
            ],
            "note": "Feature importance reflects predictive association, not causation.",
        }

    def validation_transactions(
        self,
        *,
        page: int,
        page_size: int,
        filter_name: ValidationFilter,
        search: str | None,
    ) -> dict[str, Any]:
        filtered = self._filter_transactions(self.validation_frame, filter_name)
        if search:
            normalized = search.strip()
            filtered = filtered.loc[
                filtered["TransactionID"].astype(str).str.contains(normalized, regex=False)
            ]
        filtered = filtered.sort_values("TransactionID", kind="stable")
        total = len(filtered)
        start = (page - 1) * page_size
        page_frame = filtered.iloc[start : start + page_size]
        return {
            "status": "validation_examples",
            "split": "validation",
            "threshold": self.classification_threshold,
            "filter": filter_name,
            "page": page,
            "page_size": page_size,
            "total": total,
            "page_count": math.ceil(total / page_size) if total else 0,
            "items": [self._transaction_record(row) for _, row in page_frame.iterrows()],
        }

    def _filter_transactions(
        self, frame: pd.DataFrame, filter_name: ValidationFilter
    ) -> pd.DataFrame:
        masks = {
            "all": np.ones(len(frame), dtype=bool),
            "true_fraud": frame["actual_label"] == 1,
            "true_legitimate": frame["actual_label"] == 0,
            "true_positive": frame["outcome"] == "TRUE_POSITIVE",
            "false_positive": frame["outcome"] == "FALSE_POSITIVE",
            "false_negative": frame["outcome"] == "FALSE_NEGATIVE",
            "true_negative": frame["outcome"] == "TRUE_NEGATIVE",
            "high_risk": frame["fraud_probability"] >= self.classification_threshold,
            "high_value": frame["TransactionAmt"] >= HIGH_VALUE_THRESHOLD,
        }
        return frame.loc[masks[filter_name]]

    def _transaction_record(self, row: pd.Series) -> dict[str, Any]:
        outcome = str(row["outcome"])
        operating = self.operating_config
        business_decision = None
        estimated_decision_cost = None
        review_threshold = None
        block_threshold = None
        scenario_id = None
        scenario_name = None
        if operating is not None:
            review_threshold = float(operating["review_threshold"])
            block_threshold = float(operating["block_threshold"])
            scenario_id = str(operating["scenario"])
            scenario_name = str(operating["scenario_name"])
            probability = float(row["fraud_probability"])
            business_decision = (
                "BLOCK"
                if probability >= block_threshold
                else "REVIEW"
                if probability >= review_threshold
                else "APPROVE"
            )
            estimated_decision_cost = self._estimated_row_cost(
                actual_label=int(row["actual_label"]),
                amount=float(row["TransactionAmt"]),
                decision=business_decision,
                assumptions=dict(operating["cost_assumptions"]),
            )
        return {
            "transaction_id": str(int(row["TransactionID"])),
            "transaction_dt": int(row["TransactionDT"]),
            "transaction_amount": float(row["TransactionAmt"]),
            "actual_label": int(row["actual_label"]),
            "fraud_probability": float(row["fraud_probability"]),
            "predicted_label_at_0_5": int(row["predicted_label_at_0_5"]),
            "outcome": outcome,
            "model_error": outcome in {"FALSE_POSITIVE", "FALSE_NEGATIVE"},
            "business_decision": business_decision,
            "review_threshold": review_threshold,
            "block_threshold": block_threshold,
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "estimated_decision_cost": estimated_decision_cost,
            "features": {feature: _native(row[feature]) for feature in FEATURE_SCHEMA},
        }

    def validation_transaction_for_scoring(self, transaction_id: str) -> dict[str, Any]:
        normalized = transaction_id.strip()
        if not normalized:
            raise LookupError("Validation transaction was not found")
        matched = self.validation_frame.loc[
            self.validation_frame["TransactionID"].astype(str).str.replace(r"\.0$", "", regex=True)
            == normalized
        ]
        if len(matched) != 1:
            raise LookupError("Validation transaction was not found")
        row = matched.iloc[0]
        return {
            "status": "validation_demo",
            "split": "validation",
            "held_out_test_status": "sealed",
            "transaction_id": str(int(row["TransactionID"])),
            "transaction_dt": int(row["TransactionDT"]),
            "features": {feature: _native(row[feature]) for feature in FEATURE_SCHEMA},
            "ground_truth_revealed": False,
            "note": "Ground truth is withheld until explicitly revealed.",
        }

    def validation_ground_truth(self, transaction_id: str) -> dict[str, Any]:
        normalized = transaction_id.strip()
        matched = self.validation_frame.loc[
            self.validation_frame["TransactionID"].astype(str).str.replace(r"\.0$", "", regex=True)
            == normalized
        ]
        if len(matched) != 1:
            raise LookupError("Validation transaction was not found")
        row = matched.iloc[0]
        actual_label = int(row["actual_label"])
        return {
            "transaction_id": str(int(row["TransactionID"])),
            "split": "validation",
            "actual_label": actual_label,
            "ground_truth": "FRAUD" if actual_label else "LEGITIMATE",
            "note": "Explicitly revealed validation label. The held-out test remains sealed.",
        }

    def risk_check_cases(self) -> dict[str, Any]:
        frame = self.validation_frame
        operating = self.operating_config
        if operating is None:
            raise ArtifactUnavailable("Provisional validation thresholds are not available")
        review = float(operating["review_threshold"])
        block = float(operating["block_threshold"])
        candidates = (
            (
                "highest_amount",
                "Highest transaction amount",
                "A high-value validation row.",
                frame.sort_values(["TransactionAmt", "TransactionID"], ascending=[False, True]),
            ),
            (
                "near_review_threshold",
                "Near the review boundary",
                "A score close to the provisional review threshold.",
                frame.assign(distance=(frame["fraud_probability"] - review).abs()).sort_values(
                    ["distance", "TransactionID"]
                ),
            ),
            (
                "near_block_threshold",
                "Near the block boundary",
                "A score close to the provisional block threshold.",
                frame.assign(distance=(frame["fraud_probability"] - block).abs()).sort_values(
                    ["distance", "TransactionID"]
                ),
            ),
            (
                "categorical_missingness",
                "Missing categorical values",
                "A validation row that exercises training-time missing-value handling.",
                frame.assign(
                    missing_count=frame[["ProductCD", "card4", "card6", "P_emaildomain"]]
                    .isna()
                    .sum(axis=1)
                ).sort_values(["missing_count", "TransactionID"], ascending=[False, True]),
            ),
        )
        used: set[int] = set()
        cases = []
        for case_type, label, description, ordered in candidates:
            selected = next(
                (row for _, row in ordered.iterrows() if int(row["TransactionID"]) not in used),
                ordered.iloc[0],
            )
            transaction_id = int(selected["TransactionID"])
            used.add(transaction_id)
            cases.append(
                {
                    "case_type": case_type,
                    "label": label,
                    "description": description,
                    "transaction_id": str(transaction_id),
                    "transaction_amount": float(selected["TransactionAmt"]),
                }
            )
        return {
            "status": "validation_demo",
            "split": "validation",
            "held_out_test_status": "sealed",
            "ground_truth_hidden": True,
            "cases": cases,
        }

    @staticmethod
    def _estimated_row_cost(
        *, actual_label: int, amount: float, decision: str, assumptions: dict[str, Any]
    ) -> float:
        fraud_cost = (
            amount * float(assumptions["fraud_loss_fraction"])
            + float(assumptions["chargeback_fixed_cost"])
        )
        legitimate_cost = (
            amount * float(assumptions["legitimate_margin_rate"])
            + float(assumptions["false_positive_fixed_cost"])
        )
        if decision == "APPROVE":
            return fraud_cost if actual_label == 1 else 0.0
        if decision == "BLOCK":
            return legitimate_cost if actual_label == 0 else 0.0
        residual = (
            fraud_cost * (1 - float(assumptions["review_fraud_catch_rate"]))
            if actual_label == 1
            else legitimate_cost
            * (1 - float(assumptions["review_legitimate_approval_rate"]))
        )
        return float(assumptions["manual_review_cost"]) + residual

    def interesting_cases(self) -> dict[str, Any]:
        frame = self.validation_frame
        cases = {
            "highest_value_false_negative": frame.loc[frame["outcome"] == "FALSE_NEGATIVE"].sort_values(
                ["TransactionAmt", "TransactionID"], ascending=[False, True]
            ),
            "highest_confidence_false_positive": frame.loc[
                frame["outcome"] == "FALSE_POSITIVE"
            ].sort_values(["fraud_probability", "TransactionID"], ascending=[False, True]),
            "highest_confidence_true_fraud": frame.loc[frame["actual_label"] == 1].sort_values(
                ["fraud_probability", "TransactionID"], ascending=[False, True]
            ),
            "highest_confidence_legitimate": frame.loc[frame["actual_label"] == 0].sort_values(
                ["fraud_probability", "TransactionID"], ascending=[True, True]
            ),
        }
        if any(case.empty for case in cases.values()):
            raise ArtifactUnavailable("Validation predictions do not contain every interesting case type")
        return {
            "status": "validation_examples",
            "split": "validation",
            "cases": [
                {"case_type": case_type, **self._transaction_record(case.iloc[0])}
                for case_type, case in cases.items()
            ],
        }


@lru_cache(maxsize=1)
def get_project_artifact_service() -> ProjectArtifactService:
    return ProjectArtifactService(configured_artifact_paths())
