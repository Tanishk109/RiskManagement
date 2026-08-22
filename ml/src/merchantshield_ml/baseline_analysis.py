from __future__ import annotations

import platform
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from .baseline import build_baseline_pipeline
from .evaluate import binary_metrics
from .features import validate_feature_schema

DEFAULT_EVALUATION_THRESHOLD = 0.5
MINIMUM_SLICE_FRAUD_SUPPORT = 50
MODEL_NAME = "Logistic Regression"
MODEL_VERSION = "logistic-baseline-v1"
EXPERIMENT_FIELDS = (
    "experiment_id",
    "model",
    "feature_set",
    "class_weight",
    "number_of_features_before_encoding",
    "number_of_encoded_features",
    "training_rows",
    "validation_rows",
    "precision",
    "recall",
    "f1",
    "average_precision",
    "roc_auc",
    "tp",
    "fp",
    "tn",
    "fn",
    "predicted_fraud_count",
    "predicted_fraud_rate",
    "accuracy",
    "brier_score",
    "default_threshold",
    "training_seconds",
    "inference_seconds",
    "solver",
    "max_iter",
    "converged",
    "notes",
)


@dataclass
class BaselineExperimentRun:
    record: dict[str, Any]
    pipeline: Pipeline
    fraud_probabilities: np.ndarray
    predicted_labels: np.ndarray
    encoded_feature_names: list[str]
    fit_warnings: list[str]
    convergence_warnings: list[str]


def _class_weight_name(class_weight: str | dict[int, float] | None) -> str:
    if class_weight is None:
        return "none"
    return str(class_weight)


def validate_experiment_record(record: dict[str, Any]) -> None:
    missing = [field for field in EXPERIMENT_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Experiment record is missing fields: {', '.join(missing)}")
    if record["model"] != MODEL_NAME:
        raise ValueError("Experiment record model must be Logistic Regression")
    if int(record["training_rows"]) <= 0 or int(record["validation_rows"]) <= 0:
        raise ValueError("Experiment record row counts must be positive")


def run_logistic_experiment(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    experiment_id: str,
    feature_set: str,
    features: list[str],
    class_weight: str | dict[int, float] | None,
    random_state: int,
    solver: str,
    max_iter: int,
    threshold: float = DEFAULT_EVALUATION_THRESHOLD,
) -> BaselineExperimentRun:
    validate_feature_schema(train, features)
    validate_feature_schema(validation, features)
    pipeline = build_baseline_pipeline(
        train,
        features,
        class_weight=class_weight,
        random_state=random_state,
        solver=solver,
        max_iter=max_iter,
    )

    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(train[features], train["isFraud"].astype(int))
    training_seconds = time.perf_counter() - started
    fit_warnings = [f"{item.category.__name__}: {item.message}" for item in caught]
    convergence_warnings = [
        message
        for message, item in zip(fit_warnings, caught, strict=True)
        if issubclass(item.category, ConvergenceWarning)
    ]

    inference_started = time.perf_counter()
    fraud_probabilities = pipeline.predict_proba(validation[features])[:, 1]
    inference_seconds = time.perf_counter() - inference_started
    metrics = binary_metrics(validation["isFraud"].to_numpy(), fraud_probabilities, threshold=threshold)
    predicted_labels = (fraud_probabilities >= threshold).astype(int)
    preprocessor = pipeline.named_steps["preprocessor"]
    encoded_feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    classifier = pipeline.named_steps["classifier"]
    iterations = np.asarray(classifier.n_iter_, dtype=int)
    converged = not convergence_warnings and bool(np.all(iterations < max_iter))
    warning_note = "No fit warnings." if not fit_warnings else "Fit warnings captured: " + " | ".join(fit_warnings)
    record = {
        "experiment_id": experiment_id,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "feature_set": feature_set,
        "class_weight": _class_weight_name(class_weight),
        "number_of_features_before_encoding": len(features),
        "number_of_encoded_features": len(encoded_feature_names),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "average_precision": metrics["average_precision"],
        "roc_auc": metrics["roc_auc"],
        "tp": metrics["true_positives"],
        "fp": metrics["false_positives"],
        "tn": metrics["true_negatives"],
        "fn": metrics["false_negatives"],
        "predicted_fraud_count": metrics["predicted_fraud_count"],
        "predicted_fraud_rate": metrics["predicted_fraud_rate"],
        "accuracy": metrics["accuracy"],
        "brier_score": metrics["brier_score"],
        "default_threshold": threshold,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "solver": solver,
        "max_iter": max_iter,
        "iterations": [int(value) for value in iterations],
        "converged": converged,
        "notes": (
            "Validation only; Average Precision uses sklearn average_precision_score; "
            f"classification metrics use fixed threshold {threshold:.2f}. {warning_note}"
        ),
    }
    validate_experiment_record(record)
    return BaselineExperimentRun(
        record=record,
        pipeline=pipeline,
        fraud_probabilities=fraud_probabilities,
        predicted_labels=predicted_labels,
        encoded_feature_names=encoded_feature_names,
        fit_warnings=fit_warnings,
        convergence_warnings=convergence_warnings,
    )


def select_best_experiment(runs: list[BaselineExperimentRun]) -> BaselineExperimentRun:
    if not runs:
        raise ValueError("At least one Logistic Regression experiment is required")
    return max(
        runs,
        key=lambda run: (
            float(run.record["average_precision"]),
            float(run.record["f1"]),
            float(run.record["recall"]),
        ),
    )


def metric_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    metrics = ("precision", "recall", "f1", "average_precision", "roc_auc")
    return {metric: float(candidate[metric]) - float(reference[metric]) for metric in metrics}


def categorical_cardinality(train: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, int]]:
    return {
        column: {
            "non_null_categories": int(train[column].nunique(dropna=True)),
            "missing_rows": int(train[column].isna().sum()),
        }
        for column in columns
    }


def coefficient_summary(run: BaselineExperimentRun, limit: int = 20) -> dict[str, list[dict[str, float | str]]]:
    coefficients = np.asarray(run.pipeline.named_steps["classifier"].coef_[0], dtype=float)
    if len(coefficients) != len(run.encoded_feature_names):
        raise AssertionError("Encoded feature names and coefficient count differ")
    pairs = list(zip(run.encoded_feature_names, coefficients, strict=True))
    positive = sorted(pairs, key=lambda item: item[1], reverse=True)[:limit]
    negative = sorted(pairs, key=lambda item: item[1])[:limit]
    return {
        "largest_positive": [{"feature": feature, "coefficient": float(value)} for feature, value in positive],
        "largest_negative": [{"feature": feature, "coefficient": float(value)} for feature, value in negative],
    }


def _amount_summary(frame: pd.DataFrame) -> dict[str, int | float]:
    amount = frame["TransactionAmt"].dropna()
    return {
        "count": len(amount),
        "total": float(amount.sum()),
        "mean": float(amount.mean()),
        "median": float(amount.median()),
        "p90": float(amount.quantile(0.90)),
        "p95": float(amount.quantile(0.95)),
        "max": float(amount.max()),
    }


def _categorical_distribution(frame: pd.DataFrame, column: str, limit: int = 10) -> list[dict[str, int | float | str]]:
    values = frame[column].astype("object").where(frame[column].notna(), "<MISSING>")
    counts = values.value_counts(dropna=False).head(limit)
    return [
        {
            "category": str(category),
            "count": int(count),
            "share_percent": float(100 * count / len(frame)) if len(frame) else 0.0,
        }
        for category, count in counts.items()
    ]


def _recall_slices(
    validation: pd.DataFrame,
    predicted_labels: np.ndarray,
    column: str,
    *,
    minimum_fraud_support: int,
) -> list[dict[str, int | float | str]]:
    working = validation[[column, "isFraud"]].copy()
    working["predicted_label"] = predicted_labels
    working["slice"] = working[column].astype("object").where(working[column].notna(), "<MISSING>")
    fraud = working[working["isFraud"] == 1]
    grouped = fraud.groupby("slice", dropna=False)["predicted_label"].agg(["size", "sum"])
    grouped = grouped[grouped["size"] >= minimum_fraud_support]
    rows = [
        {
            "slice": column,
            "category": str(category),
            "fraud_support": int(row["size"]),
            "true_positives": int(row["sum"]),
            "false_negatives": int(row["size"] - row["sum"]),
            "recall": float(row["sum"] / row["size"]),
        }
        for category, row in grouped.iterrows()
    ]
    return sorted(rows, key=lambda row: (float(row["recall"]), -int(row["fraud_support"])))


def validation_error_analysis(
    validation: pd.DataFrame,
    run: BaselineExperimentRun,
    *,
    minimum_slice_fraud_support: int = MINIMUM_SLICE_FRAUD_SUPPORT,
) -> dict[str, Any]:
    working = validation.copy()
    working["fraud_probability"] = run.fraud_probabilities
    working["predicted_label_at_0_5"] = run.predicted_labels
    labels = working["isFraud"].astype(int)
    predictions = working["predicted_label_at_0_5"].astype(int)
    masks = {
        "false_negatives": (labels == 1) & (predictions == 0),
        "true_positives": (labels == 1) & (predictions == 1),
        "false_positives": (labels == 0) & (predictions == 1),
        "true_negatives": (labels == 0) & (predictions == 0),
    }
    categorical_columns = ["ProductCD", "card4", "card6", "identity_available", "DeviceType"]
    outcomes = {
        name: {
            "row_count": int(mask.sum()),
            "transaction_amount": _amount_summary(working[mask]),
            "categorical_distribution": {
                column: _categorical_distribution(working[mask], column) for column in categorical_columns
            },
        }
        for name, mask in masks.items()
    }

    false_negatives = working[masks["false_negatives"]]
    high_value_columns = [
        "TransactionID",
        "TransactionAmt",
        "fraud_probability",
        "ProductCD",
        "card4",
        "card6",
        "identity_available",
        "DeviceType",
    ]
    high_value = false_negatives.nlargest(10, "TransactionAmt")[high_value_columns].copy()
    high_value_examples = [
        {
            key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
            for key, value in row.items()
        }
        for row in high_value.to_dict(orient="records")
    ]

    slice_rows: list[dict[str, int | float | str]] = []
    for column in categorical_columns:
        slice_rows.extend(
            _recall_slices(
                working,
                run.predicted_labels,
                column,
                minimum_fraud_support=minimum_slice_fraud_support,
            )
        )

    amount_labels = ["0–25", "25–50", "50–100", "100–250", "250–500", "500+"]
    amount_slice_column = "TransactionAmt_fixed_bucket"
    amount_slices = working.copy()
    amount_slices[amount_slice_column] = pd.cut(
        amount_slices["TransactionAmt"],
        bins=[-np.inf, 25, 50, 100, 250, 500, np.inf],
        labels=amount_labels,
        right=False,
    ).astype("object")
    slice_rows.extend(
        _recall_slices(
            amount_slices,
            run.predicted_labels,
            amount_slice_column,
            minimum_fraud_support=minimum_slice_fraud_support,
        )
    )
    slice_rows.sort(key=lambda row: (float(row["recall"]), -int(row["fraud_support"]), str(row["slice"])))
    return {
        "default_threshold": DEFAULT_EVALUATION_THRESHOLD,
        "minimum_slice_fraud_support": minimum_slice_fraud_support,
        "amount_bucket_definition": "Fixed, left-closed ranges: 0–25, 25–50, 50–100, 100–250, 250–500, 500+",
        "outcomes": outcomes,
        "high_value_false_negative_examples": high_value_examples,
        "recall_slices": slice_rows,
        "weakest_meaningful_slices": slice_rows[:15],
    }


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
