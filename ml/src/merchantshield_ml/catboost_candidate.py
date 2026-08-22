from __future__ import annotations

import platform
import time
import warnings
from dataclasses import dataclass
from typing import Any

import catboost
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier

from .evaluate import binary_metrics
from .features import validate_feature_schema

CATBOOST_MODEL_NAME = "CatBoostClassifier"
CATBOOST_MODEL_VERSION = "catboost-validation-v1"
CATEGORICAL_MISSING_VALUE = "__MISSING__"
DEFAULT_EVALUATION_THRESHOLD = 0.5
CATBOOST_CATEGORICAL_FEATURES = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "identity_available",
    "DeviceType",
    "DeviceInfo",
    "R_emaildomain",
)
IDENTITY_FEATURES = (
    "identity_available",
    "DeviceType",
    "DeviceInfo",
    "R_emaildomain",
)
CATBOOST_EXPERIMENT_FIELDS = (
    "experiment_id",
    "model",
    "model_version",
    "feature_set",
    "class_weight",
    "number_of_features_before_encoding",
    "number_of_encoded_features",
    "training_rows",
    "train_rows",
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
    "best_iteration",
    "actual_tree_count",
    "parameters",
    "identity_features_included",
    "converged",
    "notes",
)


@dataclass
class CatBoostExperimentRun:
    record: dict[str, Any]
    model: CatBoostClassifier
    features: list[str]
    categorical_features: list[str]
    fraud_probabilities: np.ndarray
    predicted_labels: np.ndarray
    fit_warnings: list[str]


def validate_catboost_feature_schema(
    frame: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
) -> None:
    validate_feature_schema(frame, features)
    unexpected = sorted(set(categorical_features).difference(features))
    if unexpected:
        raise ValueError(f"CatBoost categorical fields are absent from the feature set: {', '.join(unexpected)}")
    numeric_features = [feature for feature in features if feature not in categorical_features]
    non_numeric = [feature for feature in numeric_features if not pd.api.types.is_numeric_dtype(frame[feature])]
    if non_numeric:
        raise TypeError(f"CatBoost numerical fields must be numeric: {', '.join(non_numeric)}")


def normalize_catboost_features(
    frame: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """Prepare native CatBoost inputs without fitting state or target-derived transforms."""

    validate_catboost_feature_schema(frame, features, categorical_features)
    prepared = frame[features].copy()
    for feature in categorical_features:
        values = prepared[feature].astype("object")
        prepared[feature] = values.where(values.notna(), CATEGORICAL_MISSING_VALUE).map(str)
    if prepared[categorical_features].isna().any().any():
        raise AssertionError("Categorical normalization left missing values")
    return prepared


def build_catboost_classifier(
    parameters: dict[str, Any],
    *,
    auto_class_weights: str | None,
    random_seed: int,
) -> CatBoostClassifier:
    model_parameters = dict(parameters)
    model_parameters.pop("early_stopping_rounds", None)
    model_parameters.update(
        {
            "random_seed": random_seed,
            "auto_class_weights": auto_class_weights,
            "verbose": False,
        }
    )
    return CatBoostClassifier(**model_parameters)


def validate_catboost_experiment_record(record: dict[str, Any]) -> None:
    missing = [field for field in CATBOOST_EXPERIMENT_FIELDS if field not in record]
    if missing:
        raise ValueError(f"CatBoost experiment record is missing fields: {', '.join(missing)}")
    if record["model"] != CATBOOST_MODEL_NAME:
        raise ValueError("CatBoost experiment record has an unexpected model name")
    if int(record["training_rows"]) <= 0 or int(record["validation_rows"]) <= 0:
        raise ValueError("CatBoost experiment row counts must be positive")


def run_catboost_experiment(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    experiment_id: str,
    feature_set: str,
    features: list[str],
    categorical_features: list[str],
    auto_class_weights: str | None,
    parameters: dict[str, Any],
    random_seed: int,
    threshold: float = DEFAULT_EVALUATION_THRESHOLD,
) -> CatBoostExperimentRun:
    train_features = normalize_catboost_features(train, features, categorical_features)
    validation_features = normalize_catboost_features(validation, features, categorical_features)
    model = build_catboost_classifier(
        parameters,
        auto_class_weights=auto_class_weights,
        random_seed=random_seed,
    )
    early_stopping_rounds = int(parameters["early_stopping_rounds"])

    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(
            train_features,
            train["isFraud"].astype(int),
            cat_features=categorical_features,
            eval_set=(validation_features, validation["isFraud"].astype(int)),
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True,
        )
    training_seconds = time.perf_counter() - started
    fit_warnings = [f"{item.category.__name__}: {item.message}" for item in caught]

    inference_started = time.perf_counter()
    fraud_probabilities = np.asarray(model.predict_proba(validation_features)[:, 1], dtype=float)
    inference_seconds = time.perf_counter() - inference_started
    metrics = binary_metrics(validation["isFraud"].to_numpy(), fraud_probabilities, threshold=threshold)
    predicted_labels = (fraud_probabilities >= threshold).astype(int)
    best_iteration = int(model.get_best_iteration())
    tree_count = int(model.tree_count_)
    configured_iterations = int(parameters["iterations"])
    warning_note = "No fit warnings." if not fit_warnings else "Fit warnings captured: " + " | ".join(fit_warnings)
    weight_name = auto_class_weights or "none"
    record = {
        "experiment_id": experiment_id,
        "model": CATBOOST_MODEL_NAME,
        "model_version": CATBOOST_MODEL_VERSION,
        "feature_set": feature_set,
        "class_weight": weight_name,
        "number_of_features_before_encoding": len(features),
        "number_of_encoded_features": "not_applicable_native_categorical",
        "training_rows": len(train),
        "train_rows": len(train),
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
        "best_iteration": best_iteration,
        "actual_tree_count": tree_count,
        "parameters": parameters,
        "identity_features_included": all(feature in features for feature in IDENTITY_FEATURES),
        "converged": True,
        "notes": (
            "TRAIN fit and VALIDATION early stopping/evaluation only; Average Precision uses sklearn "
            f"average_precision_score; classification metrics use fixed threshold {threshold:.2f}; "
            f"CatBoost early stopping metric={parameters['eval_metric']}; tree_count={tree_count}/"
            f"{configured_iterations}. {warning_note}"
        ),
    }
    validate_catboost_experiment_record(record)
    return CatBoostExperimentRun(
        record=record,
        model=model,
        features=features,
        categorical_features=categorical_features,
        fraud_probabilities=fraud_probabilities,
        predicted_labels=predicted_labels,
        fit_warnings=fit_warnings,
    )


def select_best_catboost(runs: list[CatBoostExperimentRun]) -> CatBoostExperimentRun:
    if not runs:
        raise ValueError("At least one CatBoost experiment is required")
    return max(
        runs,
        key=lambda run: (
            float(run.record["average_precision"]),
            float(run.record["f1"]),
            float(run.record["recall"]),
            -int(run.record["actual_tree_count"]),
        ),
    )


def choose_identity_candidate(
    with_identity: CatBoostExperimentRun,
    without_identity: CatBoostExperimentRun,
    *,
    max_ap_loss: float,
    max_roc_auc_loss: float,
) -> tuple[CatBoostExperimentRun, dict[str, Any]]:
    ap_loss = float(with_identity.record["average_precision"]) - float(without_identity.record["average_precision"])
    roc_auc_loss = float(with_identity.record["roc_auc"]) - float(without_identity.record["roc_auc"])
    prefer_without = ap_loss <= max_ap_loss and roc_auc_loss <= max_roc_auc_loss
    decision = {
        "predeclared_max_ap_loss": max_ap_loss,
        "predeclared_max_roc_auc_loss": max_roc_auc_loss,
        "ap_loss_without_identity": ap_loss,
        "roc_auc_loss_without_identity": roc_auc_loss,
        "selected_identity_features": not prefer_without,
        "reason": (
            "Identity-free candidate stayed within both predeclared stability tolerances; prefer simpler features."
            if prefer_without
            else "Identity removal exceeded at least one predeclared stability tolerance; retain identity provisionally."
        ),
    }
    return (without_identity if prefer_without else with_identity), decision


def false_negative_amount_summary(
    validation: pd.DataFrame,
    predicted_labels: np.ndarray,
) -> dict[str, float | int]:
    mask = (validation["isFraud"].to_numpy(dtype=int) == 1) & (predicted_labels == 0)
    amounts = validation.loc[mask, "TransactionAmt"].dropna()
    return {
        "count": len(amounts),
        "total": float(amounts.sum()),
        "median": float(amounts.median()),
        "p90": float(amounts.quantile(0.90)),
        "p95": float(amounts.quantile(0.95)),
        "max": float(amounts.max()),
    }


def logistic_failure_slice_comparison(
    validation: pd.DataFrame,
    predicted_labels: np.ndarray,
) -> list[dict[str, float | int | str]]:
    labels = validation["isFraud"].to_numpy(dtype=int)
    slice_definitions = (
        ("ProductCD=W", validation["ProductCD"].eq("W").to_numpy(), 1_485, 0.0),
        ("TransactionAmt>=500", validation["TransactionAmt"].ge(500).to_numpy(), 224, 0.0),
        ("card4=discover", validation["card4"].eq("discover").to_numpy(), 122, 0.0),
        ("ProductCD=S", validation["ProductCD"].eq("S").to_numpy(), 107, 0.0),
        (
            "identity_available=False",
            validation["identity_available"].eq(False).to_numpy(),
            1_524,
            0.0032808398950131233,
        ),
        ("DeviceType=<MISSING>", validation["DeviceType"].isna().to_numpy(), 1_553, 0.004507405022537025),
    )
    rows = []
    for name, group_mask, frozen_support, logistic_recall in slice_definitions:
        fraud_mask = (labels == 1) & group_mask
        support = int(fraud_mask.sum())
        if support != frozen_support:
            raise ValueError(
                f"Frozen Logistic failure-slice support changed for {name}: expected {frozen_support}, got {support}"
            )
        recall = float(predicted_labels[fraud_mask].sum() / support)
        rows.append(
            {
                "slice": name,
                "fraud_support": support,
                "logistic_recall": logistic_recall,
                "catboost_recall": recall,
                "absolute_improvement": recall - logistic_recall,
            }
        )
    return rows


def prediction_artifact(
    validation: pd.DataFrame,
    run: CatBoostExperimentRun,
) -> pd.DataFrame:
    artifact = pd.DataFrame(
        {
            "TransactionID": validation["TransactionID"].to_numpy(),
            "actual_label": validation["isFraud"].astype(int).to_numpy(),
            "fraud_probability": run.fraud_probabilities,
            "predicted_label_at_0_5": run.predicted_labels,
            "experiment_id": run.record["experiment_id"],
            "model_version": CATBOOST_MODEL_VERSION,
        }
    )
    validate_prediction_artifact(artifact, expected_rows=len(validation))
    return artifact


def validate_prediction_artifact(artifact: pd.DataFrame, *, expected_rows: int) -> None:
    required = {
        "TransactionID",
        "actual_label",
        "fraud_probability",
        "predicted_label_at_0_5",
        "experiment_id",
        "model_version",
    }
    missing = sorted(required.difference(artifact.columns))
    if missing:
        raise ValueError(f"CatBoost prediction artifact is missing columns: {', '.join(missing)}")
    if len(artifact) != expected_rows:
        raise ValueError("CatBoost prediction artifact row count differs from validation")
    if artifact[list(required)].isna().any().any():
        raise ValueError("CatBoost prediction artifact contains missing values")
    if not artifact["fraud_probability"].between(0, 1).all():
        raise ValueError("CatBoost probabilities must be between zero and one")


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "catboost": catboost.__version__,
    }
