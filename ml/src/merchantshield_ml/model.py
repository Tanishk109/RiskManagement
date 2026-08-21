from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .features import build_preprocessor, validate_feature_schema

DEFAULT_XGB_PARAMS: dict[str, int | float | str] = {
    "n_estimators": 450,
    "max_depth": 6,
    "learning_rate": 0.06,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "reg_lambda": 2.0,
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": -1,
}


def build_primary_pipeline(
    train: pd.DataFrame,
    features: list[str],
    *,
    random_state: int = 42,
    parameters: dict[str, int | float | str] | None = None,
) -> Pipeline:
    validate_feature_schema(train, features)
    labels = train["isFraud"].astype(int)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0:
        raise ValueError("Training data contains no fraud labels")
    params = {**DEFAULT_XGB_PARAMS, **(parameters or {})}
    classifier = XGBClassifier(
        **params,
        scale_pos_weight=negatives / positives,
        random_state=random_state,
    )
    return Pipeline([
        ("preprocessor", build_preprocessor(train, features, scale_numeric=False)),
        ("classifier", classifier),
    ])


def fit_primary(
    train: pd.DataFrame,
    features: list[str],
    *,
    random_state: int = 42,
    parameters: dict[str, int | float | str] | None = None,
) -> Pipeline:
    pipeline = build_primary_pipeline(train, features, random_state=random_state, parameters=parameters)
    pipeline.fit(train[features], train["isFraud"].astype(int))
    return pipeline
