from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_preprocessor, validate_feature_schema


def build_baseline_pipeline(
    train: pd.DataFrame,
    features: list[str],
    *,
    class_weight: str | dict[int, float] | None = "balanced",
    random_state: int = 42,
    solver: str = "newton-cholesky",
    max_iter: int = 100,
) -> Pipeline:
    validate_feature_schema(train, features)
    return Pipeline([
        ("preprocessor", build_preprocessor(train, features, scale_numeric=True)),
        ("classifier", LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
            solver=solver,
        )),
    ])


def fit_baseline(
    train: pd.DataFrame,
    features: list[str],
    *,
    class_weight: str | dict[int, float] | None = "balanced",
    random_state: int = 42,
    solver: str = "newton-cholesky",
    max_iter: int = 100,
) -> Pipeline:
    pipeline = build_baseline_pipeline(
        train,
        features,
        class_weight=class_weight,
        random_state=random_state,
        solver=solver,
        max_iter=max_iter,
    )
    pipeline.fit(train[features], train["isFraud"].astype(int))
    return pipeline
