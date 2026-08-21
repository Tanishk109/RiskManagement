from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_preprocessor, validate_feature_schema


def build_baseline_pipeline(train: pd.DataFrame, features: list[str], *, random_state: int = 42) -> Pipeline:
    validate_feature_schema(train, features)
    return Pipeline([
        ("preprocessor", build_preprocessor(train, features, scale_numeric=True)),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=random_state,
            solver="liblinear",
        )),
    ])


def fit_baseline(train: pd.DataFrame, features: list[str], *, random_state: int = 42) -> Pipeline:
    pipeline = build_baseline_pipeline(train, features, random_state=random_state)
    pipeline.fit(train[features], train["isFraud"].astype(int))
    return pipeline
