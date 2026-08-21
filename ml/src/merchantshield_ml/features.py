from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_feature_sets(path: str | Path) -> dict[str, list[str]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    feature_sets = payload.get("feature_sets")
    if not isinstance(feature_sets, dict) or not feature_sets:
        raise ValueError("Feature config must define at least one feature set")
    normalized: dict[str, list[str]] = {}
    for name, columns in feature_sets.items():
        if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
            raise ValueError(f"Feature set '{name}' must be a list of column names")
        normalized[str(name)] = columns
    return normalized


def validate_feature_schema(frame: pd.DataFrame, features: list[str]) -> None:
    forbidden = {"isFraud", "actual_label"}
    leaked = sorted(forbidden.intersection(features))
    if leaked:
        raise ValueError(f"Feature set contains label columns: {', '.join(leaked)}")
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"Feature set columns missing from data: {', '.join(missing)}")


def feature_types(frame: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    validate_feature_schema(frame, features)
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    return numeric, categorical


def build_preprocessor(frame: pd.DataFrame, features: list[str], *, scale_numeric: bool) -> ColumnTransformer:
    numeric, categorical = feature_types(frame, features)
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ], remainder="drop")
