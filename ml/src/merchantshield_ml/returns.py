from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

RETURN_MODEL_VERSION = "returns-catboost-uci-v1"
RETURN_DATA_SOURCE = "UCI Online Retail II (dataset 502)"
RETURN_PROXY_DISCLOSURE = (
    "The dataset provides cancellation/reversal outcomes and is used as a proxy for "
    "return-risk research; it is not a perfect physical-return label."
)
RETURN_FEATURES = (
    "order_value",
    "quantity",
    "unique_stock_count",
    "country",
    "stock_code",
    "prior_order_count",
    "prior_cancellation_rate",
    "prior_average_order_value",
    "order_hour",
    "order_day_of_week",
)
RETURN_CATEGORICAL_FEATURES = ("country", "stock_code")
RETURN_NUMERIC_FEATURES = tuple(
    feature for feature in RETURN_FEATURES if feature not in RETURN_CATEGORICAL_FEATURES
)
RETURN_MISSING_CATEGORY = "__MISSING__"
RETURN_MEDIUM_THRESHOLD = 0.15
RETURN_HIGH_THRESHOLD = 0.50


@dataclass(frozen=True)
class ChronologicalReturnSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _string_token(value: object) -> str:
    if pd.isna(value):
        return RETURN_MISSING_CATEGORY
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    token = str(value).strip()
    return token or RETURN_MISSING_CATEGORY


def build_order_dataset(lines: pd.DataFrame) -> pd.DataFrame:
    """Aggregate UCI line items to orders and create strictly prior customer history."""

    required = {
        "invoice_id",
        "stock_code",
        "quantity",
        "order_datetime",
        "unit_price",
        "customer_id",
        "country",
    }
    missing = sorted(required.difference(lines.columns))
    if missing:
        raise ValueError(f"Retail input is missing columns: {', '.join(missing)}")

    frame = lines[list(required)].copy()
    frame["invoice_id"] = frame["invoice_id"].map(_string_token)
    frame["stock_code"] = frame["stock_code"].map(_string_token)
    frame["country"] = frame["country"].map(_string_token)
    frame["customer_id"] = frame["customer_id"].map(_string_token)
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame["unit_price"] = pd.to_numeric(frame["unit_price"], errors="coerce")
    frame["order_datetime"] = pd.to_datetime(frame["order_datetime"], errors="coerce")
    frame = frame.loc[
        frame["order_datetime"].notna()
        & frame["quantity"].notna()
        & frame["unit_price"].notna()
        & frame["invoice_id"].ne(RETURN_MISSING_CATEGORY)
    ].copy()
    if frame.empty:
        raise ValueError("Retail input contains no valid invoice lines")

    # Signed cancellation quantities are transformed to magnitudes so the label is not leaked.
    frame["absolute_quantity"] = frame["quantity"].abs()
    frame["absolute_line_value"] = frame["quantity"].abs() * frame["unit_price"].abs()
    frame["is_cancellation_proxy"] = (
        frame["invoice_id"].str.upper().str.startswith("C").astype("int8")
    )

    orders = (
        frame.groupby("invoice_id", sort=False, as_index=False)
        .agg(
            order_datetime=("order_datetime", "min"),
            customer_id=("customer_id", "first"),
            country=("country", "first"),
            stock_code=("stock_code", "first"),
            quantity=("absolute_quantity", "sum"),
            order_value=("absolute_line_value", "sum"),
            unique_stock_count=("stock_code", "nunique"),
            is_cancellation_proxy=("is_cancellation_proxy", "max"),
        )
        .sort_values(["order_datetime", "invoice_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    missing_customer = orders["customer_id"].eq(RETURN_MISSING_CATEGORY)
    orders["history_key"] = orders["customer_id"].where(
        ~missing_customer,
        "__MISSING_CUSTOMER__" + orders["invoice_id"],
    )
    history = orders.groupby("history_key", sort=False)
    orders["prior_order_count"] = history.cumcount().astype("int64")
    orders["prior_cancellation_count"] = history["is_cancellation_proxy"].cumsum() - orders[
        "is_cancellation_proxy"
    ]
    orders["prior_order_value_total"] = history["order_value"].cumsum() - orders["order_value"]
    denominator = orders["prior_order_count"].replace(0, np.nan)
    orders["prior_cancellation_rate"] = (
        orders["prior_cancellation_count"] / denominator
    ).fillna(0.0)
    orders["prior_average_order_value"] = (
        orders["prior_order_value_total"] / denominator
    ).fillna(0.0)
    orders["order_hour"] = orders["order_datetime"].dt.hour.astype("int16")
    orders["order_day_of_week"] = orders["order_datetime"].dt.dayofweek.astype("int8")
    return orders[
        [
            "invoice_id",
            "order_datetime",
            *RETURN_FEATURES,
            "is_cancellation_proxy",
        ]
    ].copy()


def chronological_return_split(
    orders: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> ChronologicalReturnSplit:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test partition")
    if len(orders) < 20:
        raise ValueError("At least 20 chronological orders are required")
    ordered = orders.sort_values(["order_datetime", "invoice_id"], kind="mergesort").reset_index(
        drop=True
    )
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(len(ordered) * (train_fraction + validation_fraction))
    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()
    if train["order_datetime"].max() > validation["order_datetime"].min():
        raise AssertionError("Train and validation partitions are not chronological")
    if validation["order_datetime"].max() > test["order_datetime"].min():
        raise AssertionError("Validation and test partitions are not chronological")
    return ChronologicalReturnSplit(train=train, validation=validation, test=test)


def normalize_return_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [feature for feature in RETURN_FEATURES if feature not in frame.columns]
    if missing:
        raise ValueError(f"Return model input is missing features: {', '.join(missing)}")
    normalized = frame[list(RETURN_FEATURES)].copy()
    for feature in RETURN_CATEGORICAL_FEATURES:
        normalized[feature] = normalized[feature].map(_string_token)
    for feature in RETURN_NUMERIC_FEATURES:
        normalized[feature] = pd.to_numeric(normalized[feature], errors="raise").astype(float)
    return normalized


def return_risk_level(probability: float) -> Literal["LOW", "MEDIUM", "HIGH"]:
    if not 0 <= probability <= 1:
        raise ValueError("Return-risk probability must be between zero and one")
    if probability >= RETURN_HIGH_THRESHOLD:
        return "HIGH"
    if probability >= RETURN_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def return_binary_metrics(
    labels: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float = RETURN_HIGH_THRESHOLD,
) -> dict[str, float | int]:
    # Metric calculation is used during offline evaluation, not runtime
    # inference. Lazy loading keeps sklearn/scipy out of the deployed API.
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support

    truth = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": float(average_precision_score(truth, scores)),
        "threshold": float(threshold),
        "positives": int(truth.sum()),
        "negatives": int(len(truth) - truth.sum()),
        "predicted_positive": int(predictions.sum()),
        "true_positives": int(((truth == 1) & (predictions == 1)).sum()),
        "false_positives": int(((truth == 0) & (predictions == 1)).sum()),
        "false_negatives": int(((truth == 1) & (predictions == 0)).sum()),
        "true_negatives": int(((truth == 0) & (predictions == 0)).sum()),
    }
