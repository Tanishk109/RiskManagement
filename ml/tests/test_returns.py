from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from merchantshield_ml.returns import (
    RETURN_FEATURES,
    build_order_dataset,
    chronological_return_split,
    normalize_return_features,
    return_binary_metrics,
    return_risk_level,
)


def _lines(order_count: int = 24) -> pd.DataFrame:
    rows = []
    for index in range(order_count):
        cancelled = index in {4, 11, 19}
        invoice = f"C{index:05d}" if cancelled else f"{index:06d}"
        rows.append(
            {
                "invoice_id": invoice,
                "stock_code": f"SKU{index % 3}",
                "quantity": -2 if cancelled else 2,
                "order_datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "unit_price": 5.0,
                "customer_id": "CUSTOMER-A" if index < 20 else np.nan,
                "country": "United Kingdom",
            }
        )
    return pd.DataFrame(rows)


def test_customer_features_use_prior_orders_only() -> None:
    orders = build_order_dataset(_lines())
    first = orders.iloc[0]
    after_first_cancellation = orders.iloc[5]
    assert first["prior_order_count"] == 0
    assert first["prior_cancellation_rate"] == 0
    assert after_first_cancellation["prior_order_count"] == 5
    assert after_first_cancellation["prior_cancellation_rate"] == pytest.approx(1 / 5)
    assert orders.iloc[4]["prior_cancellation_rate"] == 0


def test_missing_customers_do_not_share_history() -> None:
    orders = build_order_dataset(_lines())
    assert (orders.loc[orders.index >= 20, "prior_order_count"] == 0).all()


def test_signed_cancellation_quantity_is_not_a_feature_leak() -> None:
    orders = build_order_dataset(_lines())
    cancellation = orders.loc[orders["is_cancellation_proxy"] == 1].iloc[0]
    assert cancellation["quantity"] == 2
    assert "invoice_id" not in RETURN_FEATURES
    assert "is_cancellation_proxy" not in RETURN_FEATURES


def test_split_is_chronological_and_disjoint() -> None:
    split = chronological_return_split(build_order_dataset(_lines()))
    assert split.train["order_datetime"].max() <= split.validation["order_datetime"].min()
    assert split.validation["order_datetime"].max() <= split.test["order_datetime"].min()
    ids = [set(part["invoice_id"]) for part in (split.train, split.validation, split.test)]
    assert ids[0].isdisjoint(ids[1]) and ids[1].isdisjoint(ids[2]) and ids[0].isdisjoint(ids[2])


def test_normalization_rejects_missing_schema() -> None:
    with pytest.raises(ValueError, match="missing features"):
        normalize_return_features(pd.DataFrame({"order_value": [10]}))


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.149, "LOW"), (0.15, "MEDIUM"), (0.499, "MEDIUM"), (0.5, "HIGH")],
)
def test_risk_band_boundaries(probability: float, expected: str) -> None:
    assert return_risk_level(probability) == expected


def test_metrics_are_computed_from_supplied_labels() -> None:
    metrics = return_binary_metrics(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.2, 0.7]))
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
