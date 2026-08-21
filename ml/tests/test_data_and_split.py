from __future__ import annotations

from pathlib import Path

from merchantshield_ml.data import load_ieee_cis
from merchantshield_ml.split import temporal_split

FIXTURES = Path(__file__).parent / "fixtures"


def test_loader_left_joins_identity_without_dropping_transactions():
    frame, report = load_ieee_cis(
        FIXTURES / "train_transaction.csv",
        FIXTURES / "train_identity.csv",
        feature_names=["TransactionAmt", "ProductCD", "DeviceType"],
    )
    assert len(frame) == 12
    assert report.transaction_rows == 12
    assert report.identity_rows == 6
    assert report.identity_coverage == 50.0
    assert frame["DeviceType"].isna().sum() == 6


def test_temporal_split_is_ordered_and_has_no_overlap():
    frame, _ = load_ieee_cis(
        FIXTURES / "train_transaction.csv",
        FIXTURES / "train_identity.csv",
        feature_names=["TransactionAmt"],
    )
    splits = temporal_split(frame)
    assert splits.train["TransactionDT"].max() <= splits.validation["TransactionDT"].min()
    assert splits.validation["TransactionDT"].max() <= splits.test["TransactionDT"].min()
    assert set(splits.train["TransactionID"]).isdisjoint(splits.validation["TransactionID"])
    assert set(splits.train["TransactionID"]).isdisjoint(splits.test["TransactionID"])
    assert set(splits.validation["TransactionID"]).isdisjoint(splits.test["TransactionID"])
