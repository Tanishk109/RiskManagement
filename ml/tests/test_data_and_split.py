from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from merchantshield_ml.data import load_ieee_cis
from merchantshield_ml.processed import (
    load_baseline_partition,
    load_baseline_partitions,
    load_processed_splits,
    write_processed_splits,
)
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
    assert report.fraud_rows == 5
    assert report.legitimate_rows == 7
    assert report.matched_identity_rows == 6
    assert report.identity_coverage == 50.0
    assert report.identity_coverage_fraud == 100.0
    assert report.identity_coverage_legitimate == pytest.approx(100 / 7)
    assert report.transaction_amount_summary["min"] == 19.5
    assert report.transaction_amount_summary["median"] == 112.0
    assert report.transaction_amount_summary["max"] == 2200.0
    assert frame["identity_available"].dtype == bool
    assert int(frame["identity_available"].sum()) == 6
    assert frame["DeviceType"].isna().sum() == 6


def test_loader_requires_product_code_for_dataset_validation(tmp_path):
    transaction_path = tmp_path / "train_transaction.csv"
    transaction_path.write_text(
        "TransactionID,TransactionDT,TransactionAmt,isFraud\n1,10,20.0,0\n",
        encoding="utf-8",
    )

    try:
        load_ieee_cis(transaction_path, FIXTURES / "train_identity.csv")
    except ValueError as exc:
        assert "ProductCD" in str(exc)
    else:
        raise AssertionError("Expected ProductCD validation to fail")


def test_temporal_split_is_ordered_and_has_no_overlap():
    frame, _ = load_ieee_cis(
        FIXTURES / "train_transaction.csv",
        FIXTURES / "train_identity.csv",
        feature_names=["TransactionAmt"],
    )
    splits = temporal_split(frame)
    assert splits.train["TransactionDT"].max() < splits.validation["TransactionDT"].min()
    assert splits.validation["TransactionDT"].max() < splits.test["TransactionDT"].min()
    assert set(splits.train["TransactionID"]).isdisjoint(splits.validation["TransactionID"])
    assert set(splits.train["TransactionID"]).isdisjoint(splits.test["TransactionID"])
    assert set(splits.validation["TransactionID"]).isdisjoint(splits.test["TransactionID"])
    combined_ids = set(splits.train["TransactionID"]) | set(splits.validation["TransactionID"]) | set(
        splits.test["TransactionID"]
    )
    assert combined_ids == set(frame["TransactionID"])
    assert len(splits.train) + len(splits.validation) + len(splits.test) == len(frame)


def test_temporal_split_requires_fractions_to_sum_to_one():
    frame, _ = load_ieee_cis(
        FIXTURES / "train_transaction.csv",
        FIXTURES / "train_identity.csv",
        feature_names=["TransactionAmt"],
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        temporal_split(frame, train_fraction=0.7, validation_fraction=0.2, test_fraction=0.2)


def test_temporal_split_is_deterministic():
    frame, _ = load_ieee_cis(
        FIXTURES / "train_transaction.csv",
        FIXTURES / "train_identity.csv",
        feature_names=["TransactionAmt"],
    )
    first = temporal_split(frame.sample(frac=1, random_state=1))
    second = temporal_split(frame.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)
    assert first.metadata == second.metadata


def test_temporal_split_keeps_identical_timestamps_together():
    frame = pd.DataFrame(
        {
            "TransactionID": list(range(1, 10)),
            "TransactionDT": [10, 10, 10, 20, 20, 20, 30, 30, 30],
            "isFraud": [0, 0, 1, 0, 1, 0, 1, 0, 0],
        }
    )
    splits = temporal_split(frame)
    timestamp_partitions = [
        set(splits.train["TransactionDT"]),
        set(splits.validation["TransactionDT"]),
        set(splits.test["TransactionDT"]),
    ]
    assert timestamp_partitions == [{10}, {20}, {30}]
    assert splits.metadata.clean_timestamp_boundaries is True


def test_identity_available_uses_join_presence_not_nullable_feature(tmp_path):
    transaction_path = tmp_path / "train_transaction.csv"
    identity_path = tmp_path / "train_identity.csv"
    transaction_path.write_text(
        "TransactionID,TransactionDT,TransactionAmt,ProductCD,isFraud\n"
        "1,10,20.0,W,0\n2,20,25.0,H,1\n3,30,30.0,C,0\n",
        encoding="utf-8",
    )
    identity_path.write_text("TransactionID,DeviceType\n1,\n3,mobile\n", encoding="utf-8")

    frame, _ = load_ieee_cis(transaction_path, identity_path, feature_names=["DeviceType"])

    assert frame.set_index("TransactionID")["identity_available"].to_dict() == {1: True, 2: False, 3: True}
    assert pd.isna(frame.set_index("TransactionID").loc[1, "DeviceType"])


def test_processed_splits_round_trip_as_parquet(tmp_path):
    frame, validation = load_ieee_cis(
        FIXTURES / "train_transaction.csv",
        FIXTURES / "train_identity.csv",
        feature_names=["ProductCD", "DeviceType"],
    )
    splits = temporal_split(frame, train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2)
    manifest = write_processed_splits(
        splits,
        tmp_path,
        feature_names=["ProductCD", "DeviceType"],
        dataset_validation=validation.to_dict(),
        descriptive_analysis={},
    )
    loaded = load_processed_splits(tmp_path, ["ProductCD", "DeviceType"])

    assert manifest["format"] == "parquet"
    assert manifest["compression"] == "zstd"
    assert (tmp_path / "split_metadata.json").is_file()
    assert len(loaded.train) == len(splits.train)
    assert len(loaded.validation) == len(splits.validation)
    assert len(loaded.test) == len(splits.test)
    assert loaded.boundaries == splits.boundaries
    assert loaded.metadata == splits.metadata


def test_baseline_loader_rejects_held_out_test_partition(tmp_path):
    with pytest.raises(ValueError, match="may load only train, validation"):
        load_baseline_partition(tmp_path, "test", ["TransactionAmt"])


def test_baseline_loader_does_not_require_or_read_test_parquet(tmp_path):
    frame, validation = load_ieee_cis(
        FIXTURES / "train_transaction.csv",
        FIXTURES / "train_identity.csv",
        feature_names=["ProductCD", "DeviceType"],
    )
    splits = temporal_split(frame, train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2)
    write_processed_splits(
        splits,
        tmp_path,
        feature_names=["ProductCD", "DeviceType"],
        dataset_validation=validation.to_dict(),
        descriptive_analysis={},
    )
    (tmp_path / "test.parquet").unlink()

    loaded = load_baseline_partitions(tmp_path, ["ProductCD", "DeviceType"])

    assert len(loaded.train) == len(splits.train)
    assert len(loaded.validation) == len(splits.validation)
    assert loaded.train["TransactionDT"].max() < loaded.validation["TransactionDT"].min()
