from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ORDERING_COLUMN = "TransactionDT"
SECONDARY_ORDERING_COLUMN = "TransactionID"
CLEAN_BOUNDARY_POLICY = "nearest clean TransactionDT boundary; ties choose the earlier cut"
FALLBACK_BOUNDARY_POLICY = "stable row cut ordered by TransactionDT then TransactionID"


@dataclass(frozen=True)
class SplitBoundaries:
    train_rows: int
    validation_rows: int
    test_rows: int
    train_dt_min: float
    train_dt_max: float
    validation_dt_min: float
    validation_dt_max: float
    test_dt_min: float
    test_dt_max: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class SplitMetadata:
    strategy: str
    ordering_column: str
    secondary_ordering_column: str
    boundary_policy: str
    clean_timestamp_boundaries: bool
    train_fraction_requested: float
    validation_fraction_requested: float
    test_fraction_requested: float
    train_fraction_actual: float
    validation_fraction_actual: float
    test_fraction_actual: float
    total_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    train_transaction_dt_min: float
    train_transaction_dt_max: float
    validation_transaction_dt_min: float
    validation_transaction_dt_max: float
    test_transaction_dt_min: float
    test_transaction_dt_max: float
    train_fraud_count: int
    validation_fraud_count: int
    test_fraud_count: int
    train_legitimate_count: int
    validation_legitimate_count: int
    test_legitimate_count: int
    train_fraud_rate: float
    validation_fraud_rate: float
    test_fraud_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SplitMetadata:
        names = {field.name for field in fields(cls)}
        missing = sorted(names.difference(payload))
        if missing:
            raise ValueError(f"Split metadata is missing fields: {', '.join(missing)}")
        return cls(**{name: payload[name] for name in names})


@dataclass(frozen=True)
class TemporalSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    boundaries: SplitBoundaries
    metadata: SplitMetadata


def _validate_fractions(train_fraction: float, validation_fraction: float, test_fraction: float) -> None:
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(not 0 < fraction < 1 for fraction in fractions):
        raise ValueError("Train, validation, and test fractions must each be between 0 and 1")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test fractions must sum to 1.0")


def _nearest_position(positions: list[int], target: int) -> int:
    return min(positions, key=lambda position: (abs(position - target), position))


def _split_positions(
    ordered: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[int, int, bool, str]:
    row_count = len(ordered)
    train_target = round(row_count * train_fraction)
    validation_target = round(row_count * (train_fraction + validation_fraction))
    transaction_dt = ordered[ORDERING_COLUMN].to_numpy()
    clean_positions = (np.flatnonzero(transaction_dt[1:] != transaction_dt[:-1]) + 1).tolist()

    if len(clean_positions) >= 2:
        train_end = _nearest_position(clean_positions[:-1], train_target)
        validation_end = _nearest_position(
            [position for position in clean_positions if position > train_end],
            validation_target,
        )
        return train_end, validation_end, True, CLEAN_BOUNDARY_POLICY

    train_end = min(max(1, train_target), row_count - 2)
    validation_end = min(max(train_end + 1, validation_target), row_count - 1)
    return train_end, validation_end, False, FALLBACK_BOUNDARY_POLICY


def _partition_fraud_stats(frame: pd.DataFrame) -> tuple[int, int, float]:
    fraud_count = int(frame["isFraud"].sum())
    legitimate_count = len(frame) - fraud_count
    return fraud_count, legitimate_count, float(fraud_count / len(frame))


def _validate_partition_integrity(
    source: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    clean_timestamp_boundaries: bool,
) -> None:
    if len(train) + len(validation) + len(test) != len(source):
        raise AssertionError("Temporal split lost or added transaction rows")
    if not train[ORDERING_COLUMN].is_monotonic_increasing:
        raise AssertionError("Training partition is not chronologically ordered")
    if not validation[ORDERING_COLUMN].is_monotonic_increasing:
        raise AssertionError("Validation partition is not chronologically ordered")
    if not test[ORDERING_COLUMN].is_monotonic_increasing:
        raise AssertionError("Held-out test partition is not chronologically ordered")

    train_max = train[ORDERING_COLUMN].max()
    validation_min = validation[ORDERING_COLUMN].min()
    validation_max = validation[ORDERING_COLUMN].max()
    test_min = test[ORDERING_COLUMN].min()
    if train_max > validation_min or validation_max > test_min:
        raise AssertionError("Temporal partitions overlap in time")
    if clean_timestamp_boundaries and (train_max >= validation_min or validation_max >= test_min):
        raise AssertionError("A clean timestamp boundary split identical timestamps")

    source_ids = set(source[SECONDARY_ORDERING_COLUMN])
    partition_ids = {
        "train": set(train[SECONDARY_ORDERING_COLUMN]),
        "validation": set(validation[SECONDARY_ORDERING_COLUMN]),
        "test": set(test[SECONDARY_ORDERING_COLUMN]),
    }
    if (
        partition_ids["train"] & partition_ids["validation"]
        or partition_ids["train"] & partition_ids["test"]
        or partition_ids["validation"] & partition_ids["test"]
    ):
        raise AssertionError("Temporal partitions contain overlapping TransactionID values")
    if set().union(*partition_ids.values()) != source_ids:
        raise AssertionError("Temporal partitions do not cover every source TransactionID")


def temporal_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> TemporalSplits:
    required = {SECONDARY_ORDERING_COLUMN, ORDERING_COLUMN, "isFraud"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot split data; missing columns: {', '.join(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("TransactionID, TransactionDT, and isFraud must not be missing before splitting")
    if frame[SECONDARY_ORDERING_COLUMN].duplicated().any():
        raise ValueError("TransactionID must be unique before splitting")
    if len(frame) < 3:
        raise ValueError("At least three rows are required for train/validation/test")
    _validate_fractions(train_fraction, validation_fraction, test_fraction)

    ordered = frame.sort_values([ORDERING_COLUMN, SECONDARY_ORDERING_COLUMN], kind="stable").reset_index(drop=True)
    train_end, validation_end, clean_boundaries, boundary_policy = _split_positions(
        ordered,
        train_fraction,
        validation_fraction,
    )
    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()
    _validate_partition_integrity(
        ordered,
        train,
        validation,
        test,
        clean_timestamp_boundaries=clean_boundaries,
    )

    boundaries = SplitBoundaries(
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        train_dt_min=float(train[ORDERING_COLUMN].min()),
        train_dt_max=float(train[ORDERING_COLUMN].max()),
        validation_dt_min=float(validation[ORDERING_COLUMN].min()),
        validation_dt_max=float(validation[ORDERING_COLUMN].max()),
        test_dt_min=float(test[ORDERING_COLUMN].min()),
        test_dt_max=float(test[ORDERING_COLUMN].max()),
    )
    train_fraud, train_legitimate, train_fraud_rate = _partition_fraud_stats(train)
    validation_fraud, validation_legitimate, validation_fraud_rate = _partition_fraud_stats(validation)
    test_fraud, test_legitimate, test_fraud_rate = _partition_fraud_stats(test)
    metadata = SplitMetadata(
        strategy="chronological",
        ordering_column=ORDERING_COLUMN,
        secondary_ordering_column=SECONDARY_ORDERING_COLUMN,
        boundary_policy=boundary_policy,
        clean_timestamp_boundaries=clean_boundaries,
        train_fraction_requested=train_fraction,
        validation_fraction_requested=validation_fraction,
        test_fraction_requested=test_fraction,
        train_fraction_actual=len(train) / len(ordered),
        validation_fraction_actual=len(validation) / len(ordered),
        test_fraction_actual=len(test) / len(ordered),
        total_rows=len(ordered),
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        train_transaction_dt_min=boundaries.train_dt_min,
        train_transaction_dt_max=boundaries.train_dt_max,
        validation_transaction_dt_min=boundaries.validation_dt_min,
        validation_transaction_dt_max=boundaries.validation_dt_max,
        test_transaction_dt_min=boundaries.test_dt_min,
        test_transaction_dt_max=boundaries.test_dt_max,
        train_fraud_count=train_fraud,
        validation_fraud_count=validation_fraud,
        test_fraud_count=test_fraud,
        train_legitimate_count=train_legitimate,
        validation_legitimate_count=validation_legitimate,
        test_legitimate_count=test_legitimate,
        train_fraud_rate=train_fraud_rate,
        validation_fraud_rate=validation_fraud_rate,
        test_fraud_rate=test_fraud_rate,
    )
    return TemporalSplits(
        train=train,
        validation=validation,
        test=test,
        boundaries=boundaries,
        metadata=metadata,
    )


def write_split_boundaries(boundaries: SplitBoundaries, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(boundaries.to_dict(), indent=2) + "\n", encoding="utf-8")
