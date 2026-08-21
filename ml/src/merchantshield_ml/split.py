from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


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
class TemporalSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    boundaries: SplitBoundaries


def temporal_split(frame: pd.DataFrame, *, train_fraction: float = 0.70, validation_fraction: float = 0.15) -> TemporalSplits:
    required = {"TransactionID", "TransactionDT", "isFraud"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot split data; missing columns: {', '.join(missing)}")
    if frame["TransactionID"].duplicated().any():
        raise ValueError("TransactionID must be unique before splitting")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must be positive and leave a non-empty test fraction")
    if len(frame) < 3:
        raise ValueError("At least three rows are required for train/validation/test")

    ordered = frame.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    train_end = max(1, int(len(ordered) * train_fraction))
    validation_end = max(train_end + 1, int(len(ordered) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(ordered) - 1)
    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()

    assert train["TransactionDT"].max() <= validation["TransactionDT"].min()
    assert validation["TransactionDT"].max() <= test["TransactionDT"].min()
    assert set(train["TransactionID"]).isdisjoint(validation["TransactionID"])
    assert set(train["TransactionID"]).isdisjoint(test["TransactionID"])
    assert set(validation["TransactionID"]).isdisjoint(test["TransactionID"])

    boundaries = SplitBoundaries(
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        train_dt_min=float(train["TransactionDT"].min()),
        train_dt_max=float(train["TransactionDT"].max()),
        validation_dt_min=float(validation["TransactionDT"].min()),
        validation_dt_max=float(validation["TransactionDT"].max()),
        test_dt_min=float(test["TransactionDT"].min()),
        test_dt_max=float(test["TransactionDT"].max()),
    )
    return TemporalSplits(train=train, validation=validation, test=test, boundaries=boundaries)


def write_split_boundaries(boundaries: SplitBoundaries, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(boundaries.to_dict(), indent=2) + "\n", encoding="utf-8")
