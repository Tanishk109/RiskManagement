from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .split import SplitBoundaries, SplitMetadata, TemporalSplits

REQUIRED_PROCESSED_COLUMNS = (
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "isFraud",
    "identity_available",
)
SPLIT_METADATA_FILENAME = "split_metadata.json"
BASELINE_ALLOWED_SPLITS = frozenset({"train", "validation"})


@dataclass(frozen=True)
class BaselinePartitions:
    train: pd.DataFrame
    validation: pd.DataFrame
    metadata: dict[str, Any]


def _validate_processed_frames(splits: TemporalSplits) -> None:
    frames = (splits.train, splits.validation, splits.test)
    for name, frame in zip(("train", "validation", "test"), frames, strict=True):
        missing = sorted(set(REQUIRED_PROCESSED_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"Cannot write {name} partition; missing columns: {', '.join(missing)}")
        if frame["TransactionID"].duplicated().any():
            raise ValueError(f"Cannot write {name} partition with duplicate TransactionID values")
    if sum(len(frame) for frame in frames) != splits.metadata.total_rows:
        raise ValueError("Processed partitions do not match split metadata row count")


def write_processed_splits(
    splits: TemporalSplits,
    destination: str | Path,
    *,
    feature_names: list[str],
    dataset_validation: dict[str, Any],
    descriptive_analysis: dict[str, Any],
    compression: str = "zstd",
) -> dict[str, Any]:
    _validate_processed_frames(splits)
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    parquet_files: dict[str, dict[str, int | str]] = {}
    for name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        path = output / f"{name}.parquet"
        frame.to_parquet(path, index=False, compression=compression)
        parquet_files[name] = {
            "path": path.name,
            "rows": len(frame),
            "bytes": path.stat().st_size,
        }

    metadata = {
        **splits.metadata.to_dict(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "local IEEE-CIS labeled train_transaction.csv and train_identity.csv",
        "merge_policy": "left join identity on TransactionID; no transaction rows dropped",
        "identity_available_definition": "TransactionID matched a row in train_identity.csv",
        "format": "parquet",
        "compression": compression,
        "feature_names": feature_names,
        "retained_columns": list(splits.train.columns),
        "dataset_validation": dataset_validation,
        "split_boundaries": splits.boundaries.to_dict(),
        "parquet_files": parquet_files,
        "descriptive_analysis": descriptive_analysis,
        "ml_performance": {
            "model_trained": False,
            "held_out_test_used_for_tuning": False,
            "precision": "Not evaluated yet",
            "recall": "Not evaluated yet",
            "f1": "Not evaluated yet",
            "pr_auc": "Not evaluated yet",
            "false_positives": "Not evaluated yet",
            "false_negatives": "Not evaluated yet",
            "estimated_cost": "Not evaluated yet",
        },
    }
    (output / SPLIT_METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_processed_splits(directory: str | Path, feature_names: list[str]) -> TemporalSplits:
    source = Path(directory)
    paths = {
        "train": source / "train.parquet",
        "validation": source / "validation.parquet",
        "test": source / "test.parquet",
    }
    metadata_path = source / SPLIT_METADATA_FILENAME
    missing = [str(path) for path in (*paths.values(), metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Processed temporal datasets are missing. Run `make prepare-data`; missing: " + ", ".join(missing)
        )
    columns = list(dict.fromkeys([*REQUIRED_PROCESSED_COLUMNS, *feature_names]))
    frames = {name: pd.read_parquet(path, columns=columns) for name, path in paths.items()}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = SplitMetadata.from_dict(payload)
    boundaries = SplitBoundaries(**payload["split_boundaries"])
    if len(frames["train"]) != metadata.train_rows:
        raise ValueError("Processed train row count differs from split metadata")
    if len(frames["validation"]) != metadata.validation_rows:
        raise ValueError("Processed validation row count differs from split metadata")
    if len(frames["test"]) != metadata.test_rows:
        raise ValueError("Processed held-out test row count differs from split metadata")
    if frames["train"]["TransactionDT"].max() > frames["validation"]["TransactionDT"].min():
        raise ValueError("Processed train and validation data are not chronological")
    if frames["validation"]["TransactionDT"].max() > frames["test"]["TransactionDT"].min():
        raise ValueError("Processed validation and test data are not chronological")
    if metadata.clean_timestamp_boundaries:
        if frames["train"]["TransactionDT"].max() >= frames["validation"]["TransactionDT"].min():
            raise ValueError("Processed train and validation data split an identical timestamp")
        if frames["validation"]["TransactionDT"].max() >= frames["test"]["TransactionDT"].min():
            raise ValueError("Processed validation and test data split an identical timestamp")
    ids = {name: set(frame["TransactionID"]) for name, frame in frames.items()}
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise ValueError("Processed temporal datasets contain overlapping TransactionID values")
    if len(set().union(*ids.values())) != metadata.total_rows:
        raise ValueError("Processed temporal datasets do not cover every source TransactionID")
    return TemporalSplits(
        train=frames["train"],
        validation=frames["validation"],
        test=frames["test"],
        boundaries=boundaries,
        metadata=metadata,
    )


def load_baseline_partition(
    directory: str | Path,
    split: str,
    feature_names: list[str],
) -> pd.DataFrame:
    """Load only a model-selection partition; held-out test access is forbidden."""

    if split not in BASELINE_ALLOWED_SPLITS:
        allowed = ", ".join(sorted(BASELINE_ALLOWED_SPLITS))
        raise ValueError(f"Baseline selection may load only {allowed}; received {split!r}")
    source = Path(directory)
    path = source / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Processed {split} partition is missing: {path}")
    columns = list(dict.fromkeys(["TransactionID", "TransactionDT", "isFraud", *feature_names]))
    return pd.read_parquet(path, columns=columns)


def load_baseline_partitions(directory: str | Path, feature_names: list[str]) -> BaselinePartitions:
    source = Path(directory)
    metadata_path = source / SPLIT_METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Processed split metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train = load_baseline_partition(source, "train", feature_names)
    validation = load_baseline_partition(source, "validation", feature_names)
    if len(train) != int(metadata["train_rows"]):
        raise ValueError("Baseline train row count differs from frozen split metadata")
    if len(validation) != int(metadata["validation_rows"]):
        raise ValueError("Baseline validation row count differs from frozen split metadata")
    if train["TransactionDT"].max() >= validation["TransactionDT"].min():
        raise ValueError("Baseline train/validation partitions are not strictly chronological")
    return BaselinePartitions(train=train, validation=validation, metadata=metadata)
