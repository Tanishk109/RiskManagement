from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .split import SplitBoundaries, TemporalSplits

REQUIRED_MODEL_COLUMNS = ("TransactionID", "TransactionDT", "TransactionAmt", "isFraud")


def write_processed_splits(
    splits: TemporalSplits,
    destination: str | Path,
    *,
    feature_names: list[str],
    dataset_validation: dict[str, Any],
) -> dict[str, Any]:
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        frame.to_parquet(output / f"{name}.parquet", index=False, compression="zstd")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "local IEEE-CIS labeled train files",
        "format": "parquet",
        "feature_names": feature_names,
        "dataset_validation": dataset_validation,
        "split_boundaries": splits.boundaries.to_dict(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_processed_splits(directory: str | Path, feature_names: list[str]) -> TemporalSplits:
    source = Path(directory)
    paths = {
        "train": source / "train.parquet",
        "validation": source / "validation.parquet",
        "test": source / "test.parquet",
    }
    manifest_path = source / "manifest.json"
    missing = [str(path) for path in (*paths.values(), manifest_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Processed temporal datasets are missing. Run `make prepare-data`; missing: "
            + ", ".join(missing)
        )
    columns = list(dict.fromkeys([*REQUIRED_MODEL_COLUMNS, *feature_names]))
    frames = {name: pd.read_parquet(path, columns=columns) for name, path in paths.items()}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boundaries = SplitBoundaries(**manifest["split_boundaries"])
    if frames["train"]["TransactionDT"].max() > frames["validation"]["TransactionDT"].min():
        raise ValueError("Processed train and validation data are not chronological")
    if frames["validation"]["TransactionDT"].max() > frames["test"]["TransactionDT"].min():
        raise ValueError("Processed validation and test data are not chronological")
    ids = {name: set(frame["TransactionID"]) for name, frame in frames.items()}
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise ValueError("Processed temporal datasets contain overlapping TransactionID values")
    return TemporalSplits(
        train=frames["train"],
        validation=frames["validation"],
        test=frames["test"],
        boundaries=boundaries,
    )
