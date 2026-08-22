from __future__ import annotations

import json

from common import (
    ARTIFACTS,
    IEEE_CIS_RAW_DATA,
    PROCESSED_DATA,
    feature_sets,
    training_config,
)
from merchantshield_ml.data import load_ieee_cis
from merchantshield_ml.processed import write_processed_splits
from merchantshield_ml.split import temporal_split
from merchantshield_ml.split_analysis import (
    build_temporal_descriptive_analysis,
    write_temporal_split_report,
)


def main() -> None:
    configured_sets = feature_sets()
    features = list(dict.fromkeys(feature for group in configured_sets.values() for feature in group))
    retained_features = [*features, "identity_available"]
    frame, validation = load_ieee_cis(
        IEEE_CIS_RAW_DATA / "train_transaction.csv",
        IEEE_CIS_RAW_DATA / "train_identity.csv",
        feature_names=retained_features,
    )
    missing = sorted(set(retained_features).difference(frame.columns))
    if missing:
        raise ValueError(f"Configured features are missing from the joined IEEE-CIS data: {', '.join(missing)}")
    config = training_config()
    splits = temporal_split(
        frame,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
        test_fraction=float(config["test_fraction"]),
    )
    descriptive_analysis = build_temporal_descriptive_analysis(splits)
    metadata = write_processed_splits(
        splits,
        PROCESSED_DATA,
        feature_names=retained_features,
        dataset_validation=validation.to_dict(),
        descriptive_analysis=descriptive_analysis,
    )
    write_temporal_split_report(metadata, ARTIFACTS / "reports/temporal_split.md")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
