from __future__ import annotations

import json

from common import PROCESSED_DATA, RAW_DATA, feature_sets, training_config
from merchantshield_ml.data import load_ieee_cis
from merchantshield_ml.processed import write_processed_splits
from merchantshield_ml.split import temporal_split


def main() -> None:
    configured_sets = feature_sets()
    features = list(dict.fromkeys(feature for group in configured_sets.values() for feature in group))
    frame, validation = load_ieee_cis(
        RAW_DATA / "train_transaction.csv",
        RAW_DATA / "train_identity.csv",
        feature_names=features,
    )
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"Configured features are missing from the joined IEEE-CIS data: {', '.join(missing)}")
    config = training_config()
    splits = temporal_split(
        frame,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    manifest = write_processed_splits(
        splits,
        PROCESSED_DATA,
        feature_names=features,
        dataset_validation=validation.to_dict(),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
