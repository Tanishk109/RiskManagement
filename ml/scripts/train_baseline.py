from __future__ import annotations

import json
import time

import joblib
from common import ARTIFACTS, feature_sets, load_splits, training_config
from merchantshield_ml.baseline import fit_baseline
from merchantshield_ml.evaluate import binary_metrics


def main() -> None:
    config = training_config()
    feature_set_name = str(config["baseline_feature_set"])
    features = feature_sets()[feature_set_name]
    splits = load_splits(features)
    started = time.perf_counter()
    pipeline = fit_baseline(splits.train, features, random_state=int(config["random_seed"]))
    training_seconds = time.perf_counter() - started
    scores = pipeline.predict_proba(splits.validation[features])[:, 1]
    metrics = binary_metrics(splits.validation["isFraud"].to_numpy(), scores)
    artifact = {
        "model": "logistic_regression",
        "split": "validation",
        "feature_set": feature_set_name,
        "feature_names": features,
        "training_seconds": training_seconds,
        **metrics,
    }
    metrics_dir = ARTIFACTS / "metrics"
    models_dir = ARTIFACTS / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "baseline_validation.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    joblib.dump({"pipeline": pipeline, "feature_names": features, "split": "validation"}, models_dir / "baseline_validation.joblib")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
