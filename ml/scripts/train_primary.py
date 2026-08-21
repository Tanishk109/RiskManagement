from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone

import joblib
from common import (
    ARTIFACTS,
    ROOT,
    feature_sets,
    load_splits,
    replace_marked_section,
    training_config,
)
from merchantshield_ml.evaluate import binary_metrics
from merchantshield_ml.model import fit_primary


def main() -> None:
    config = training_config()
    available_sets = feature_sets()
    candidates = [str(name) for name in config["primary_feature_sets"]]
    union_features = list(dict.fromkeys(feature for name in candidates for feature in available_sets[name]))
    splits = load_splits(union_features)
    experiments: list[dict[str, object]] = []
    trained: dict[str, object] = {}

    for index, feature_set_name in enumerate(candidates, start=1):
        features = available_sets[feature_set_name]
        started = time.perf_counter()
        pipeline = fit_primary(
            splits.train,
            features,
            random_state=int(config["random_seed"]),
            parameters=dict(config["primary_parameters"]),
        )
        training_seconds = time.perf_counter() - started
        scores = pipeline.predict_proba(splits.validation[features])[:, 1]
        metrics = binary_metrics(splits.validation["isFraud"].to_numpy(), scores)
        experiment = {
            "experiment_id": f"xgb-{index:02d}",
            "model": "xgboost",
            "features": feature_set_name,
            "parameters": json.dumps(config["primary_parameters"], sort_keys=True),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "average_precision": metrics["average_precision"],
            "training_seconds": training_seconds,
            "notes": "Validation only; temporal split; no test access",
        }
        experiments.append(experiment)
        trained[feature_set_name] = pipeline

    selected = max(experiments, key=lambda row: float(row["average_precision"]))
    selected_name = str(selected["features"])
    metrics_dir = ARTIFACTS / "metrics"
    models_dir = ARTIFACTS / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    with (metrics_dir / "experiments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(experiments[0]))
        writer.writeheader()
        writer.writerows(experiments)

    selection = {
        "selection_split": "validation",
        "selection_metric": "average_precision",
        "selected_model": "xgboost",
        "selected_feature_set": selected_name,
        "feature_names": available_sets[selected_name],
        "selected_experiment": selected,
        "split_boundaries": splits.boundaries.to_dict(),
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
    }
    (metrics_dir / "validation_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    joblib.dump({
        "pipeline": trained[selected_name],
        "feature_names": available_sets[selected_name],
        "feature_set": selected_name,
        "selection_split": "validation",
    }, models_dir / "selected_model.joblib")
    replace_marked_section(
        ROOT / "docs/modeling-decisions.md",
        "PRIMARY_SELECTION",
        "\n".join([
            f"Validation selected XGBoost with the `{selected_name}` feature set by average precision.",
            "",
            f"Selected validation precision: {float(selected['precision']):.6f}; recall: {float(selected['recall']):.6f}; F1: {float(selected['f1']):.6f}; average precision: {float(selected['average_precision']):.6f}.",
            "",
            "The held-out test was not accessed. All compared experiments are in `artifacts/metrics/experiments.csv`.",
        ]),
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
