from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "merchantshield-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import ARTIFACTS, feature_sets, load_baseline_data, training_config
from merchantshield_ml.catboost_candidate import (
    CATBOOST_CATEGORICAL_FEATURES,
    CATBOOST_EXPERIMENT_FIELDS,
    CATBOOST_MODEL_NAME,
    CATBOOST_MODEL_VERSION,
    DEFAULT_EVALUATION_THRESHOLD,
    IDENTITY_FEATURES,
    CatBoostExperimentRun,
    choose_identity_candidate,
    false_negative_amount_summary,
    logistic_failure_slice_comparison,
    prediction_artifact,
    run_catboost_experiment,
    select_best_catboost,
    software_versions,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

FROZEN_LOGISTIC_EXPERIMENT_ID = "lr-07-conservative_combined-none"
FROZEN_LOGISTIC_AP = 0.23143384660363925
FROZEN_LOGISTIC_ROC_AUC = 0.7934804128969273
FROZEN_LOGISTIC_FN = 2_825
FROZEN_LOGISTIC_FP = 83
FROZEN_LOGISTIC_FN_AMOUNT_TOTAL = 481_947.732
MAIN_EXPERIMENTS = (
    ("cb-01-combined-none", None),
    ("cb-02-combined-balanced", "Balanced"),
    ("cb-03-combined-sqrt-balanced", "SqrtBalanced"),
)
PREDICTION_FILENAMES = {
    "cb-01-combined-none": "catboost_cb01_validation.parquet",
    "cb-02-combined-balanced": "catboost_cb02_validation.parquet",
    "cb-03-combined-sqrt-balanced": "catboost_cb03_validation.parquet",
}


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _load_frozen_logistic() -> tuple[dict[str, Any], pd.DataFrame]:
    metrics_path = ARTIFACTS / "metrics/baseline_validation.json"
    predictions_path = ARTIFACTS / "predictions/baseline_validation.parquet"
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("Frozen Logistic Regression validation artifacts are required before CatBoost")
    artifact = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = artifact["best_experiment"]
    if selected["experiment_id"] != FROZEN_LOGISTIC_EXPERIMENT_ID:
        raise ValueError("Frozen Logistic Regression experiment identifier changed")
    if not np.isclose(float(selected["average_precision"]), FROZEN_LOGISTIC_AP, rtol=0, atol=1e-12):
        raise ValueError("Frozen Logistic Regression Average Precision changed")
    if not np.isclose(float(selected["roc_auc"]), FROZEN_LOGISTIC_ROC_AUC, rtol=0, atol=1e-12):
        raise ValueError("Frozen Logistic Regression ROC-AUC changed")
    predictions = pd.read_parquet(predictions_path)
    predictions = predictions[predictions["experiment_id"] == FROZEN_LOGISTIC_EXPERIMENT_ID].copy()
    if len(predictions) != int(selected["validation_rows"]):
        raise ValueError("Frozen Logistic Regression prediction row count changed")
    return selected, predictions


def _metric_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float | int]:
    keys = ("precision", "recall", "f1", "average_precision", "roc_auc")
    delta: dict[str, float | int] = {
        key: float(candidate[key]) - float(reference[key]) for key in keys
    }
    delta.update(
        {
            "fp": int(candidate["fp"]) - int(reference["fp"]),
            "fn": int(candidate["fn"]) - int(reference["fn"]),
        }
    )
    return delta


def _baseline_comparison(run: CatBoostExperimentRun, logistic: dict[str, Any]) -> dict[str, float | int]:
    ap_delta = float(run.record["average_precision"]) - float(logistic["average_precision"])
    roc_delta = float(run.record["roc_auc"]) - float(logistic["roc_auc"])
    return {
        **_metric_delta(run.record, logistic),
        "average_precision_relative_percent": 100 * ap_delta / float(logistic["average_precision"]),
        "roc_auc_relative_percent": 100 * roc_delta / float(logistic["roc_auc"]),
    }


def _append_experiments(path: Path, runs: list[CatBoostExperimentRun]) -> None:
    existing_rows: list[dict[str, Any]] = []
    existing_fields: list[str] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = [row for row in reader if not str(row.get("experiment_id", "")).startswith("cb-")]
    fields = list(dict.fromkeys([*existing_fields, *CATBOOST_EXPERIMENT_FIELDS]))
    rows: list[dict[str, Any]] = [*existing_rows]
    for run in runs:
        row = dict(run.record)
        row["parameters"] = json.dumps(row["parameters"], sort_keys=True)
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _high_value_false_negatives(
    validation: pd.DataFrame,
    run: CatBoostExperimentRun,
    limit: int = 10,
) -> list[dict[str, Any]]:
    working = validation.copy()
    working["fraud_probability"] = run.fraud_probabilities
    mask = (working["isFraud"] == 1) & (run.predicted_labels == 0)
    columns = [
        "TransactionID",
        "TransactionAmt",
        "fraud_probability",
        "ProductCD",
        "card4",
        "card6",
        "identity_available",
        "DeviceType",
    ]
    frame = working.loc[mask].nlargest(limit, "TransactionAmt")[columns]
    return [
        {
            key: None if pd.isna(value) else value.item() if hasattr(value, "item") else value
            for key, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def _feature_importance(run: CatBoostExperimentRun) -> list[dict[str, float | str]]:
    values = np.asarray(run.model.get_feature_importance(), dtype=float)
    if len(values) != len(run.features):
        raise AssertionError("CatBoost feature importance length differs from raw feature count")
    rows = [
        {"feature": feature, "importance": float(value)}
        for feature, value in zip(run.features, values, strict=True)
    ]
    return sorted(rows, key=lambda row: float(row["importance"]), reverse=True)


def _plot_precision_recall_comparison(
    path: Path,
    validation: pd.DataFrame,
    selected: CatBoostExperimentRun,
    logistic_predictions: pd.DataFrame,
) -> None:
    selected_frame = pd.DataFrame(
        {
            "TransactionID": validation["TransactionID"].to_numpy(),
            "actual_label": validation["isFraud"].astype(int).to_numpy(),
            "catboost_probability": selected.fraud_probabilities,
        }
    )
    comparison = selected_frame.merge(
        logistic_predictions[["TransactionID", "actual_label", "fraud_probability"]],
        on="TransactionID",
        how="inner",
        validate="one_to_one",
        suffixes=("_catboost", "_logistic"),
    )
    if len(comparison) != len(validation):
        raise ValueError("Logistic and CatBoost validation prediction IDs do not align")
    if not (comparison["actual_label_catboost"] == comparison["actual_label_logistic"]).all():
        raise ValueError("Logistic and CatBoost validation labels do not align")
    labels = comparison["actual_label_catboost"].to_numpy(dtype=int)
    logistic_precision, logistic_recall, _ = precision_recall_curve(labels, comparison["fraud_probability"])
    catboost_precision, catboost_recall, _ = precision_recall_curve(labels, comparison["catboost_probability"])
    plt.figure(figsize=(7.8, 5.4))
    plt.plot(
        logistic_recall,
        logistic_precision,
        color="#8b96a0",
        linewidth=2,
        label=f"Logistic Regression (AP={FROZEN_LOGISTIC_AP:.4f})",
    )
    plt.plot(
        catboost_recall,
        catboost_precision,
        color="#6f7bf7",
        linewidth=2,
        label=f"CatBoost (AP={float(selected.record['average_precision']):.4f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Validation precision–recall: CatBoost vs Logistic Regression")
    plt.grid(alpha=0.2)
    plt.legend()
    _save_figure(path)


def _plot_reliability(
    path: Path,
    validation: pd.DataFrame,
    selected: CatBoostExperimentRun,
) -> dict[str, list[float]]:
    observed, predicted = calibration_curve(
        validation["isFraud"].astype(int).to_numpy(),
        selected.fraud_probabilities,
        n_bins=10,
        strategy="quantile",
    )
    plt.figure(figsize=(6.8, 5.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#7b8580", label="Perfect calibration")
    plt.plot(predicted, observed, marker="o", color="#6f7bf7", linewidth=2, label="CatBoost candidate")
    plt.xlabel("Mean predicted fraud probability")
    plt.ylabel("Observed fraud fraction")
    plt.title(f"CatBoost validation reliability (Brier={float(selected.record['brier_score']):.4f})")
    plt.grid(alpha=0.2)
    plt.legend()
    _save_figure(path)
    return {
        "mean_predicted_probability": [float(value) for value in predicted],
        "observed_fraud_fraction": [float(value) for value in observed],
    }


def _experiment_table(records: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Experiment | Weight | AP | ROC-AUC | Precision | Recall | F1 | FP | FN | Seconds | Best iteration | Trees |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        lines.append(
            f"| {row['experiment_id']} | {row['class_weight']} | {float(row['average_precision']):.6f} | "
            f"{float(row['roc_auc']):.6f} | {float(row['precision']):.6f} | {float(row['recall']):.6f} | "
            f"{float(row['f1']):.6f} | {int(row['fp']):,} | {int(row['fn']):,} | "
            f"{float(row['training_seconds']):.3f} | {int(row['best_iteration']):,} | "
            f"{int(row['actual_tree_count']):,} |"
        )
    return lines


def _write_report(path: Path, artifact: dict[str, Any]) -> None:
    selected = artifact["selected_candidate"]
    comparison = artifact["selected_vs_logistic"]
    identity = artifact["identity_ablation"]
    fn = artifact["selected_false_negative_amounts"]
    lines = [
        "# CatBoost Validation Candidate",
        "",
        "## Objective",
        "",
        "Compare one nonlinear, native-categorical model family against the frozen Logistic Regression baseline using TRAIN for fitting and VALIDATION for early stopping, selection, and evaluation.",
        "",
        "## Why CatBoost",
        "",
        "CatBoost handles the mixed numerical/categorical conservative feature set natively. Categorical values use one consistent `__MISSING__` token; numerical NaN values are preserved. No scaling, one-hot encoding, external target encoding, or V features are used.",
        "",
        "## Frozen Logistic Regression Baseline",
        "",
        f"`{FROZEN_LOGISTIC_EXPERIMENT_ID}`: AP {FROZEN_LOGISTIC_AP:.6f}, ROC-AUC {FROZEN_LOGISTIC_ROC_AUC:.6f}, FP {FROZEN_LOGISTIC_FP:,}, FN {FROZEN_LOGISTIC_FN:,} at threshold 0.50.",
        "",
        "## Features",
        "",
        f"Selected raw features: {', '.join(f'`{feature}`' for feature in selected['features'])}.",
        "",
        "Raw `TransactionDT` and all V features were excluded. `TransactionID` and `isFraud` were forbidden as predictors.",
        "",
        "## Experiments",
        "",
        "All three main experiments share one parameter configuration and use CatBoost `PRAUC:type=Classic` for early stopping. External metrics come from sklearn; threshold metrics use the descriptive 0.50 cutoff.",
        "",
        *_experiment_table(artifact["experiments"]),
        "",
        "## Class Weight Comparison",
        "",
    ]
    for row in artifact["main_experiments"]:
        delta = artifact["main_vs_logistic"][row["experiment_id"]]
        lines.append(
            f"- `{row['class_weight']}`: AP {float(row['average_precision']):.6f} "
            f"({float(delta['average_precision']):+.6f}, {float(delta['average_precision_relative_percent']):+.2f}% vs LR); "
            f"FP {int(row['fp']):,}, FN {int(row['fn']):,}."
        )
    lines.extend(
        [
            "",
            "Weighting is interpreted through both ranking AP and threshold behavior; higher recall alone is not treated as sufficient.",
            "",
            "## Best Candidate",
            "",
            f"Selected `{selected['experiment_id']}` with `{selected['class_weight']}` weighting and identity features included={selected['identity_features_included']}. Selection used AP first, then the predeclared identity-stability tolerance.",
            "",
            "## Improvement Over Logistic Regression",
            "",
            f"AP changed by {float(comparison['average_precision']):+.6f} ({float(comparison['average_precision_relative_percent']):+.2f}%). ROC-AUC changed by {float(comparison['roc_auc']):+.6f} ({float(comparison['roc_auc_relative_percent']):+.2f}%). At threshold 0.50, precision changed by {float(comparison['precision']):+.6f}, recall by {float(comparison['recall']):+.6f}, F1 by {float(comparison['f1']):+.6f}, FP by {int(comparison['fp']):+,}, and FN by {int(comparison['fn']):+,}.",
            "",
            "Threshold 0.50 is descriptive and is not a merchant approve/review/block recommendation.",
            "",
            "## Failure-Slice Comparison",
            "",
            "The slice definitions and supports were frozen from the Logistic Regression analysis before CatBoost was run.",
            "",
            "| Slice | Fraud support | LR recall | CatBoost recall | Absolute improvement |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in artifact["failure_slice_comparison"]:
        lines.append(
            f"| {row['slice']} | {int(row['fraud_support']):,} | {float(row['logistic_recall']):.6f} | "
            f"{float(row['catboost_recall']):.6f} | {float(row['absolute_improvement']):+.6f} |"
        )
    lines.extend(
        [
            "",
            "## High-Value False Negatives",
            "",
            f"Count {int(fn['count']):,}; amount total {float(fn['total']):,.3f}; median {float(fn['median']):,.3f}; P90 {float(fn['p90']):,.3f}; P95 {float(fn['p95']):,.3f}; maximum {float(fn['max']):,.3f}.",
            f"Compared with Logistic Regression, FN count changed by {int(fn['count']) - FROZEN_LOGISTIC_FN:+,} and FN amount total by {float(fn['total']) - FROZEN_LOGISTIC_FN_AMOUNT_TOTAL:+,.3f}.",
            "",
            "## Identity Ablation",
            "",
            f"With identity AP: {float(identity['with_identity']['average_precision']):.6f}; without identity AP: {float(identity['without_identity']['average_precision']):.6f}. AP loss without identity: {float(identity['decision']['ap_loss_without_identity']):+.6f}; ROC-AUC loss: {float(identity['decision']['roc_auc_loss_without_identity']):+.6f}.",
            f"Decision: {identity['decision']['reason']}",
            "",
            "Identity coverage changes materially over time. The selected model excludes identity; any later reintroduction would remain provisional and process-dependent.",
            "",
            "## Feature Importance",
            "",
        ]
    )
    for row in artifact["feature_importance"][:10]:
        lines.append(f"- `{row['feature']}`: {float(row['importance']):.6f}")
    lines.extend(
        [
            "",
            "Native feature importance is associative, not causal. No semantic meaning is inferred for masked C* or D* fields, and SHAP was not run.",
            "",
            "## Calibration Diagnostic",
            "",
            f"Validation Brier score: {float(selected['brier_score']):.6f}. The reliability curve is descriptive; no calibration model was fitted.",
            "",
            "## Limitations",
            "",
            "- All results come from one chronological validation period, not the sealed held-out test.",
            "- Early stopping and candidate selection use VALIDATION, so these are development metrics.",
            "- Identity availability may encode source-system enrichment behavior.",
            "- The unweighted full and identity-free models selected best iterations 998 and 989, close to the 1,000-tree ceiling; no iteration extension was tested in this controlled phase.",
            "- No merchant thresholds, rules, costs, savings, or final dashboard metrics were produced.",
            "",
            "## Recommendation for Next Phase",
            "",
            artifact["recommendation"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = training_config()
    available_sets = feature_sets()
    feature_set = str(config["catboost_feature_set"])
    if feature_set != "baseline_conservative_combined":
        raise ValueError("CatBoost primary experiment must use the frozen conservative combined feature set")
    features = list(available_sets[feature_set])
    if "TransactionDT" in features or any(feature.startswith("V") for feature in features):
        raise ValueError("CatBoost primary features must exclude TransactionDT and all V fields")
    categorical_features = [feature for feature in CATBOOST_CATEGORICAL_FEATURES if feature in features]
    partitions = load_baseline_data(features)
    train = partitions.train
    validation = partitions.validation
    parameters = dict(config["catboost_parameters"])
    random_seed = int(config["random_seed"])
    strategies = [str(value) for value in config["catboost_class_weight_strategies"]]
    if strategies != ["none", "Balanced", "SqrtBalanced"]:
        raise ValueError("CatBoost class-weight strategies must remain none, Balanced, SqrtBalanced")
    logistic, logistic_predictions = _load_frozen_logistic()

    main_runs: list[CatBoostExperimentRun] = []
    for index, (experiment_id, weight) in enumerate(MAIN_EXPERIMENTS, start=1):
        print(f"Running CatBoost experiment {index}/3: {weight or 'none'}", flush=True)
        main_runs.append(
            run_catboost_experiment(
                train,
                validation,
                experiment_id=experiment_id,
                feature_set=feature_set,
                features=features,
                categorical_features=categorical_features,
                auto_class_weights=weight,
                parameters=parameters,
                random_seed=random_seed,
            )
        )
    best_main = select_best_catboost(main_runs)

    identity_free_features = [feature for feature in features if feature not in IDENTITY_FEATURES]
    identity_free_categorical = [
        feature for feature in categorical_features if feature not in IDENTITY_FEATURES
    ]
    selected_weight_name = str(best_main.record["class_weight"])
    selected_weight = None if selected_weight_name == "none" else selected_weight_name
    print(f"Running identity ablation with class weighting={selected_weight_name}", flush=True)
    identity_ablation = run_catboost_experiment(
        train,
        validation,
        experiment_id=f"cb-04-without-identity-{selected_weight_name.lower()}",
        feature_set="baseline_conservative_combined_without_identity",
        features=identity_free_features,
        categorical_features=identity_free_categorical,
        auto_class_weights=selected_weight,
        parameters=parameters,
        random_seed=random_seed,
    )
    selected, identity_decision = choose_identity_candidate(
        best_main,
        identity_ablation,
        max_ap_loss=float(config["catboost_identity_ablation_max_ap_loss"]),
        max_roc_auc_loss=float(config["catboost_identity_ablation_max_roc_auc_loss"]),
    )
    all_runs = [*main_runs, identity_ablation]

    metrics_dir = ARTIFACTS / "metrics"
    models_dir = ARTIFACTS / "models"
    reports_dir = ARTIFACTS / "reports"
    figures_dir = ARTIFACTS / "figures"
    predictions_dir = ARTIFACTS / "predictions"
    for directory in (metrics_dir, models_dir, reports_dir, figures_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for run in all_runs:
        filename = PREDICTION_FILENAMES.get(
            str(run.record["experiment_id"]),
            "catboost_identity_ablation_validation.parquet",
        )
        prediction_artifact(validation, run).to_parquet(
            predictions_dir / filename,
            index=False,
            compression="zstd",
        )
    _append_experiments(metrics_dir / "experiments.csv", all_runs)

    feature_importance = _feature_importance(selected)
    pd.DataFrame(feature_importance).to_csv(
        reports_dir / "catboost_feature_importance.csv",
        index=False,
    )
    failure_slices = logistic_failure_slice_comparison(validation, selected.predicted_labels)
    selected_fn = false_negative_amount_summary(validation, selected.predicted_labels)
    high_value_fn = _high_value_false_negatives(validation, selected)
    fn_by_experiment = {
        str(run.record["experiment_id"]): false_negative_amount_summary(validation, run.predicted_labels)
        for run in all_runs
    }
    with_identity_slices = logistic_failure_slice_comparison(validation, best_main.predicted_labels)
    without_identity_slices = logistic_failure_slice_comparison(validation, identity_ablation.predicted_labels)
    slice_ablation = [
        {
            "slice": with_row["slice"],
            "fraud_support": with_row["fraud_support"],
            "with_identity_recall": with_row["catboost_recall"],
            "without_identity_recall": without_row["catboost_recall"],
            "without_minus_with": (
                float(without_row["catboost_recall"]) - float(with_row["catboost_recall"])
            ),
        }
        for with_row, without_row in zip(with_identity_slices, without_identity_slices, strict=True)
    ]

    _plot_precision_recall_comparison(
        figures_dir / "catboost_vs_logistic_precision_recall_curve.png",
        validation,
        selected,
        logistic_predictions,
    )
    reliability = _plot_reliability(
        figures_dir / "catboost_reliability.png",
        validation,
        selected,
    )

    selected_vs_logistic = _baseline_comparison(selected, logistic)
    main_vs_logistic = {
        str(run.record["experiment_id"]): _baseline_comparison(run, logistic) for run in main_runs
    }
    identity_metrics_delta = _metric_delta(identity_ablation.record, best_main.record)
    recommendation = (
        "Freeze the selected CatBoost validation candidate and its feature decision before any later threshold, "
        "calibration, cost, or final held-out evaluation phase. Do not access the test partition yet."
    )
    artifact = {
        "artifact_type": "catboost_validation_selection",
        "status": "validation_candidate",
        "model": CATBOOST_MODEL_NAME,
        "model_version": CATBOOST_MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_split": "train",
        "validation_split": "validation",
        "held_out_test_accessed": False,
        "held_out_test_predictions_generated": False,
        "threshold_optimized": False,
        "merchant_cost_calculated": False,
        "merchant_facing_metrics_updated": False,
        "default_evaluation_threshold": DEFAULT_EVALUATION_THRESHOLD,
        "average_precision_definition": "sklearn.metrics.average_precision_score",
        "catboost_early_stopping_metric": parameters["eval_metric"],
        "parameters": parameters,
        "main_experiments": [run.record for run in main_runs],
        "experiments": [run.record for run in all_runs],
        "best_main_experiment": best_main.record,
        "selected_candidate": {**selected.record, "features": selected.features},
        "selection_policy": (
            "Highest main-experiment validation AP, then recall/precision tradeoff, F1, and stability; "
            "identity-free candidate preferred only within predeclared AP and ROC-AUC tolerances."
        ),
        "frozen_logistic_baseline": logistic,
        "main_vs_logistic": main_vs_logistic,
        "selected_vs_logistic": selected_vs_logistic,
        "false_negative_amounts_by_experiment": fn_by_experiment,
        "selected_false_negative_amounts": selected_fn,
        "high_value_false_negative_examples": high_value_fn,
        "failure_slice_comparison": failure_slices,
        "identity_ablation": {
            "with_identity": best_main.record,
            "without_identity": identity_ablation.record,
            "without_minus_with_metrics": identity_metrics_delta,
            "slice_comparison": slice_ablation,
            "decision": identity_decision,
        },
        "feature_importance": feature_importance,
        "reliability": reliability,
        "software_versions": software_versions(),
        "recommendation": recommendation,
    }
    (metrics_dir / "catboost_validation.json").write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )

    selected.model.save_model(models_dir / "catboost_candidate.cbm")
    model_metadata = {
        "status": "validation_candidate",
        "model_name": CATBOOST_MODEL_NAME,
        "model_version": CATBOOST_MODEL_VERSION,
        "training_timestamp": artifact["generated_at"],
        "feature_set": selected.record["feature_set"],
        "feature_names": selected.features,
        "categorical_feature_names": selected.categorical_features,
        "class_weight": selected.record["class_weight"],
        "catboost_parameters": parameters,
        "best_iteration": selected.record["best_iteration"],
        "actual_tree_count": selected.record["actual_tree_count"],
        "training_period": {
            "split": "train",
            "rows": len(train),
            "TransactionDT_min": int(train["TransactionDT"].min()),
            "TransactionDT_max": int(train["TransactionDT"].max()),
        },
        "validation_period": {
            "split": "validation",
            "rows": len(validation),
            "TransactionDT_min": int(validation["TransactionDT"].min()),
            "TransactionDT_max": int(validation["TransactionDT"].max()),
        },
        "validation_metrics": {
            key: selected.record[key]
            for key in (
                "precision",
                "recall",
                "f1",
                "average_precision",
                "roc_auc",
                "tp",
                "fp",
                "tn",
                "fn",
                "predicted_fraud_count",
                "predicted_fraud_rate",
                "brier_score",
            )
        },
        "identity_feature_decision": identity_decision,
        "software_versions": artifact["software_versions"],
        "random_seed": random_seed,
        "held_out_test_accessed": False,
        "default_evaluation_threshold": DEFAULT_EVALUATION_THRESHOLD,
    }
    (models_dir / "catboost_candidate_metadata.json").write_text(
        json.dumps(model_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(reports_dir / "catboost_validation.md", artifact)
    print(
        json.dumps(
            {
                "best_main": best_main.record,
                "selected_candidate": selected.record,
                "selected_vs_logistic": selected_vs_logistic,
                "identity_decision": identity_decision,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
