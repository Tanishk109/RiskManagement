from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "merchantshield-matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import ARTIFACTS, feature_sets, load_baseline_data, training_config
from merchantshield_ml.baseline_analysis import (
    DEFAULT_EVALUATION_THRESHOLD,
    EXPERIMENT_FIELDS,
    MODEL_NAME,
    MODEL_VERSION,
    BaselineExperimentRun,
    categorical_cardinality,
    coefficient_summary,
    metric_delta,
    run_logistic_experiment,
    select_best_experiment,
    software_versions,
    validation_error_analysis,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

FEATURE_LABELS = {
    "baseline_core": "A — Core",
    "baseline_core_time": "B — Core + Time",
    "baseline_core_masked": "C — Core + Masked Numeric",
    "baseline_core_identity": "D — Core + Identity",
    "baseline_conservative_combined": "E — Conservative Combined",
    "baseline_conservative_combined_time": "F — Combined + Time",
}
SOLVER_ADJUSTMENT = {
    "initial_solver": "saga",
    "initial_max_iter": 1000,
    "issue": "All seven initial fits reached max_iter and emitted ConvergenceWarning; provisional results were discarded.",
    "action": (
        "Switched once to newton-cholesky with max_iter=100 after a combined-set diagnostic converged "
        "in 5 iterations without fit warnings. This solver is appropriate because training rows greatly "
        "outnumber the 1,696 encoded combined-set features."
    ),
}


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _write_experiments_csv(path: Path, runs: list[BaselineExperimentRun]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(EXPERIMENT_FIELDS),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(run.record for run in runs)


def _write_predictions(path: Path, validation: pd.DataFrame, runs: list[BaselineExperimentRun]) -> None:
    frames = []
    actual = validation["isFraud"].astype(int).to_numpy()
    transaction_ids = validation["TransactionID"].to_numpy()
    for run in runs:
        frames.append(
            pd.DataFrame(
                {
                    "TransactionID": transaction_ids,
                    "actual_label": actual,
                    "fraud_probability": run.fraud_probabilities,
                    "predicted_label_at_0_5": run.predicted_labels,
                    "feature_set": run.record["feature_set"],
                    "model_version": MODEL_VERSION,
                    "experiment_id": run.record["experiment_id"],
                }
            )
        )
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False, compression="zstd")


def _plot_precision_recall(path: Path, labels: np.ndarray, run: BaselineExperimentRun) -> None:
    precision, recall, _ = precision_recall_curve(labels, run.fraud_probabilities)
    plt.figure(figsize=(7.6, 5.2))
    plt.plot(recall, precision, color="#6f7bf7", linewidth=2)
    plt.scatter(
        [float(run.record["recall"])],
        [float(run.record["precision"])],
        color="#ff6b6b",
        label="Fixed threshold 0.50",
        zorder=3,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Validation precision–recall curve (AP={float(run.record['average_precision']):.4f})")
    plt.grid(alpha=0.2)
    plt.legend()
    _save_figure(path)


def _plot_confusion_matrix(path: Path, run: BaselineExperimentRun) -> None:
    matrix = np.array(
        [
            [int(run.record["tn"]), int(run.record["fp"])],
            [int(run.record["fn"]), int(run.record["tp"])],
        ]
    )
    plt.figure(figsize=(6.2, 5.2))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Validation transactions")
    plt.xticks([0, 1], ["Predicted legitimate", "Predicted fraud"])
    plt.yticks([0, 1], ["Actual legitimate", "Actual fraud"])
    for row in range(2):
        for column in range(2):
            plt.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", fontsize=13)
    plt.title("Validation confusion matrix at threshold 0.50")
    _save_figure(path)


def _plot_reliability(path: Path, labels: np.ndarray, run: BaselineExperimentRun) -> dict[str, list[float]]:
    observed, predicted = calibration_curve(labels, run.fraud_probabilities, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6.8, 5.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#7b8580", label="Perfect calibration")
    plt.plot(predicted, observed, marker="o", color="#6f7bf7", linewidth=2, label="Logistic baseline")
    plt.xlabel("Mean predicted fraud probability")
    plt.ylabel("Observed fraud fraction")
    plt.title(f"Validation reliability (Brier={float(run.record['brier_score']):.4f})")
    plt.grid(alpha=0.2)
    plt.legend()
    _save_figure(path)
    return {
        "mean_predicted_probability": [float(value) for value in predicted],
        "observed_fraud_fraction": [float(value) for value in observed],
    }


def _delta_markdown(delta: dict[str, float]) -> str:
    return ", ".join(
        [
            f"precision {delta['precision']:+.6f}",
            f"recall {delta['recall']:+.6f}",
            f"F1 {delta['f1']:+.6f}",
            f"AP {delta['average_precision']:+.6f}",
            f"ROC-AUC {delta['roc_auc']:+.6f}",
        ]
    )


def _experiment_table(records: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Experiment | Feature set | Weight | Precision | Recall | F1 | AP | ROC-AUC | FP | FN | Seconds | Converged |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record['experiment_id']} | {FEATURE_LABELS[str(record['feature_set'])]} | {record['class_weight']} | "
            f"{float(record['precision']):.6f} | {float(record['recall']):.6f} | {float(record['f1']):.6f} | "
            f"{float(record['average_precision']):.6f} | {float(record['roc_auc']):.6f} | "
            f"{int(record['fp']):,} | {int(record['fn']):,} | {float(record['training_seconds']):.3f} |"
            f" {'yes' if record['converged'] else 'no'} |"
        )
    return lines


def _write_logistic_report(path: Path, artifact: dict[str, Any]) -> None:
    best = artifact["best_experiment"]
    error = artifact["validation_error_analysis"]
    false_negatives = error["outcomes"]["false_negatives"]
    coefficients = artifact["coefficient_summary"]
    lines = [
        "# Logistic Regression Baseline",
        "",
        "## Objective",
        "",
        "Establish a transparent validation-only fraud baseline and measure the incremental value of time, masked numeric, and identity feature groups. This is not the final production model.",
        "",
        "## Training Data",
        "",
        f"- Frozen chronological TRAIN: {int(best['training_rows']):,} rows",
        "- All preprocessing and Logistic Regression parameters were fit on TRAIN only.",
        "",
        "## Validation Data",
        "",
        f"- Frozen chronological VALIDATION: {int(best['validation_rows']):,} rows",
        "- Validation class distribution was not sampled, stratified, or rebalanced.",
        "- HELD-OUT TEST was not loaded and no test predictions were generated.",
        "",
        "## Features Tested",
        "",
    ]
    for feature_set, features in artifact["feature_sets"].items():
        lines.append(f"- **{FEATURE_LABELS[feature_set]}:** {', '.join(f'`{feature}`' for feature in features)}")
    lines.extend(
        [
            "",
            "## Preprocessing",
            "",
            "- Numeric: TRAIN-fitted median imputation followed by `StandardScaler`.",
            "- Categorical: TRAIN-fitted constant `__MISSING__` imputation followed by `OneHotEncoder(handle_unknown=\"ignore\")`.",
            "- Raw `TransactionAmt` was used; no silent transformation or clipping was applied.",
            "- No automatic missingness indicators were added beyond `identity_available`.",
            (
                "- TRAIN non-null cardinalities: "
                f"`DeviceInfo`={int(artifact['training_cardinality']['DeviceInfo']['non_null_categories']):,}, "
                f"`P_emaildomain`={int(artifact['training_cardinality']['P_emaildomain']['non_null_categories']):,}, "
                f"`R_emaildomain`={int(artifact['training_cardinality']['R_emaildomain']['non_null_categories']):,}. "
                "The sparse encoded dimensionality remained reasonable, so no rare-category grouping was applied."
            ),
            "- Solver diagnostic: the initial SAGA run reached `max_iter=1000` for all seven fits and those provisional results were discarded. `newton-cholesky` was selected once, then the complete experiment set was rerun; all final convergence states and warnings are recorded below.",
            "",
            "## Class Imbalance Strategy",
            "",
            "Each of the six named feature sets was fit with `class_weight=balanced`. The best balanced feature set was then refit with `class_weight=None` for a direct comparison. No SMOTE, over-sampling, under-sampling, or validation rebalancing was used.",
            "",
            "## Experiment Results",
            "",
            "Average Precision (AP) is computed with `sklearn.metrics.average_precision_score`. Classification metrics use the fixed descriptive threshold 0.50.",
            "",
            *_experiment_table(artifact["experiments"]),
            "",
            "## Best Baseline",
            "",
            f"- Selected experiment: `{best['experiment_id']}`",
            f"- Feature set: {FEATURE_LABELS[str(best['feature_set'])]}",
            f"- Class weight: `{best['class_weight']}`",
            f"- Encoded features: {int(best['number_of_encoded_features']):,}",
            f"- Selection reason: {artifact['selection_reason']}",
            f"- At threshold 0.50 it trades recall for precision: recall {float(best['recall']):.6f}, precision {float(best['precision']):.6f}. This is not an operational threshold recommendation.",
            "",
            "## TransactionDT Ablation",
            "",
            f"- Core + Time versus Core: {_delta_markdown(artifact['ablations']['transaction_dt_core'])}.",
            f"- Combined + Time versus Combined: {_delta_markdown(artifact['ablations']['transaction_dt_combined'])}.",
            "- `TransactionDT` is relative dataset position. Any gain is treated as validation evidence with temporal-generalization risk, not automatically as leakage or proof it belongs in the final model.",
            "",
            "## Identity Ablation",
            "",
            f"- Core + Identity versus Core: {_delta_markdown(artifact['ablations']['identity_core'])}.",
            f"- Combined versus Core + Masked Numeric: {_delta_markdown(artifact['ablations']['identity_combined'])}.",
            "- Identity availability shifted materially between TRAIN and VALIDATION, so any improvement remains potentially process-dependent.",
            "",
            "## Masked-Feature Ablation",
            "",
            f"Core + C1–C5 + D1–D3 versus Core: {_delta_markdown(artifact['ablations']['masked_numeric'])}.",
            "Masked variables are reported only by source name; no business meaning is inferred.",
            "",
            "## Validation Error Analysis",
            "",
            f"At threshold 0.50 the best baseline produced {int(best['fp']):,} false positives and {int(best['fn']):,} false negatives. Detailed supported slices are in `artifacts/reports/baseline_error_analysis.md`.",
            "",
            "## High-Value False Negatives",
            "",
            f"- Count: {int(false_negatives['row_count']):,}",
            f"- Total TransactionAmt: {float(false_negatives['transaction_amount']['total']):,.3f}",
            f"- Median TransactionAmt: {float(false_negatives['transaction_amount']['median']):,.3f}",
            f"- P90 / P95: {float(false_negatives['transaction_amount']['p90']):,.3f} / {float(false_negatives['transaction_amount']['p95']):,.3f}",
            "",
            "## Calibration Diagnostic",
            "",
            f"Validation Brier score: {float(best['brier_score']):.6f}. The reliability plot is descriptive; no Platt or isotonic calibration was fitted.",
            "",
            "## Coefficient Associations",
            "",
            "Largest positive coefficients:",
            "",
        ]
    )
    for row in coefficients["largest_positive"][:10]:
        lines.append(f"- `{row['feature']}`: {float(row['coefficient']):+.6f}")
    lines.extend(["", "Largest negative coefficients:", ""])
    for row in coefficients["largest_negative"][:10]:
        lines.append(f"- `{row['feature']}`: {float(row['coefficient']):+.6f}")
    lines.extend(
        [
            "",
            "Coefficient magnitude reflects model association after scaling and one-hot encoding, not causation.",
            "",
            "## Limitations",
            "",
            "- Results describe one later validation period and are not final held-out performance.",
            "- Threshold 0.50 is a comparison convention, not an operational approve/review/block decision.",
            "- Identity signals may encode enrichment-process behavior that changes over time.",
            "- No merchant cost, rule, threshold, or money-saved claim was calculated.",
            "",
            "## Next Experiment",
            "",
            "Train one stronger gradient-boosted tree model on TRAIN and compare it against this frozen Logistic Regression validation baseline. That phase has not started.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_error_report(path: Path, artifact: dict[str, Any]) -> None:
    error = artifact["validation_error_analysis"]
    outcomes = error["outcomes"]
    lines = [
        "# Logistic Baseline Validation Error Analysis",
        "",
        "All results use the selected Logistic Regression validation baseline and the fixed 0.50 classification threshold. No rule or operational threshold was derived from these errors.",
        "",
        "## Outcome counts and amounts",
        "",
        "| Outcome | Rows | Amount total | Amount median | P90 | P95 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("false_negatives", "true_positives", "false_positives", "true_negatives"):
        row = outcomes[name]
        amount = row["transaction_amount"]
        lines.append(
            f"| {name.replace('_', ' ').title()} | {int(row['row_count']):,} | {float(amount['total']):,.3f} | "
            f"{float(amount['median']):,.3f} | {float(amount['p90']):,.3f} | "
            f"{float(amount['p95']):,.3f} | {float(amount['max']):,.3f} |"
        )

    comparisons = (
        ("False negatives", "false_negatives", "True positives", "true_positives"),
        ("False positives", "false_positives", "True negatives", "true_negatives"),
    )
    for left_label, left_key, right_label, right_key in comparisons:
        lines.extend(["", f"## {left_label} versus {right_label}", ""])
        for column in ("ProductCD", "card4", "card6", "identity_available", "DeviceType"):
            left = outcomes[left_key]["categorical_distribution"][column][:5]
            right = outcomes[right_key]["categorical_distribution"][column][:5]
            lines.extend(
                [
                    f"### {column}",
                    "",
                    f"- {left_label}: " + ", ".join(f"{row['category']}={float(row['share_percent']):.2f}%" for row in left),
                    f"- {right_label}: " + ", ".join(f"{row['category']}={float(row['share_percent']):.2f}%" for row in right),
                    "",
                ]
            )

    lines.extend(
        [
            "## High-value false negatives",
            "",
            "These are actual validation errors retained for inspection only; no rules were created.",
            "",
            "| TransactionID | Amount | Probability | ProductCD | card4 | card6 | Identity | DeviceType |",
            "| ---: | ---: | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for row in error["high_value_false_negative_examples"]:
        lines.append(
            f"| {int(row['TransactionID'])} | {float(row['TransactionAmt']):,.3f} | "
            f"{float(row['fraud_probability']):.6f} | {row['ProductCD']} | {row['card4']} | "
            f"{row['card6']} | {row['identity_available']} | {row['DeviceType']} |"
        )

    lines.extend(
        [
            "",
            "## Weakest meaningful recall slices",
            "",
            f"Only slices with at least {int(error['minimum_slice_fraud_support'])} actual fraud rows are included.",
            f"Amount buckets are {error['amount_bucket_definition']}.",
            "",
            "| Slice | Category | Fraud support | TP | FN | Recall |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in error["weakest_meaningful_slices"]:
        lines.append(
            f"| `{row['slice']}` | {row['category']} | {int(row['fraud_support']):,} | "
            f"{int(row['true_positives']):,} | {int(row['false_negatives']):,} | {float(row['recall']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = training_config()
    available_sets = feature_sets()
    experiment_feature_sets = [str(name) for name in config["baseline_feature_sets"]]
    if set(experiment_feature_sets) != set(FEATURE_LABELS):
        raise ValueError("Baseline configuration must contain exactly feature sets A–F")
    union_features = list(
        dict.fromkeys(feature for feature_set in experiment_feature_sets for feature in available_sets[feature_set])
    )
    partitions = load_baseline_data(union_features)
    train = partitions.train
    validation = partitions.validation
    random_state = int(config["random_seed"])
    solver = str(config["baseline_solver"])
    max_iter = int(config["baseline_max_iter"])
    threshold = float(config["baseline_default_threshold"])
    if threshold != DEFAULT_EVALUATION_THRESHOLD:
        raise ValueError("Baseline comparison threshold must remain fixed at 0.50")

    runs: list[BaselineExperimentRun] = []
    for index, feature_set in enumerate(experiment_feature_sets, start=1):
        print(f"Running Logistic Regression experiment {index}/6: {FEATURE_LABELS[feature_set]} (balanced)", flush=True)
        runs.append(
            run_logistic_experiment(
                train,
                validation,
                experiment_id=f"lr-{index:02d}-{feature_set.removeprefix('baseline_')}-balanced",
                feature_set=feature_set,
                features=available_sets[feature_set],
                class_weight="balanced",
                random_state=random_state,
                solver=solver,
                max_iter=max_iter,
                threshold=threshold,
            )
        )

    best_balanced = select_best_experiment(runs)
    best_balanced_feature_set = str(best_balanced.record["feature_set"])
    print(f"Running class-weight comparison: {FEATURE_LABELS[best_balanced_feature_set]} (unweighted)", flush=True)
    runs.append(
        run_logistic_experiment(
            train,
            validation,
            experiment_id=f"lr-07-{best_balanced_feature_set.removeprefix('baseline_')}-none",
            feature_set=best_balanced_feature_set,
            features=available_sets[best_balanced_feature_set],
            class_weight=None,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            threshold=threshold,
        )
    )
    best = select_best_experiment(runs)

    by_feature = {
        str(run.record["feature_set"]): run.record
        for run in runs
        if str(run.record["class_weight"]) == "balanced"
    }
    unweighted = runs[-1].record
    ablations = {
        "transaction_dt_core": metric_delta(by_feature["baseline_core_time"], by_feature["baseline_core"]),
        "transaction_dt_combined": metric_delta(
            by_feature["baseline_conservative_combined_time"],
            by_feature["baseline_conservative_combined"],
        ),
        "identity_core": metric_delta(by_feature["baseline_core_identity"], by_feature["baseline_core"]),
        "identity_combined": metric_delta(
            by_feature["baseline_conservative_combined"], by_feature["baseline_core_masked"]
        ),
        "masked_numeric": metric_delta(by_feature["baseline_core_masked"], by_feature["baseline_core"]),
        "class_weight_unweighted_minus_balanced": metric_delta(unweighted, best_balanced.record),
    }
    error_analysis = validation_error_analysis(validation, best)
    coefficients = coefficient_summary(best)
    training_cardinality = categorical_cardinality(
        train,
        ["DeviceInfo", "P_emaildomain", "R_emaildomain"],
    )

    metrics_dir = ARTIFACTS / "metrics"
    models_dir = ARTIFACTS / "models"
    reports_dir = ARTIFACTS / "reports"
    figures_dir = ARTIFACTS / "figures"
    predictions_dir = ARTIFACTS / "predictions"
    for directory in (metrics_dir, models_dir, reports_dir, figures_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    labels = validation["isFraud"].astype(int).to_numpy()
    _plot_precision_recall(figures_dir / "baseline_precision_recall_curve.png", labels, best)
    _plot_confusion_matrix(figures_dir / "baseline_confusion_matrix.png", best)
    reliability = _plot_reliability(figures_dir / "baseline_reliability.png", labels, best)
    _write_experiments_csv(metrics_dir / "experiments.csv", runs)
    _write_predictions(predictions_dir / "baseline_validation.parquet", validation, runs)

    trained_at = datetime.now(timezone.utc).isoformat()
    selection_reason = (
        "Highest validation Average Precision; F1 and recall are deterministic tie-breakers. "
        "Accuracy was not used for selection, and threshold 0.50 was not optimized."
    )
    artifact = {
        "artifact_type": "validation_baseline_experiment",
        "status": "validation_baseline",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "training_split": "train",
        "validation_split": "validation",
        "held_out_test_accessed": False,
        "held_out_test_predictions_generated": False,
        "default_evaluation_threshold": threshold,
        "threshold_optimized": False,
        "average_precision_definition": "sklearn.metrics.average_precision_score",
        "feature_sets": {name: available_sets[name] for name in experiment_feature_sets},
        "training_cardinality": training_cardinality,
        "experiments": [run.record for run in runs],
        "best_experiment": best.record,
        "best_balanced_experiment": best_balanced.record,
        "selection_reason": selection_reason,
        "ablations": ablations,
        "validation_error_analysis": error_analysis,
        "coefficient_summary": coefficients,
        "reliability": reliability,
        "software_versions": software_versions(),
        "solver_adjustment": SOLVER_ADJUSTMENT,
        "merchant_facing_metrics_updated": False,
    }
    (metrics_dir / "baseline_validation.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    model_metadata = {
        "status": "validation_baseline",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "feature_set": best.record["feature_set"],
        "feature_names": available_sets[str(best.record["feature_set"])],
        "encoded_feature_count": best.record["number_of_encoded_features"],
        "class_weight": best.record["class_weight"],
        "solver": best.record["solver"],
        "max_iter": best.record["max_iter"],
        "converged": best.record["converged"],
        "training_split": "train",
        "training_rows": best.record["training_rows"],
        "validation_split": "validation",
        "validation_rows": best.record["validation_rows"],
        "held_out_test_accessed": False,
        "default_evaluation_threshold": threshold,
        "validation_metrics": {
            key: best.record[key]
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
                "brier_score",
            )
        },
        "software_versions": artifact["software_versions"],
        "solver_adjustment": SOLVER_ADJUSTMENT,
    }
    (models_dir / "logistic_baseline_metadata.json").write_text(
        json.dumps(model_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    joblib.dump(
        {
            "pipeline": best.pipeline,
            "metadata": model_metadata,
        },
        models_dir / "logistic_baseline.joblib",
    )
    _write_logistic_report(reports_dir / "logistic_baseline.md", artifact)
    _write_error_report(reports_dir / "baseline_error_analysis.md", artifact)
    print(json.dumps({"best_experiment": best.record, "ablations": ablations}, indent=2))


if __name__ == "__main__":
    main()
