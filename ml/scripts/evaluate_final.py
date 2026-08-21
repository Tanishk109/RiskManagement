from __future__ import annotations

import importlib.metadata
import json
import platform
from datetime import datetime, timezone

import joblib
import yaml
from common import (
    ARTIFACTS,
    ROOT,
    cost_assumptions,
    load_splits,
    replace_marked_section,
    training_config,
)
from merchantshield_ml.calibration import calibration_diagnostics
from merchantshield_ml.cost import decisions_from_scores, simulate_cost
from merchantshield_ml.evaluate import binary_metrics
from merchantshield_ml.thresholds import search_thresholds


def _package_versions() -> dict[str, str]:
    packages = ["pandas", "numpy", "scikit-learn", "xgboost", "joblib"]
    return {name: importlib.metadata.version(name) for name in packages}


def _enabled_rule_count() -> int:
    payload = yaml.safe_load((ROOT / "rules/merchant_rules.yaml").read_text(encoding="utf-8")) or {}
    return sum(1 for rule in payload.get("rules", []) if rule.get("enabled"))


def main() -> None:
    models_dir = ARTIFACTS / "models"
    metrics_dir = ARTIFACTS / "metrics"
    reports_dir = ARTIFACTS / "reports"
    selected_path = models_dir / "selected_model.joblib"
    selection_path = metrics_dir / "validation_selection.json"
    if not selected_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("Run train_primary.py before final evaluation")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selection_split") != "validation" or selection.get("test_accessed") is not False:
        raise ValueError("Validation selection artifact does not satisfy the held-out test policy")
    bundle = joblib.load(selected_path)
    features = list(bundle["feature_names"])
    splits = load_splits(features)
    pipeline = bundle["pipeline"]
    assumptions = cost_assumptions()

    validation_scores = pipeline.predict_proba(splits.validation[features])[:, 1]
    threshold_analysis = search_thresholds(
        labels=splits.validation["isFraud"].to_numpy(),
        amounts=splits.validation["TransactionAmt"].to_numpy(),
        risk_scores=validation_scores,
        assumptions=assumptions,
    )
    frozen = threshold_analysis["lowest_estimated_cost_configuration"]
    review_threshold = float(frozen["review_threshold"])
    block_threshold = float(frozen["block_threshold"])
    threshold_analysis["business_assumptions"] = assumptions.__dict__
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "threshold_analysis.json").write_text(json.dumps(threshold_analysis, indent=2) + "\n", encoding="utf-8")
    replace_marked_section(
        ROOT / "docs/modeling-decisions.md",
        "THRESHOLD_SELECTION",
        "\n".join([
            f"Validation cost search froze review threshold `{review_threshold:.4f}` and block threshold `{block_threshold:.4f}`.",
            "",
            "Selection wording: Lowest estimated cost under the currently selected merchant assumptions. The full search and assumptions are in `artifacts/metrics/threshold_analysis.json`.",
        ]),
    )

    # The held-out test is accessed only after model, feature set, and thresholds are frozen.
    test_scores = pipeline.predict_proba(splits.test[features])[:, 1]
    test_labels = splits.test["isFraud"].to_numpy(dtype=int)
    test_amounts = splits.test["TransactionAmt"].to_numpy(dtype=float)
    binary = binary_metrics(test_labels, test_scores, threshold=block_threshold)
    costs = simulate_cost(
        labels=test_labels,
        amounts=test_amounts,
        risk_scores=test_scores,
        review_threshold=review_threshold,
        block_threshold=block_threshold,
        assumptions=assumptions,
    )
    decisions = decisions_from_scores(test_scores, review_threshold, block_threshold)
    counts = {name: int((decisions == name).sum()) for name in ["APPROVE", "REVIEW", "BLOCK"]}
    total = len(decisions)
    generated_at = datetime.now(timezone.utc).isoformat()
    active_rule_count = _enabled_rule_count()
    final_metrics = {
        "evaluation_status": "complete",
        "generated_at": generated_at,
        "split": "test",
        "metric_definition": "Binary precision/recall/F1 use the frozen block threshold; review-band outcomes are represented separately in cost calculations.",
        "test_transaction_count": total,
        "fraud_count": int(test_labels.sum()),
        "precision": binary["precision"],
        "recall": binary["recall"],
        "f1": binary["f1"],
        "average_precision": binary["average_precision"],
        "roc_auc": binary["roc_auc"],
        "brier_score": binary["brier_score"],
        "true_positives": binary["true_positives"],
        "false_positives": binary["false_positives"],
        "true_negatives": binary["true_negatives"],
        "false_negatives": binary["false_negatives"],
        "approve_count": counts["APPROVE"],
        "review_count": counts["REVIEW"],
        "block_count": counts["BLOCK"],
        "decision_distribution": {
            "approve": {"count": counts["APPROVE"], "share": counts["APPROVE"] / total},
            "review": {"count": counts["REVIEW"], "share": counts["REVIEW"] / total},
            "block": {"count": counts["BLOCK"], "share": counts["BLOCK"] / total},
        },
        "confusion_matrix": {
            "true_positives": binary["true_positives"],
            "false_positives": binary["false_positives"],
            "true_negatives": binary["true_negatives"],
            "false_negatives": binary["false_negatives"],
        },
        "false_positive_estimated_cost": costs["false_positive_cost"],
        "false_negative_estimated_cost": costs["fraud_loss"],
        "review_cost": costs["review_cost"],
        "total_estimated_cost": costs["total_estimated_cost"],
        "review_threshold": review_threshold,
        "block_threshold": block_threshold,
        "business_assumptions": assumptions.__dict__,
        "active_rule_count": active_rule_count,
    }
    (metrics_dir / "final_test_metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n", encoding="utf-8")

    prediction_frame = splits.test[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]].copy()
    prediction_frame["risk_score"] = test_scores
    prediction_frame["decision"] = decisions
    prediction_frame.to_csv(metrics_dir / "final_test_predictions.csv", index=False)
    (metrics_dir / "final_test_calibration.json").write_text(json.dumps(calibration_diagnostics(test_labels, test_scores), indent=2) + "\n", encoding="utf-8")

    model_version = f"merchantshield-xgboost-{generated_at[:10]}"
    metadata = {
        "model_name": "XGBoost",
        "model_version": model_version,
        "trained_at": selection["selected_at"],
        "feature_set": bundle["feature_set"],
        "feature_names": features,
        "training_split": "first 70% by TransactionDT",
        "selection_split": "next 15% by TransactionDT",
        "evaluation_split": "final 15% by TransactionDT",
        "thresholds": {"review": review_threshold, "block": block_threshold},
        "threshold_config_id": f"validation-cost-{generated_at[:10]}",
        "calibration": "No post-hoc calibrator retained unless separately documented by a validation experiment.",
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, models_dir / "model.joblib")
    (models_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "generated_at": generated_at,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "random_seed": training_config()["random_seed"],
        "model_parameters": training_config()["primary_parameters"],
        "feature_names": features,
        "split_boundaries": splits.boundaries.to_dict(),
        "test_policy": "Held-out test used once after validation freeze.",
    }
    (models_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    reports_dir.mkdir(parents=True, exist_ok=True)
    report = [
        "# Final Held-Out Evaluation",
        "",
        "Generated from the final chronological 15% of the local labeled IEEE-CIS train data after model and threshold freeze.",
        "",
        "## Model-derived results",
        "",
        f"- Test transactions: {total:,}",
        f"- Fraud transactions: {int(test_labels.sum()):,}",
        f"- Precision at block threshold: {float(binary['precision']):.6f}",
        f"- Recall at block threshold: {float(binary['recall']):.6f}",
        f"- F1: {float(binary['f1']):.6f}",
        f"- Average precision / PR-AUC: {float(binary['average_precision']):.6f}",
        f"- ROC-AUC: {float(binary['roc_auc']):.6f}",
        f"- TP / FP / TN / FN: {binary['true_positives']} / {binary['false_positives']} / {binary['true_negatives']} / {binary['false_negatives']}",
        f"- APPROVE / REVIEW / BLOCK: {counts['APPROVE']} / {counts['REVIEW']} / {counts['BLOCK']}",
        "",
        "## Frozen thresholds",
        "",
        f"- Review threshold: {review_threshold:.4f}",
        f"- Block threshold: {block_threshold:.4f}",
        "- Selection wording: Lowest estimated cost under the currently selected merchant assumptions.",
        "",
        "## Cost estimates",
        "",
        f"- Fraud loss: {float(costs['fraud_loss']):.2f} {assumptions.currency}",
        f"- False-positive cost: {float(costs['false_positive_cost']):.2f} {assumptions.currency}",
        f"- Review cost: {float(costs['review_cost']):.2f} {assumptions.currency}",
        f"- Total estimated cost: {float(costs['total_estimated_cost']):.2f} {assumptions.currency}",
        "",
        "## Merchant assumptions",
        "",
        "These values are configurable scenario inputs, not industry facts:",
        "",
        *[f"- {key}: {value}" for key, value in assumptions.__dict__.items()],
    ]
    (reports_dir / "final_evaluation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
