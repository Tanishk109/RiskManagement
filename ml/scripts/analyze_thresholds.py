from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ARTIFACTS, PROCESSED_DATA, merchant_scenario_config
from merchantshield_ml.cost import (
    ESTIMATED_COST_LABEL,
    CostAssumptions,
    decisions_from_scores,
    simulate_decisions,
)
from merchantshield_ml.evaluate import binary_metrics
from merchantshield_ml.processed import load_baseline_partition
from merchantshield_ml.thresholds import (
    capacity_selections,
    evaluate_threshold_grid,
    select_highest_metric,
    select_lowest_cost,
    select_lowest_false_positive_cost,
    threshold_candidates,
    validate_threshold_grid_artifact,
)

SELECTED_PREDICTIONS = (
    ARTIFACTS / "predictions/catboost_identity_ablation_validation.parquet"
)
BALANCED_PREDICTIONS = ARTIFACTS / "predictions/catboost_cb02_validation.parquet"
MODEL_METADATA = ARTIFACTS / "models/catboost_candidate_metadata.json"
THRESHOLD_JSON = ARTIFACTS / "metrics/threshold_analysis.json"
THRESHOLD_GRID = ARTIFACTS / "metrics/threshold_grid.parquet"
REPORT_PATH = ARTIFACTS / "reports/cost_threshold_analysis.md"
OPERATING_CONFIG = ARTIFACTS / "models/validation_operating_config.json"
FIGURES = ARTIFACTS / "figures"

EXPECTED_ROWS = 88_581
EXPECTED_FRAUD = 3_042
EXPECTED_EXPERIMENT = "cb-04-without-identity-none"
EXPECTED_METRICS = {
    "average_precision": 0.4260031781816555,
    "roc_auc": 0.8603319124559099,
    "precision": 0.7695516162669447,
    "recall": 0.24260355029585798,
    "f1": 0.36890777305673583,
    "false_positives": 221,
    "false_negatives": 2304,
    "brier_score": 0.024688924808000338,
}
OUTCOME_SUMMARY_FIELDS = (
    "review_threshold",
    "block_threshold",
    "approve_count",
    "review_count",
    "block_count",
    "approve_rate",
    "review_rate",
    "block_rate",
    "fraud_approved",
    "fraud_reviewed",
    "fraud_blocked",
    "legitimate_approved",
    "legitimate_reviewed",
    "legitimate_blocked",
    "block_precision",
    "block_recall",
    "detected_fraud_recall",
    "false_positives",
    "false_negatives",
    "total_fraud_amount",
    "fraud_amount_approved",
    "fraud_amount_reviewed",
    "fraud_amount_blocked",
    "captured_fraud_amount",
    "fraud_amount_capture_rate",
    "fraud_loss",
    "false_positive_cost",
    "manual_review_cost_total",
    "review_expected_residual_cost",
    "review_total_cost",
    "total_estimated_cost",
    "cost_output_label",
    "currency",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def summarize(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in OUTCOME_SUMMARY_FIELDS if field in row}


def load_and_validate_predictions() -> tuple[
    pd.DataFrame, pd.DataFrame, dict[str, Any]
]:
    predictions = pd.read_parquet(SELECTED_PREDICTIONS)
    required = {
        "TransactionID",
        "actual_label",
        "fraud_probability",
        "predicted_label_at_0_5",
        "experiment_id",
        "model_version",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Selected predictions are missing: {', '.join(missing)}")
    if (
        len(predictions) != EXPECTED_ROWS
        or predictions["TransactionID"].nunique() != EXPECTED_ROWS
    ):
        raise ValueError(
            "Selected prediction row count or TransactionID uniqueness changed"
        )
    if int(predictions["actual_label"].sum()) != EXPECTED_FRAUD:
        raise ValueError("Selected prediction fraud count changed")
    if set(predictions["experiment_id"].unique()) != {EXPECTED_EXPERIMENT}:
        raise ValueError("Selected prediction artifact has the wrong experiment ID")
    if not predictions["fraud_probability"].between(0, 1).all():
        raise ValueError("Selected probabilities must be between 0 and 1")

    metrics = binary_metrics(
        predictions["actual_label"].to_numpy(dtype=int),
        predictions["fraud_probability"].to_numpy(dtype=float),
        threshold=0.5,
    )
    for field, expected in EXPECTED_METRICS.items():
        actual = metrics[field]
        if isinstance(expected, int):
            if int(actual) != expected:
                raise ValueError(
                    f"Metric {field} did not reproduce: {actual} != {expected}"
                )
        elif not np.isclose(float(actual), expected, rtol=0, atol=1e-12):
            raise ValueError(
                f"Metric {field} did not reproduce: {actual} != {expected}"
            )

    validation = load_baseline_partition(
        PROCESSED_DATA,
        "validation",
        ["TransactionAmt", "ProductCD", "card4"],
    )
    if (
        len(validation) != EXPECTED_ROWS
        or validation["TransactionID"].nunique() != EXPECTED_ROWS
    ):
        raise ValueError(
            "Validation amount/slice source does not contain the expected unique rows"
        )
    joined = predictions.merge(
        validation[
            ["TransactionID", "isFraud", "TransactionAmt", "ProductCD", "card4"]
        ],
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )
    if joined["TransactionAmt"].isna().any():
        raise ValueError("TransactionAmt join coverage is incomplete")
    if not (joined["actual_label"].to_numpy() == joined["isFraud"].to_numpy()).all():
        raise ValueError("Prediction labels differ from the validation source labels")

    balanced = pd.read_parquet(BALANCED_PREDICTIONS)
    if (
        len(balanced) != EXPECTED_ROWS
        or balanced["TransactionID"].nunique() != EXPECTED_ROWS
    ):
        raise ValueError(
            "Balanced prediction artifact is not a valid validation comparison"
        )
    balanced = joined[["TransactionID", "actual_label"]].merge(
        balanced[
            ["TransactionID", "actual_label", "fraud_probability", "experiment_id"]
        ],
        on="TransactionID",
        how="left",
        validate="one_to_one",
        suffixes=("_selected", "_balanced"),
    )
    if not (
        balanced["actual_label_selected"].to_numpy()
        == balanced["actual_label_balanced"].to_numpy()
    ).all():
        raise ValueError(
            "Balanced validation labels do not align with the selected candidate"
        )
    metadata = read_json(MODEL_METADATA)
    return (
        joined,
        balanced,
        {"reproduced_metrics_at_0_5": metrics, "model_metadata": metadata},
    )


def decisions_for_binary(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.where(scores >= threshold, "BLOCK", "APPROVE")


def policy_outcomes(
    *,
    labels: np.ndarray,
    amounts: np.ndarray,
    selected_scores: np.ndarray,
    balanced_scores: np.ndarray,
    assumptions: CostAssumptions,
    three_way: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    policies = {
        "approve_all": np.full(labels.shape, "APPROVE"),
        "selected_catboost_binary_at_0_50": decisions_for_binary(selected_scores),
        "balanced_catboost_binary_at_0_50": decisions_for_binary(balanced_scores),
        "three_way_lowest_estimated_cost": decisions_from_scores(
            selected_scores,
            float(three_way["review_threshold"]),
            float(three_way["block_threshold"]),
        ),
    }
    outcomes = {
        name: summarize(
            simulate_decisions(
                labels=labels,
                amounts=amounts,
                decisions=decisions,
                assumptions=assumptions,
            )
        )
        for name, decisions in policies.items()
    }
    outcomes["three_way_lowest_estimated_cost"].update(
        {
            "review_threshold": three_way["review_threshold"],
            "block_threshold": three_way["block_threshold"],
        }
    )
    return outcomes


def frontier(rows: list[dict[str, Any]], minimum_recall: float) -> list[dict[str, Any]]:
    definitions = [
        ("lowest_estimated_cost", select_lowest_cost(rows)),
        (
            "highest_detected_fraud_recall_under_1pct_review",
            select_highest_metric(rows, "detected_fraud_recall", max_review_rate=0.01),
        ),
        (
            "highest_detected_fraud_recall_under_2pct_review",
            select_highest_metric(rows, "detected_fraud_recall", max_review_rate=0.02),
        ),
        (
            "highest_fraud_amount_capture_under_2pct_review",
            select_highest_metric(
                rows, "fraud_amount_capture_rate", max_review_rate=0.02
            ),
        ),
        (
            f"lowest_false_positive_cost_at_min_{minimum_recall:.0%}_detected_recall",
            select_lowest_false_positive_cost(
                rows, minimum_detected_recall=minimum_recall
            ),
        ),
    ]
    return [
        {"objective": objective, "configuration": summarize(configuration)}
        for objective, configuration in definitions
        if configuration is not None
    ]


def fraud_decision_summary(
    frame: pd.DataFrame, decisions: np.ndarray
) -> dict[str, Any]:
    fraud = frame.loc[frame["actual_label"] == 1].copy()
    fraud["decision"] = decisions[frame["actual_label"].to_numpy(dtype=int) == 1]
    result: dict[str, Any] = {"fraud_rows": len(fraud)}
    for decision in ("APPROVE", "REVIEW", "BLOCK"):
        selected = fraud["decision"] == decision
        result[decision.lower()] = {
            "count": int(selected.sum()),
            "rate": float(selected.mean()) if len(fraud) else 0.0,
            "transaction_amount": float(fraud.loc[selected, "TransactionAmt"].sum()),
        }
    return result


def residual_analysis(
    frame: pd.DataFrame,
    review_threshold: float,
    block_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    decisions = decisions_from_scores(
        frame["fraud_probability"].to_numpy(dtype=float),
        review_threshold,
        block_threshold,
    )
    slice_masks = {
        "ProductCD=W": frame["ProductCD"].fillna("__MISSING__").eq("W").to_numpy(),
        "TransactionAmt>=500": frame["TransactionAmt"].ge(500).to_numpy(),
        "card4=discover": frame["card4"]
        .fillna("__MISSING__")
        .eq("discover")
        .to_numpy(),
        "ProductCD=S": frame["ProductCD"].fillna("__MISSING__").eq("S").to_numpy(),
    }
    slices = {
        name: fraud_decision_summary(
            frame.loc[mask].reset_index(drop=True), decisions[mask]
        )
        for name, mask in slice_masks.items()
    }
    high_value = fraud_decision_summary(frame, decisions)
    approved_fraud = frame.loc[
        (frame["actual_label"] == 1) & (decisions == "APPROVE")
    ].copy()
    approved_fraud = approved_fraud.sort_values(
        ["TransactionAmt", "TransactionID"], ascending=[False, True]
    ).head(10)
    high_value["highest_value_approved_fraud_examples"] = [
        {
            "TransactionID": int(row.TransactionID),
            "TransactionAmt": float(row.TransactionAmt),
            "fraud_probability": float(row.fraud_probability),
            "ProductCD": None if pd.isna(row.ProductCD) else str(row.ProductCD),
            "card4": None if pd.isna(row.card4) else str(row.card4),
        }
        for row in approved_fraud.itertuples(index=False)
    ]
    return slices, high_value


def plot_results(
    main_grids: dict[str, list[dict[str, Any]]],
    provisional: dict[str, Any],
    policies: dict[str, dict[str, Any]],
    scenario_names: dict[str, str],
) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    moderate = pd.DataFrame(main_grids["moderate"])

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        moderate["review_threshold"],
        moderate["block_threshold"],
        c=moderate["total_estimated_cost"] / 1_000,
        cmap="viridis",
        s=28,
    )
    ax.scatter(
        provisional["review_threshold"],
        provisional["block_threshold"],
        color="#ef4444",
        marker="*",
        s=180,
        label="Provisional validation point",
    )
    ax.set(
        xlabel="Review threshold",
        ylabel="Block threshold",
        title="Scenario B threshold cost surface",
    )
    fig.colorbar(scatter, ax=ax, label="Estimated total cost (INR thousands)")
    ax.legend()
    fig.tight_layout()
    path = FIGURES / "threshold_cost_surface.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(ARTIFACTS.parent)))

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ("#0f766e", "#2563eb", "#9333ea")
    for (scenario_id, rows), color in zip(main_grids.items(), colors, strict=True):
        data = pd.DataFrame(rows)
        ax.scatter(
            data["review_rate"] * 100,
            data["total_estimated_cost"] / 1_000,
            s=12,
            alpha=0.45,
            color=color,
            label=scenario_names[scenario_id],
        )
    ax.set(
        xlabel="Validation transactions sent to review (%)",
        ylabel="Estimated total cost (INR thousands)",
        title="Review capacity and estimated cost",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "review_rate_vs_estimated_cost.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(ARTIFACTS.parent)))

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        moderate["fraud_amount_capture_rate"] * 100,
        moderate["false_positive_cost"] / 1_000,
        c=moderate["review_rate"] * 100,
        cmap="plasma",
        s=25,
    )
    ax.set(
        xlabel="Expected fraud amount capture (%)",
        ylabel="Estimated false-positive cost (INR thousands)",
        title="Scenario B fraud-value capture vs false-positive cost",
    )
    fig.colorbar(scatter, ax=ax, label="Review rate (%)")
    fig.tight_layout()
    path = FIGURES / "fraud_amount_capture_vs_false_positive_cost.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(ARTIFACTS.parent)))

    binary = policies["selected_catboost_binary_at_0_50"]
    selected = policies["provisional_three_way_under_2pct_review"]
    components = ("fraud_loss", "false_positive_cost", "manual_review_cost_total")
    labels = ("Fraud loss", "False-positive cost", "Manual-review cost")
    positions = np.arange(2)
    bottoms = np.zeros(2)
    fig, ax = plt.subplots(figsize=(8, 6))
    for field, label, color in zip(
        components, labels, ("#dc2626", "#f59e0b", "#2563eb"), strict=True
    ):
        values = np.array([binary[field], selected[field]], dtype=float) / 1_000
        ax.bar(positions, values, bottom=bottoms, label=label, color=color)
        bottoms += values
    ax.set_xticks(positions, ["Binary block at 0.50", "Provisional three-way"])
    ax.set_ylabel("Estimated cost (INR thousands)")
    ax.set_title("Scenario B: provisional point vs fixed 0.50 binary policy")
    ax.legend()
    fig.tight_layout()
    path = FIGURES / "provisional_vs_binary_0_50_cost.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(ARTIFACTS.parent)))
    return paths


def money(value: Any) -> str:
    return f"INR {float(value):,.2f}"


def pct(value: Any) -> str:
    return f"{float(value):.3%}"


def outcome_table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "| Configuration | Review | Block | Review rate | Block rate | Detected recall | Amount capture | FP | FN | Fraud loss | FP cost | Manual review cost | Total cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in rows:
        review = (
            "—" if "review_threshold" not in row else f"{row['review_threshold']:.3f}"
        )
        block = "—" if "block_threshold" not in row else f"{row['block_threshold']:.3f}"
        lines.append(
            f"| {name} | {review} | {block} | {pct(row['review_rate'])} | {pct(row['block_rate'])} | "
            f"{pct(row['detected_fraud_recall'])} | {pct(row['fraud_amount_capture_rate'])} | "
            f"{row['false_positives']:,} | {row['false_negatives']:,} | {money(row['fraud_loss'])} | "
            f"{money(row['false_positive_cost'])} | {money(row['manual_review_cost_total'])} | "
            f"{money(row['total_estimated_cost'])} |"
        )
    return "\n".join(lines)


def write_report(artifact: dict[str, Any]) -> None:
    policy_labels = {
        "approve_all": "Approve all",
        "selected_catboost_binary_at_0_50": "Selected CatBoost: binary block at 0.50",
        "balanced_catboost_binary_at_0_50": "Balanced CatBoost artifact: binary block at 0.50",
        "three_way_lowest_estimated_cost": "Three-way lowest estimated cost",
        "provisional_three_way_under_2pct_review": "Provisional three-way under 2% review",
    }
    lines = [
        "# Validation Cost and Threshold Analysis",
        "",
        f"**{ESTIMATED_COST_LABEL}**",
        "",
        "This is validation-only development evidence, not held-out or production performance. All three scenarios are **ILLUSTRATIVE MERCHANT ASSUMPTIONS**, not industry facts.",
        "",
        "## Data and model integrity",
        "",
        f"The saved `{EXPECTED_EXPERIMENT}` validation predictions contain {artifact['data_validation']['rows']:,} rows, including {artifact['data_validation']['fraud_count']:,} fraud and {artifact['data_validation']['legitimate_count']:,} legitimate transactions. The fixed-threshold metrics reproduce the frozen candidate artifact. CatBoost was not retrained and the held-out test was not accessed.",
        "",
        "## Merchant scenarios",
        "",
        "| Scenario | Fraud loss fraction | Fraud fixed cost | Margin rate | FP fixed cost | Review cost | Review catch | Legit approval |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in artifact["scenarios"].values():
        assumptions = scenario["assumptions"]
        lines.append(
            f"| {scenario['name']} | {pct(assumptions['fraud_loss_fraction'])} | "
            f"{money(assumptions['chargeback_fixed_cost'])} | {pct(assumptions['legitimate_margin_rate'])} | "
            f"{money(assumptions['false_positive_fixed_cost'])} | {money(assumptions['manual_review_cost'])} | "
            f"{pct(assumptions['review_fraud_catch_rate'])} | "
            f"{pct(assumptions['review_legitimate_approval_rate'])} |"
        )
    lines.extend(
        [
            "",
            "A fraud approval incurs the configured amount fraction plus the explicitly configured fixed fraud cost. A legitimate block incurs estimated lost contribution plus the fixed false-positive cost. A review incurs its manual cost on every reviewed row plus expected residual fraud/false-positive cost. These are simplifying assumptions and are configurable.",
            "",
            "## Lowest estimated cost by scenario",
            "",
            outcome_table(
                [
                    (scenario["name"], scenario["lowest_estimated_cost"])
                    for scenario in artifact["scenarios"].values()
                ]
            ),
            "",
            "## Scenario B capacity tradeoff",
            "",
            outcome_table(
                [
                    (
                        "Unconstrained"
                        if item["review_capacity"] is None
                        else f"Review ≤ {item['review_capacity']:.0%}",
                        item["selected_configuration"],
                    )
                    for item in artifact["scenarios"]["moderate"]["capacity_results"]
                ]
            ),
            "",
            "## Scenario B policy comparison",
            "",
            outcome_table(
                [
                    (policy_labels[name], result)
                    for name, result in artifact["scenarios"]["moderate"][
                        "policy_comparison"
                    ].items()
                ]
            ),
            "",
            "## Scenario B validation frontier",
            "",
            "These configurations serve different objectives; only the first minimizes estimated cost under Scenario B assumptions.",
            "",
            outcome_table(
                [
                    (item["objective"].replace("_", " "), item["configuration"])
                    for item in artifact["scenarios"]["moderate"]["frontier"]
                ]
            ),
            "",
            "## Provisional validation operating point",
            "",
        ]
    )
    provisional = artifact["provisional_validation_operating_point"]
    row = provisional["validation_outcome"]
    lines.extend(
        [
            f"Scenario B uses review threshold **{row['review_threshold']:.3f}** and block threshold **{row['block_threshold']:.3f}**. It is the lowest estimated-cost grid point feasible under the predeclared 2% review-capacity limit. It is provisional, validation-only, and merchant-dependent—not a final or universal threshold.",
            "",
            outcome_table([("Provisional validation point", row)]),
            "",
            "## Sensitivity analysis",
            "",
            "| Changed assumption | Value | Review threshold | Block threshold | Review rate | Total estimated cost |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in artifact["sensitivity_analysis"]:
        row = item["lowest_estimated_cost"]
        lines.append(
            f"| {item['parameter']} | {item['value']:.3f} | {row['review_threshold']:.3f} | "
            f"{row['block_threshold']:.3f} | {pct(row['review_rate'])} | {money(row['total_estimated_cost'])} |"
        )
    lines.extend(
        [
            "",
            "The movements show that threshold choice is a business decision as well as an ML decision. Only one Scenario B assumption changes in each row; all others remain at the declared Scenario B values.",
            "",
            "## Failure slices at the provisional point",
            "",
            "| Slice | Fraud rows | Approve | Review | Block |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, values in artifact["failure_slices"].items():
        lines.append(
            f"| `{name}` | {values['fraud_rows']:,} | {values['approve']['count']:,} ({pct(values['approve']['rate'])}) | "
            f"{values['review']['count']:,} ({pct(values['review']['rate'])}) | "
            f"{values['block']['count']:,} ({pct(values['block']['rate'])}) |"
        )
    high = artifact["high_value_fraud"]
    lines.extend(
        [
            "",
            "## High-value remaining failures",
            "",
            f"At the provisional point, {high['approve']['count']:,} fraud rows totaling {money(high['approve']['transaction_amount'])} remain approved; {high['review']['count']:,} totaling {money(high['review']['transaction_amount'])} are reviewed; and {high['block']['count']:,} totaling {money(high['block']['transaction_amount'])} are blocked.",
            "",
            "| TransactionID | Amount | Fraud probability | ProductCD | card4 |",
            "| ---: | ---: | ---: | --- | --- |",
        ]
    )
    for example in high["highest_value_approved_fraud_examples"]:
        lines.append(
            f"| {example['TransactionID']} | {money(example['TransactionAmt'])} | "
            f"{example['fraud_probability']:.6f} | {example['ProductCD']} | {example['card4']} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Validation was used for threshold selection, so these results are development evidence and will be optimistic for that operating choice.",
            "- Probabilities are uncalibrated; no probability calibrator was fitted in this phase.",
            "- Cost outputs depend directly on hypothetical merchant assumptions and review-effectiveness estimates.",
            "- The fixed cost model does not capture lifetime value, delayed fraud labels, reviewer queues, or customer recovery behavior.",
            "- No rules were created and no held-out observations were inspected.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    frame, balanced, validation_evidence = load_and_validate_predictions()
    config = merchant_scenario_config()
    if config.get("assumption_status") != "ILLUSTRATIVE MERCHANT ASSUMPTIONS":
        raise ValueError(
            "Merchant scenario assumptions must be explicitly labelled illustrative"
        )
    if config.get("cost_output_label") != ESTIMATED_COST_LABEL:
        raise ValueError("Merchant cost output label changed")

    labels = frame["actual_label"].to_numpy(dtype=int)
    amounts = frame["TransactionAmt"].to_numpy(dtype=float)
    scores = frame["fraud_probability"].to_numpy(dtype=float)
    balanced_scores = balanced["fraud_probability"].to_numpy(dtype=float)
    grid_config = config["threshold_grid"]
    review_values = threshold_candidates(
        float(grid_config["review_start"]),
        float(grid_config["review_stop"]),
        float(grid_config["step"]),
    )
    block_values = threshold_candidates(
        float(grid_config["block_start"]),
        float(grid_config["block_stop"]),
        float(grid_config["step"]),
    )

    grid_records: list[dict[str, Any]] = []
    main_grids: dict[str, list[dict[str, Any]]] = {}
    scenario_results: dict[str, Any] = {}
    scenario_names: dict[str, str] = {}
    for scenario_id, scenario in config["scenarios"].items():
        assumptions = CostAssumptions(**scenario["assumptions"])
        rows = evaluate_threshold_grid(
            labels=labels,
            amounts=amounts,
            risk_scores=scores,
            assumptions=assumptions,
            review_candidates=review_values,
            block_candidates=block_values,
            minimum_gap=float(grid_config["minimum_gap"]),
        )
        main_grids[scenario_id] = rows
        scenario_names[scenario_id] = scenario["name"]
        for row in rows:
            grid_records.append(
                {
                    "analysis_type": "merchant_scenario",
                    "scenario_id": scenario_id,
                    "sensitivity_parameter": None,
                    "sensitivity_value": np.nan,
                    **row,
                }
            )
        lowest = select_lowest_cost(rows)
        capacities = capacity_selections(rows, config["review_capacity_rates"])
        policies = policy_outcomes(
            labels=labels,
            amounts=amounts,
            selected_scores=scores,
            balanced_scores=balanced_scores,
            assumptions=assumptions,
            three_way=lowest,
        )
        scenario_results[scenario_id] = {
            "name": scenario["name"],
            "description": scenario["description"],
            "assumption_status": config["assumption_status"],
            "assumptions": assumptions.to_dict(),
            "evaluated_configuration_count": len(rows),
            "lowest_estimated_cost": summarize(lowest),
            "capacity_results": [
                {
                    **{
                        key: value
                        for key, value in item.items()
                        if key != "selected_configuration"
                    },
                    "selected_configuration": summarize(item["selected_configuration"]),
                }
                for item in capacities
            ],
            "frontier": frontier(
                rows, float(config["frontier"]["minimum_detected_fraud_recall"])
            ),
            "policy_comparison": policies,
        }

    provisional_spec = config["provisional_selection"]
    provisional_scenario = str(provisional_spec["scenario"])
    provisional_row = select_lowest_cost(
        main_grids[provisional_scenario],
        max_review_rate=float(provisional_spec["max_review_rate"]),
    )
    provisional_assumptions = CostAssumptions(
        **config["scenarios"][provisional_scenario]["assumptions"]
    )
    provisional_decisions = decisions_from_scores(
        scores,
        float(provisional_row["review_threshold"]),
        float(provisional_row["block_threshold"]),
    )
    provisional_policy = summarize(
        simulate_decisions(
            labels=labels,
            amounts=amounts,
            decisions=provisional_decisions,
            assumptions=provisional_assumptions,
        )
    )
    provisional_policy.update(
        {
            "review_threshold": provisional_row["review_threshold"],
            "block_threshold": provisional_row["block_threshold"],
        }
    )
    scenario_results[provisional_scenario]["policy_comparison"][
        "provisional_three_way_under_2pct_review"
    ] = provisional_policy

    sensitivity_results: list[dict[str, Any]] = []
    sensitivity_config = config["sensitivity"]
    base_scenario_id = str(sensitivity_config["base_scenario"])
    base_values = dict(config["scenarios"][base_scenario_id]["assumptions"])
    for parameter in (
        "fraud_loss_fraction",
        "legitimate_margin_rate",
        "manual_review_cost",
    ):
        for raw_value in sensitivity_config[parameter]:
            value = float(raw_value)
            if np.isclose(value, float(base_values[parameter]), rtol=0, atol=1e-12):
                selected = select_lowest_cost(main_grids[base_scenario_id])
                source = "base_scenario_grid"
            else:
                changed = {**base_values, parameter: value}
                sensitivity_rows = evaluate_threshold_grid(
                    labels=labels,
                    amounts=amounts,
                    risk_scores=scores,
                    assumptions=CostAssumptions(**changed),
                    review_candidates=review_values,
                    block_candidates=block_values,
                    minimum_gap=float(grid_config["minimum_gap"]),
                )
                for row in sensitivity_rows:
                    grid_records.append(
                        {
                            "analysis_type": "one_at_a_time_sensitivity",
                            "scenario_id": base_scenario_id,
                            "sensitivity_parameter": parameter,
                            "sensitivity_value": value,
                            **row,
                        }
                    )
                selected = select_lowest_cost(sensitivity_rows)
                source = "one_at_a_time_grid"
            sensitivity_results.append(
                {
                    "base_scenario": base_scenario_id,
                    "parameter": parameter,
                    "value": value,
                    "other_assumptions_held_at_base": True,
                    "grid_source": source,
                    "lowest_estimated_cost": summarize(selected),
                }
            )

    failure_slices, high_value = residual_analysis(
        frame,
        float(provisional_row["review_threshold"]),
        float(provisional_row["block_threshold"]),
    )

    figures = plot_results(
        main_grids,
        provisional_row,
        scenario_results[provisional_scenario]["policy_comparison"],
        scenario_names,
    )
    generated_at = datetime.now(UTC).isoformat()
    artifact = {
        "artifact_type": "validation_cost_threshold_analysis",
        "status": "provisional_validation_analysis",
        "generated_at": generated_at,
        "selection_split": "validation",
        "held_out_test_accessed": False,
        "catboost_retrained": False,
        "fraud_rules_implemented": False,
        "merchant_facing_final_metrics_updated": False,
        "assumption_status": config["assumption_status"],
        "cost_output_label": ESTIMATED_COST_LABEL,
        "model": {
            "name": validation_evidence["model_metadata"]["model_name"],
            "version": validation_evidence["model_metadata"]["model_version"],
            "experiment_id": EXPECTED_EXPERIMENT,
            "feature_set": validation_evidence["model_metadata"]["feature_set"],
            "feature_names": validation_evidence["model_metadata"]["feature_names"],
            "class_weight": validation_evidence["model_metadata"]["class_weight"],
        },
        "data_validation": {
            "rows": len(frame),
            "fraud_count": int(labels.sum()),
            "legitimate_count": int((labels == 0).sum()),
            "unique_transaction_ids": int(frame["TransactionID"].nunique()),
            "transaction_amount_join_coverage": float(
                frame["TransactionAmt"].notna().mean()
            ),
            **validation_evidence["reproduced_metrics_at_0_5"],
        },
        "threshold_grid": {
            **grid_config,
            "review_candidates": review_values,
            "block_candidates": block_values,
            "total_stored_rows": len(grid_records),
            "all_evaluated_combinations_stored_in": str(
                THRESHOLD_GRID.relative_to(ARTIFACTS.parent)
            ),
        },
        "scenarios": scenario_results,
        "provisional_validation_operating_point": {
            "status": "provisional_validation_config",
            "scenario": provisional_scenario,
            "scenario_name": config["scenarios"][provisional_scenario]["name"],
            "reason": provisional_spec["objective"],
            "max_review_rate": provisional_spec["max_review_rate"],
            "validation_outcome": summarize(provisional_row),
            "cost_assumptions": provisional_assumptions.to_dict(),
        },
        "sensitivity_analysis": sensitivity_results,
        "failure_slices": failure_slices,
        "high_value_fraud": high_value,
        "figures": figures,
        "integrity": {
            "catboost_retrained": "NO",
            "held_out_test_accessed": "NO",
            "thresholds_selected_using_validation_only": "YES",
            "merchant_assumptions_clearly_labelled": "YES",
            "money_saved_claims_presented_as_estimates": "YES",
            "fraud_rules_implemented": "NO",
            "final_merchant_facing_metrics_updated": "NO",
        },
    }

    grid_frame = pd.DataFrame(grid_records)
    validate_threshold_grid_artifact(grid_frame)
    THRESHOLD_JSON.parent.mkdir(parents=True, exist_ok=True)
    grid_frame.to_parquet(THRESHOLD_GRID, index=False)
    THRESHOLD_JSON.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    operating = {
        "status": "provisional_validation_config",
        "not_final": True,
        "generated_at": generated_at,
        "model_name": artifact["model"]["name"],
        "model_version": artifact["model"]["version"],
        "experiment_id": EXPECTED_EXPERIMENT,
        "feature_set": artifact["model"]["feature_set"],
        "selection_split": "validation",
        "held_out_test_accessed": False,
        "scenario": provisional_scenario,
        "scenario_name": config["scenarios"][provisional_scenario]["name"],
        "review_threshold": provisional_row["review_threshold"],
        "block_threshold": provisional_row["block_threshold"],
        "review_capacity_limit": provisional_spec["max_review_rate"],
        "selection_reason": provisional_spec["objective"],
        "assumption_status": config["assumption_status"],
        "cost_output_label": ESTIMATED_COST_LABEL,
        "cost_assumptions": provisional_assumptions.to_dict(),
        "validation_metrics": summarize(provisional_row),
        "limitations": [
            "Selected on validation and not evaluated on the sealed held-out test.",
            "Merchant economics and review-effectiveness inputs are illustrative assumptions.",
            "This is not a final, production, or universal threshold recommendation.",
        ],
    }
    OPERATING_CONFIG.write_text(
        json.dumps(operating, indent=2) + "\n", encoding="utf-8"
    )
    write_report(artifact)
    print(
        json.dumps(
            {
                "rows": len(frame),
                "fraud": int(labels.sum()),
                "grid_rows": len(grid_frame),
                "provisional_review_threshold": provisional_row["review_threshold"],
                "provisional_block_threshold": provisional_row["block_threshold"],
                "provisional_review_rate": provisional_row["review_rate"],
                "provisional_total_estimated_cost": provisional_row[
                    "total_estimated_cost"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
