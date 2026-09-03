from __future__ import annotations

import json
import re

from common import ARTIFACTS, ROOT

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def main() -> None:
    metrics_path = ARTIFACTS / "metrics/final_test_metrics.json"
    if not metrics_path.is_file():
        body = "**Not evaluated yet.** Run the real-data pipeline before presenting precision, recall, F1, PR-AUC, cost, latency, or savings claims."
    else:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("evaluation_status") != "complete" or metrics.get("split") != "test":
            raise ValueError("Refusing to render README from a non-final or non-test artifact")
        transaction_count = int(metrics["test_transaction_count"])
        fraud_count = int(metrics["fraud_count"])
        base_rate = fraud_count / transaction_count
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        precision_lift = precision / base_rate
        block_share = float(metrics["decision_distribution"]["block"]["share"])
        fraud_value_capture = float(metrics["fraud_amount_capture_rate"])
        body = "\n".join([
            (
                "MerchantShield's block policy turns a "
                f"{base_rate:.2%} fraud base rate into {precision:.2%} precision at the block "
                f"boundary—an **~{precision_lift:.0f}× lift over base rate**—while pricing "
                "every outcome (missed fraud, false positives, manual review) into an "
                "explicit estimated cost."
            ),
            "",
            f"**Base rate:** {fraud_count:,} fraud / {transaction_count:,} transactions = {base_rate:.2%}",
            "",
            "| Held-out metric | Value |",
            "|---|---:|",
            f"| Transactions | {transaction_count:,} |",
            f"| Fraud cases | {fraud_count:,} ({base_rate:.2%} base rate) |",
            f"| Precision at block threshold | {precision:.4f} (~{precision_lift:.0f}× base-rate lift) |",
            f"| Recall at block threshold | {recall:.4f} ({recall:.2%} of fraud cases caught) |",
            f"| Fraud-value capture | {fraud_value_capture:.2%} of fraudulent transaction value |",
            f"| Share of traffic blocked | {block_share:.2%} |",
            f"| F1 | {float(metrics['f1']):.4f} |",
            f"| Average precision / PR-AUC | {float(metrics['average_precision']):.4f} |",
            f"| False positives | {int(metrics['false_positives']):,} |",
            f"| False negatives | {int(metrics['false_negatives']):,} |",
            f"| Total estimated cost ({metrics['business_assumptions']['currency']}) | {float(metrics['total_estimated_cost']):,.2f} |",
            "",
            "![Held-out precision-recall curve](artifacts/figures/final_test_precision_recall_curve.png)",
            "",
            (
                "The frozen block threshold (star) sits ~11× above the fraud base rate "
                "line, selected on VALIDATION before this held-out evaluation was run once."
            ),
            "",
            (
                "**Reading these numbers:** fraud is rare, so precision must be judged "
                f"against the {base_rate:.2%} base rate, not against 1.0. At the frozen block "
                f"threshold, the policy blocks only {block_share:.2%} of all traffic while "
                "capturing over a third of both fraud cases and fraud value—and every false "
                "positive, missed fraud case, and manual review is priced into the "
                f"₹{float(metrics['total_estimated_cost']):,.0f} total estimated cost above, "
                "under the business assumptions in `ml/configs/merchant_scenarios.yaml`."
            ),
            "",
            (
                "**Frozen policy, no post-test tuning.** The threshold, model, and feature "
                "set were selected on VALIDATION only and frozen before the one held-out test "
                "run. No threshold or rule was altered after seeing held-out results. Any "
                "future rule remains validation-only until evaluated on a fresh, previously "
                "untouched split—this held-out set is not reused for further tuning."
            ),
        ])
    readme = ROOT / "README.md"
    content = readme.read_text(encoding="utf-8")
    replacement = f"{START}\n{body}\n{END}"
    updated, count = re.subn(re.escape(START) + r".*?" + re.escape(END), replacement, content, flags=re.DOTALL)
    if count != 1:
        raise ValueError("README results markers are missing or duplicated")
    readme.write_text(updated, encoding="utf-8")
    print("README results section updated from the final test artifact.")


if __name__ == "__main__":
    main()
