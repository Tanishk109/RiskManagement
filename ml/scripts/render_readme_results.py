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
        body = "\n".join([
            "| Held-out metric | Value |",
            "|---|---:|",
            f"| Transactions | {int(metrics['test_transaction_count']):,} |",
            f"| Fraud cases | {int(metrics['fraud_count']):,} |",
            f"| Precision at block threshold | {float(metrics['precision']):.6f} |",
            f"| Recall at block threshold | {float(metrics['recall']):.6f} |",
            f"| F1 | {float(metrics['f1']):.6f} |",
            f"| Average precision / PR-AUC | {float(metrics['average_precision']):.6f} |",
            f"| False positives | {int(metrics['false_positives']):,} |",
            f"| False negatives | {int(metrics['false_negatives']):,} |",
            f"| Total estimated cost ({metrics['business_assumptions']['currency']}) | {float(metrics['total_estimated_cost']):,.2f} |",
            "",
            "Calculated from the held-out temporal test set using the business assumptions listed in the final evaluation artifact.",
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
