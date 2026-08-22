# Fraud Pulse

Fraud Pulse is a transparent score-volume monitor around the existing frozen CatBoost validation candidate. It does not train or introduce another classifier.

## Data and isolation

- Validation replay reads `TransactionID`, `TransactionDT`, and `TransactionAmt` from the chronological IEEE-CIS validation Parquet and joins them one-to-one to `fraud_probability` and `model_version` from the CatBoost validation predictions.
- The replay deliberately does not read `isFraud` or `actual_label`; ground truth is not an input to spike detection.
- Both artifact paths must be explicitly validation-scoped. Paths containing `test`, `held`, or `heldout` are rejected before any file read.
- Merchant uploads require `EventTime` plus the exact 13 Risk Check fields. Every valid row goes through the existing frozen CatBoost batch scorer before time aggregation. `isFraud` and `actual_label` are forbidden. Uploads are limited to 1 MB and 1,000 rows and are not retained.

## Window statistics

Each chronological window reports transaction count, mean risk score, high-risk count, review count, block count, and high-risk transaction amount. “High risk” means a probability at or above the current provisional review threshold. Review and block use the existing provisional validation thresholds; Fraud Pulse does not modify them.

## Detector methods

- **Rolling z-score** compares the current metric with the mean and population standard deviation of the configured prior windows.
- **EWMA deviation** compares the current metric with an exponentially weighted baseline over the configured prior windows.
- **Percent deviation** compares the current metric with the rolling mean of the configured prior windows.

Every window declares whether its baseline is `WARMING_UP` or `READY`, shows the baseline and change when ready, and marks an active `SPIKE ALERT` only when the selected rule crosses its visible sensitivity. Alerts are configuration-dependent changes in score behavior, not confirmed attacks.

## Evaluation status and limitations

Detector performance is **Not evaluated yet**. The validation replay is operational development evidence, not held-out detector evaluation. No precision, recall, loss prevention, or incident-confirmation claim is attached to alerts. Alert counts vary with the selected window, baseline length, metric, and sensitivity.
