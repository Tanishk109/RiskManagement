# Return-risk cancellation proxy evaluation

Data source: UCI Online Retail II (dataset 502).

> The dataset provides cancellation/reversal outcomes and is used as a proxy for return-risk research; it is not a perfect physical-return label.

The unit of modeling is an invoice/order, not a line item. Invoice identifiers are used only to construct the proxy label and are excluded from model features. Signed quantities are converted to magnitudes because negative quantity directly exposes cancellation rows.

## Chronological partitions

| Partition | Orders | Cancellations | Prevalence | Time range |
| --- | ---: | ---: | ---: | --- |
| Train | 37,539 | 5,972 | 15.91% | 2009-12-01T07:45:00 to 2011-05-17T14:16:00 |
| Validation | 8,044 | 1,230 | 15.29% | 2011-05-17T14:32:00 to 2011-09-21T09:54:00 |
| Test | 8,045 | 1,090 | 13.55% | 2011-09-21T09:58:00 to 2011-12-09T12:50:00 |

## Measured performance

Classification metrics use the documented HIGH boundary of 0.50. Average Precision is threshold-free.

| Model / partition | Precision | Recall | F1 | Average Precision |
| --- | ---: | ---: | ---: | ---: |
| Logistic baseline / validation | 0.4231 | 0.9122 | 0.5781 | 0.6244 |
| Logistic baseline / test | 0.3757 | 0.8550 | 0.5220 | 0.5316 |
| CatBoost candidate / validation | 0.6887 | 0.9407 | 0.7952 | 0.8953 |
| CatBoost candidate / test | 0.6617 | 0.8972 | 0.7617 | 0.8562 |

## Leakage controls

- Partitions are chronological and the UCI test partition is never used for fitting or early stopping.
- Customer history uses cumulative values shifted to exclude the current and all future orders.
- Missing customer IDs never share one synthetic history bucket.
- Invoice number, description, cancellation prefix, signed quantity, and outcome are excluded from features.
- The payment-fraud IEEE-CIS model and held-out test were not opened or modified.

## Product interpretation

LOW is below 0.15, MEDIUM is 0.15 to below 0.50, and HIGH is at least 0.50. These labels prioritize merchant review and never automatically reject an order.

The proxy may include reversals, corrections, and administrative cancellations that are not physical returns. Performance must not be presented as physical-return performance.
