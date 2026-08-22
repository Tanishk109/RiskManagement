# IEEE-CIS EDA Summary

Generated from the official local labeled IEEE-CIS training files. The Kaggle test files were not used.
No model was trained and no temporal train/validation/test split was created in this phase.

## Evidence status

- Precision, recall, F1, PR-AUC, FP/FN, thresholds, and cost savings: **Not evaluated yet**
- Merchant-facing ML metrics fabricated: **No**

## Source files

- `train_transaction.csv`: 683,351,067 bytes
- `train_identity.csv`: 26,529,680 bytes

## Dataset validation

- Transactions: 590,540
- Identity rows: 144,233
- Fraud: 20,663
- Legitimate: 569,877
- Fraud prevalence: 3.499001%
- `TransactionDT` range: 86,400 to 15,811,131
- Identity join coverage: 24.423917% (144,233 transactions)
- Joined feature columns inspected for missingness: 434

## TransactionAmt summary

| Group | Count | Mean | Median | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 590,540 | 135.0272 | 68.7690 | 445.0000 | 1,104.0000 | 31,937.3910 |
| Legitimate | 569,877 | 134.5117 | 68.5000 | 435.0000 | 1,104.0000 | 31,937.3910 |
| Fraud | 20,663 | 149.2448 | 75.0000 | 500.0000 | 994.0000 | 5,191.0000 |

The amount-distribution figure uses `log1p(TransactionAmt)` only for display; it includes every non-missing row and does not cap or remove outliers.

## Identity availability by actual label

| Label | Transactions | With identity | Coverage |
| --- | ---: | ---: | ---: |
| Legitimate | 569,877 | 132,915 | 23.323454% |
| Fraud | 20,663 | 11,318 | 54.774234% |

## Temporal prevalence

Across 20 equal-duration chronological buckets, fraud prevalence ranges from 2.030319% to 4.776534%.
`TransactionDT` is treated only as a relative ordering variable, not converted to a calendar date.

## Highest missingness after the left join

12 of 434 columns are more than 90% missing.

| Column | Missing |
| --- | ---: |
| `id_24` | 99.196159% |
| `id_25` | 99.130965% |
| `id_07` | 99.127070% |
| `id_08` | 99.127070% |
| `id_21` | 99.126393% |
| `id_26` | 99.125715% |
| `id_27` | 99.124699% |
| `id_23` | 99.124699% |
| `id_22` | 99.124699% |
| `dist2` | 93.628374% |
| `D7` | 93.409930% |
| `id_18` | 92.360721% |
| `D13` | 89.509263% |
| `D14` | 89.469469% |
| `D12` | 89.041047% |
| `id_03` | 88.768923% |
| `id_04` | 88.768923% |
| `D6` | 87.606767% |
| `id_33` | 87.589494% |
| `id_10` | 87.312290% |

## Supported categorical fraud rates

Categories below meet the minimum support of 591 transactions. The five highest fraud-rate categories per field are shown; the JSON report contains all supported categories.

### ProductCD

| Category | Transactions | Fraud | Fraud rate |
| --- | ---: | ---: | ---: |
| C | 68,519 | 8,008 | 11.687269% |
| S | 11,628 | 686 | 5.899553% |
| H | 33,024 | 1,574 | 4.766231% |
| R | 37,699 | 1,426 | 3.782594% |
| W | 439,670 | 8,969 | 2.039939% |

### card4

| Category | Transactions | Fraud | Fraud rate |
| --- | ---: | ---: | ---: |
| discover | 6,651 | 514 | 7.728161% |
| visa | 384,767 | 13,373 | 3.475610% |
| mastercard | 189,217 | 6,496 | 3.433095% |
| american express | 8,328 | 239 | 2.869837% |
| <MISSING> | 1,577 | 41 | 2.599873% |

### card6

| Category | Transactions | Fraud | Fraud rate |
| --- | ---: | ---: | ---: |
| credit | 148,986 | 9,950 | 6.678480% |
| <MISSING> | 1,571 | 39 | 2.482495% |
| debit | 439,938 | 10,674 | 2.426251% |

### DeviceType

| Category | Transactions | Fraud | Fraud rate |
| --- | ---: | ---: | ---: |
| mobile | 55,645 | 5,657 | 10.166232% |
| desktop | 85,165 | 5,554 | 6.521458% |
| <MISSING> | 449,730 | 9,452 | 2.101705% |

### P_emaildomain

| Category | Transactions | Fraud | Fraud rate |
| --- | ---: | ---: | ---: |
| outlook.com | 5,096 | 482 | 9.458399% |
| live.com.mx | 749 | 41 | 5.473965% |
| hotmail.com | 45,250 | 2,396 | 5.295028% |
| gmail.com | 228,355 | 9,943 | 4.354185% |
| icloud.com | 6,267 | 197 | 3.143450% |

### R_emaildomain

| Category | Transactions | Fraud | Fraud rate |
| --- | ---: | ---: | ---: |
| outlook.com | 2,507 | 414 | 16.513761% |
| icloud.com | 1,398 | 180 | 12.875536% |
| gmail.com | 57,147 | 6,811 | 11.918386% |
| hotmail.com | 27,509 | 2,140 | 7.779272% |
| live.com.mx | 754 | 44 | 5.835544% |

## Figures

- `artifacts/figures/class_balance.png`
- `artifacts/figures/transaction_amount_distribution.png`
- `artifacts/figures/fraud_rate_over_time.png`
- `artifacts/figures/top_missingness.png`
- `artifacts/figures/identity_coverage_by_label.png`
- `artifacts/figures/categorical_fraud_rates.png`

## Leakage guardrails before modeling

- `isFraud` is the target and must never enter a feature matrix or rule condition.
- `TransactionID` is an identifier aligned with dataset order and is excluded as a model feature.
- `TransactionDT` may encode time/regime drift; it is retained only as an available-at-transaction-time candidate and requires chronological validation.
- Any future aggregate or velocity feature must use only transactions strictly earlier than the scored row.
- Identity availability and masked `C*`, `D*`, and `V*` fields require scoring-time availability checks before retention.
