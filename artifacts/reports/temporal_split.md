# IEEE-CIS Chronological Split Report

## Why chronological splitting was chosen

Fraud detection predicts future transactions. A random split would mix earlier and later regimes and could make evaluation unrealistically easy. Rows were therefore ordered by `TransactionDT`, using `TransactionID` only as a deterministic secondary key.

## Exact split boundaries

Boundary policy: nearest clean TransactionDT boundary; ties choose the earlier cut.

| Partition | TransactionDT minimum | TransactionDT maximum |
| --- | ---: | ---: |
| Train | 86,400 | 10,437,996 |
| Validation | 10,438,003 | 13,151,840 |
| Held-out test | 13,151,880 | 15,811,131 |

The selected boundaries are clean: identical `TransactionDT` values are not split across partitions.

## Partition sizes and fraud distribution

| Partition | Rows | Actual share | Fraud | Legitimate | Fraud prevalence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 413,378 | 70.000000% | 14,538 | 398,840 | 3.516878% |
| Validation | 88,581 | 15.000000% | 3,042 | 85,539 | 3.434145% |
| Held-out test | 88,581 | 15.000000% | 3,083 | 85,498 | 3.480430% |

No partition was stratified or rebalanced; the observed prevalence changes are preserved.

## Transaction amount distribution

| Partition | Mean | P25 | Median | P75 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 134.6049 | 42.9500 | 68.9500 | 125.0000 | 444.9500 | 1,104.0000 | 31,937.3910 |
| Validation | 134.8765 | 43.9500 | 67.9500 | 117.0000 | 440.3900 | 1,104.0000 | 5,279.9500 |
| Held-out test | 137.1485 | 44.0000 | 68.5000 | 117.0000 | 445.0000 | 1,226.8940 | 5,366.8200 |

## Identity availability stability

`identity_available` is true when `TransactionID` matched a row in `train_identity.csv`; it is not inferred from any nullable identity feature.

| Partition | Overall coverage | Fraud coverage | Legitimate coverage |
| --- | ---: | ---: | ---: |
| Train | 26.588498% | 55.282707% | 25.542573% |
| Validation | 17.671961% | 49.901381% | 16.525795% |
| Held-out test | 21.074497% | 57.184560% | 19.772392% |

## Observed temporal distribution changes

The largest category-share change for each requested field is shown below. These are descriptive shifts, not model metrics.

| Field | Category | Train | Validation | Test | Maximum spread |
| --- | --- | ---: | ---: | ---: | ---: |
| `ProductCD` | W | 72.067454% | 81.610052% | 78.423138% | 9.542598 pp |
| `card4` | mastercard | 31.682141% | 32.869351% | 32.889672% | 1.207531 pp |
| `card6` | debit | 73.334091% | 78.727944% | 75.696820% | 5.393853 pp |
| `DeviceType` | <MISSING> | 74.019662% | 82.805568% | 79.474154% | 8.785905 pp |
| `identity_available` | False | 73.411502% | 82.328039% | 78.925503% | 8.916536 pp |

## Missingness stability

A substantial shift is defined before inspection as an absolute max-minus-min difference of at least 5.0 percentage points across partitions.

| Column | Train missing | Validation missing | Test missing | Maximum spread |
| --- | ---: | ---: | ---: | ---: |
| `D3` | 47.106764% | 38.907892% | 38.026213% | 9.080551 pp |
| `DeviceType` | 74.019662% | 82.805568% | 79.474154% | 8.785905 pp |
| `DeviceInfo` | 77.898679% | 86.212619% | 82.963615% | 8.313940 pp |
| `D2` | 49.914122% | 42.223502% | 41.838543% | 8.075579 pp |
| `R_emaildomain` | 75.073419% | 82.239984% | 79.094840% | 7.166564 pp |

## Evaluation policy

```text
TRAIN
→ fit preprocessing and model

VALIDATION
→ model comparison, feature decisions, calibration,
  thresholds, cost optimization and rule design

HELD-OUT TEST
→ final reporting only
```

No model or preprocessing object was fitted while creating these partitions. Precision, recall, F1, PR-AUC, FP/FN, thresholds, and merchant-cost metrics remain **Not evaluated yet**.
