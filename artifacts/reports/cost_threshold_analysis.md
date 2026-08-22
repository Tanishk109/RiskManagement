# Validation Cost and Threshold Analysis

**ESTIMATED BUSINESS COST UNDER USER-SUPPLIED ASSUMPTIONS**

This is validation-only development evidence, not held-out or production performance. All three scenarios are **ILLUSTRATIVE MERCHANT ASSUMPTIONS**, not industry facts.

## Data and model integrity

The saved `cb-04-without-identity-none` validation predictions contain 88,581 rows, including 3,042 fraud and 85,539 legitimate transactions. The fixed-threshold metrics reproduce the frozen candidate artifact. CatBoost was not retrained and the held-out test was not accessed.

## Merchant scenarios

| Scenario | Fraud loss fraction | Fraud fixed cost | Margin rate | FP fixed cost | Review cost | Review catch | Legit approval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scenario A — Low-margin / low-review-cost merchant | 60.000% | INR 0.00 | 8.000% | INR 5.00 | INR 10.00 | 80.000% | 97.000% |
| Scenario B — Moderate merchant | 85.000% | INR 50.00 | 18.000% | INR 20.00 | INR 25.00 | 90.000% | 98.000% |
| Scenario C — High-margin / high-fraud-cost merchant | 100.000% | INR 150.00 | 30.000% | INR 50.00 | INR 60.00 | 92.000% | 97.000% |

A fraud approval incurs the configured amount fraction plus the explicitly configured fixed fraud cost. A legitimate block incurs estimated lost contribution plus the fixed false-positive cost. A review incurs its manual cost on every reviewed row plus expected residual fraud/false-positive cost. These are simplifying assumptions and are configurable.

## Lowest estimated cost by scenario

| Configuration | Review | Block | Review rate | Block rate | Detected recall | Amount capture | FP | FN | Fraud loss | FP cost | Manual review cost | Total cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scenario A — Low-margin / low-review-cost merchant | 0.150 | 0.225 | 0.963% | 2.162% | 42.045% | 42.483% | 800 | 1,763 | INR 171,482.67 | INR 15,062.27 | INR 8,530.00 | INR 195,074.94 |
| Scenario B — Moderate merchant | 0.175 | 0.250 | 0.713% | 1.961% | 40.039% | 41.163% | 667 | 1,824 | INR 340,451.48 | INR 33,596.47 | INR 15,800.00 | INR 389,847.95 |
| Scenario C — High-margin / high-fraud-cost merchant | 0.175 | 0.250 | 0.713% | 1.961% | 40.039% | 41.263% | 667 | 1,824 | INR 567,245.81 | INR 67,778.95 | INR 37,920.00 | INR 672,944.76 |

## Scenario B capacity tradeoff

| Configuration | Review | Block | Review rate | Block rate | Detected recall | Amount capture | FP | FN | Fraud loss | FP cost | Manual review cost | Total cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconstrained | 0.175 | 0.250 | 0.713% | 1.961% | 40.039% | 41.163% | 667 | 1,824 | INR 340,451.48 | INR 33,596.47 | INR 15,800.00 | INR 389,847.95 |
| Review ≤ 1% | 0.175 | 0.250 | 0.713% | 1.961% | 40.039% | 41.163% | 667 | 1,824 | INR 340,451.48 | INR 33,596.47 | INR 15,800.00 | INR 389,847.95 |
| Review ≤ 2% | 0.175 | 0.250 | 0.713% | 1.961% | 40.039% | 41.163% | 667 | 1,824 | INR 340,451.48 | INR 33,596.47 | INR 15,800.00 | INR 389,847.95 |
| Review ≤ 5% | 0.175 | 0.250 | 0.713% | 1.961% | 40.039% | 41.163% | 667 | 1,824 | INR 340,451.48 | INR 33,596.47 | INR 15,800.00 | INR 389,847.95 |

## Scenario B policy comparison

| Configuration | Review | Block | Review rate | Block rate | Detected recall | Amount capture | FP | FN | Fraud loss | FP cost | Manual review cost | Total cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Approve all | — | — | 0.000% | 0.000% | 0.000% | 0.000% | 0 | 3,042 | INR 574,471.45 | INR 0.00 | INR 0.00 | INR 574,471.45 |
| Selected CatBoost: binary block at 0.50 | — | — | 0.000% | 1.083% | 24.260% | 23.862% | 221 | 2,304 | INR 436,783.67 | INR 11,387.51 | INR 0.00 | INR 448,171.18 |
| Balanced CatBoost artifact: binary block at 0.50 | — | — | 0.000% | 14.271% | 66.831% | 69.564% | 10,608 | 1,009 | INR 179,004.57 | INR 520,904.82 | INR 0.00 | INR 699,909.39 |
| Three-way lowest estimated cost | 0.175 | 0.250 | 0.713% | 1.961% | 40.039% | 41.163% | 667 | 1,824 | INR 340,451.48 | INR 33,596.47 | INR 15,800.00 | INR 389,847.95 |
| Provisional three-way under 2% review | 0.175 | 0.250 | 0.713% | 1.961% | 40.039% | 41.163% | 667 | 1,824 | INR 340,451.48 | INR 33,596.47 | INR 15,800.00 | INR 389,847.95 |

## Scenario B validation frontier

These configurations serve different objectives; only the first minimizes estimated cost under Scenario B assumptions.

| Configuration | Review | Block | Review rate | Block rate | Detected recall | Amount capture | FP | FN | Fraud loss | FP cost | Manual review cost | Total cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lowest estimated cost | 0.175 | 0.250 | 0.713% | 1.961% | 40.039% | 41.163% | 667 | 1,824 | INR 340,451.48 | INR 33,596.47 | INR 15,800.00 | INR 389,847.95 |
| highest detected fraud recall under 1pct review | 0.125 | 0.150 | 0.685% | 3.125% | 44.576% | 45.425% | 1,489 | 1,686 | INR 315,193.95 | INR 76,277.24 | INR 15,175.00 | INR 406,646.20 |
| highest detected fraud recall under 2pct review | 0.100 | 0.150 | 1.891% | 3.125% | 49.441% | 48.707% | 1,489 | 1,538 | INR 294,672.69 | INR 77,179.57 | INR 41,875.00 | INR 413,727.26 |
| highest fraud amount capture under 2pct review | 0.100 | 0.125 | 1.206% | 3.810% | 49.441% | 48.912% | 2,019 | 1,538 | INR 293,422.08 | INR 102,904.66 | INR 26,700.00 | INR 423,026.74 |
| lowest false positive cost at min 50% detected recall | 0.075 | 0.950 | 6.882% | 0.219% | 54.569% | 49.373% | 17 | 1,382 | INR 290,350.79 | INR 5,268.10 | INR 152,400.00 | INR 448,018.89 |

## Provisional validation operating point

Scenario B uses review threshold **0.175** and block threshold **0.250**. It is the lowest estimated-cost grid point feasible under the predeclared 2% review-capacity limit. It is provisional, validation-only, and merchant-dependent—not a final or universal threshold.

| Configuration | Review | Block | Review rate | Block rate | Detected recall | Amount capture | FP | FN | Fraud loss | FP cost | Manual review cost | Total cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Provisional validation point | 0.175 | 0.250 | 0.713% | 1.961% | 40.039% | 41.163% | 667 | 1,824 | INR 340,451.48 | INR 33,596.47 | INR 15,800.00 | INR 389,847.95 |

## Sensitivity analysis

| Changed assumption | Value | Review threshold | Block threshold | Review rate | Total estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| fraud_loss_fraction | 0.600 | 0.175 | 0.250 | 0.713% | INR 316,756.34 |
| fraud_loss_fraction | 0.850 | 0.175 | 0.250 | 0.713% | INR 389,847.95 |
| fraud_loss_fraction | 1.000 | 0.150 | 0.250 | 1.164% | INR 432,696.93 |
| legitimate_margin_rate | 0.080 | 0.150 | 0.175 | 0.450% | INR 376,624.37 |
| legitimate_margin_rate | 0.180 | 0.175 | 0.250 | 0.713% | INR 389,847.95 |
| legitimate_margin_rate | 0.300 | 0.175 | 0.375 | 1.297% | INR 397,854.64 |
| manual_review_cost | 10.000 | 0.050 | 0.450 | 9.912% | INR 340,757.75 |
| manual_review_cost | 25.000 | 0.175 | 0.250 | 0.713% | INR 389,847.95 |
| manual_review_cost | 60.000 | 0.175 | 0.200 | 0.294% | INR 402,341.90 |

The movements show that threshold choice is a business decision as well as an ML decision. Only one Scenario B assumption changes in each row; all others remain at the declared Scenario B values.

## Failure slices at the provisional point

| Slice | Fraud rows | Approve | Review | Block |
| --- | ---: | ---: | ---: | ---: |
| `ProductCD=W` | 1,485 | 1,237 (83.300%) | 72 (4.848%) | 176 (11.852%) |
| `TransactionAmt>=500` | 224 | 119 (53.125%) | 11 (4.911%) | 94 (41.964%) |
| `card4=discover` | 122 | 59 (48.361%) | 11 (9.016%) | 52 (42.623%) |
| `ProductCD=S` | 107 | 58 (54.206%) | 11 (10.280%) | 38 (35.514%) |

## High-value remaining failures

At the provisional point, 1,824 fraud rows totaling INR 289,883.25 remain approved; 148 totaling INR 24,832.03 are reviewed; and 1,070 totaling INR 182,192.31 are blocked.

| TransactionID | Amount | Fraud probability | ProductCD | card4 |
| ---: | ---: | ---: | --- | --- |
| 3455844 | INR 3,822.95 | 0.008379 | W | visa |
| 3436956 | INR 3,260.05 | 0.058370 | W | mastercard |
| 3418094 | INR 2,963.95 | 0.006177 | W | visa |
| 3418107 | INR 2,963.95 | 0.008556 | W | visa |
| 3455964 | INR 2,268.39 | 0.008379 | W | visa |
| 3488473 | INR 2,161.00 | 0.025877 | W | visa |
| 3476245 | INR 1,795.80 | 0.002062 | W | visa |
| 3403485 | INR 1,651.00 | 0.144229 | W | visa |
| 3409798 | INR 1,651.00 | 0.011979 | W | visa |
| 3409807 | INR 1,651.00 | 0.011979 | W | visa |

## Limitations

- Validation was used for threshold selection, so these results are development evidence and will be optimistic for that operating choice.
- Probabilities are uncalibrated; no probability calibrator was fitted in this phase.
- Cost outputs depend directly on hypothetical merchant assumptions and review-effectiveness estimates.
- The fixed cost model does not capture lifetime value, delayed fraud labels, reviewer queues, or customer recovery behavior.
- No rules were created and no held-out observations were inspected.
