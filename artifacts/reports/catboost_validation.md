# CatBoost Validation Candidate

## Objective

Compare one nonlinear, native-categorical model family against the frozen Logistic Regression baseline using TRAIN for fitting and VALIDATION for early stopping, selection, and evaluation.

## Why CatBoost

CatBoost handles the mixed numerical/categorical conservative feature set natively. Categorical values use one consistent `__MISSING__` token; numerical NaN values are preserved. No scaling, one-hot encoding, external target encoding, or V features are used.

## Frozen Logistic Regression Baseline

`lr-07-conservative_combined-none`: AP 0.231434, ROC-AUC 0.793480, FP 83, FN 2,825 at threshold 0.50.

## Features

Selected raw features: `TransactionAmt`, `ProductCD`, `card4`, `card6`, `P_emaildomain`, `C1`, `C2`, `C3`, `C4`, `C5`, `D1`, `D2`, `D3`.

Raw `TransactionDT` and all V features were excluded. `TransactionID` and `isFraud` were forbidden as predictors.

## Experiments

All three main experiments share one parameter configuration and use CatBoost `PRAUC:type=Classic` for early stopping. External metrics come from sklearn; threshold metrics use the descriptive 0.50 cutoff.

| Experiment | Weight | AP | ROC-AUC | Precision | Recall | F1 | FP | FN | Seconds | Best iteration | Trees |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cb-01-combined-none | none | 0.427701 | 0.859978 | 0.779396 | 0.246220 | 0.374219 | 212 | 2,293 | 209.005 | 998 | 999 |
| cb-02-combined-balanced | Balanced | 0.410351 | 0.862914 | 0.160826 | 0.668310 | 0.259262 | 10,608 | 1,009 | 131.856 | 521 | 522 |
| cb-03-combined-sqrt-balanced | SqrtBalanced | 0.413876 | 0.861801 | 0.492029 | 0.375411 | 0.425881 | 1,179 | 1,900 | 111.378 | 391 | 392 |
| cb-04-without-identity-none | none | 0.426003 | 0.860332 | 0.769552 | 0.242604 | 0.368908 | 221 | 2,304 | 176.649 | 989 | 990 |

## Class Weight Comparison

- `none`: AP 0.427701 (+0.196267, +84.80% vs LR); FP 212, FN 2,293.
- `Balanced`: AP 0.410351 (+0.178917, +77.31% vs LR); FP 10,608, FN 1,009.
- `SqrtBalanced`: AP 0.413876 (+0.182443, +78.83% vs LR); FP 1,179, FN 1,900.

Weighting is interpreted through both ranking AP and threshold behavior; higher recall alone is not treated as sufficient.

## Best Candidate

Selected `cb-04-without-identity-none` with `none` weighting and identity features included=False. Selection used AP first, then the predeclared identity-stability tolerance.

## Improvement Over Logistic Regression

AP changed by +0.194569 (+84.07%). ROC-AUC changed by +0.066851 (+8.43%). At threshold 0.50, precision changed by +0.046218, recall by +0.171269, F1 by +0.239045, FP by +138, and FN by -521.

Threshold 0.50 is descriptive and is not a merchant approve/review/block recommendation.

## Failure-Slice Comparison

The slice definitions and supports were frozen from the Logistic Regression analysis before CatBoost was run.

| Slice | Fraud support | LR recall | CatBoost recall | Absolute improvement |
| --- | ---: | ---: | ---: | ---: |
| ProductCD=W | 1,485 | 0.000000 | 0.047811 | +0.047811 |
| TransactionAmt>=500 | 224 | 0.000000 | 0.258929 | +0.258929 |
| card4=discover | 122 | 0.000000 | 0.196721 | +0.196721 |
| ProductCD=S | 107 | 0.000000 | 0.168224 | +0.168224 |
| identity_available=False | 1,524 | 0.003281 | 0.056430 | +0.053150 |
| DeviceType=<MISSING> | 1,553 | 0.004507 | 0.059240 | +0.054733 |

## High-Value False Negatives

Count 2,304; amount total 378,333.733; median 85.796; P90 400.000; P95 554.000; maximum 3,822.950.
Compared with Logistic Regression, FN count changed by -521 and FN amount total by -103,613.999.

## Identity Ablation

With identity AP: 0.427701; without identity AP: 0.426003. AP loss without identity: +0.001697; ROC-AUC loss: -0.000354.
Decision: Identity-free candidate stayed within both predeclared stability tolerances; prefer simpler features.

Identity coverage changes materially over time. The selected model excludes identity; any later reintroduction would remain provisional and process-dependent.

## Feature Importance

- `C1`: 11.319853
- `TransactionAmt`: 10.963356
- `C5`: 10.718185
- `D2`: 9.962970
- `C2`: 9.566109
- `P_emaildomain`: 8.406483
- `C4`: 7.905678
- `D1`: 7.486585
- `D3`: 7.080571
- `ProductCD`: 6.720806

Native feature importance is associative, not causal. No semantic meaning is inferred for masked C* or D* fields, and SHAP was not run.

## Calibration Diagnostic

Validation Brier score: 0.024689. The reliability curve is descriptive; no calibration model was fitted.

## Limitations

- All results come from one chronological validation period, not the sealed held-out test.
- Early stopping and candidate selection use VALIDATION, so these are development metrics.
- Identity availability may encode source-system enrichment behavior.
- The unweighted full and identity-free models selected best iterations 998 and 989, close to the 1,000-tree ceiling; no iteration extension was tested in this controlled phase.
- No merchant thresholds, rules, costs, savings, or final dashboard metrics were produced.

## Recommendation for Next Phase

Freeze the selected CatBoost validation candidate and its feature decision before any later threshold, calibration, cost, or final held-out evaluation phase. Do not access the test partition yet.
