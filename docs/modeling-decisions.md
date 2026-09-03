# Modeling Decisions

## Current status

The official local IEEE-CIS labeled training files passed validation and EDA and were materialized into chronological train, validation, and held-out test partitions. The identity-free CatBoost candidate, its 13-feature schema, and the Scenario B operating point were frozen from validation, then evaluated once on the final temporal 15% without retraining. Held-out precision is 0.382498, recall 0.351606, F1 0.366402, and PR-AUC 0.382920 at the frozen 0.250 block threshold. No rule was enabled. These are final offline test results, not production guarantees.

## Real Dataset Validation

- `train_transaction.csv`: 590,540 unique transactions and 683,351,067 bytes.
- `train_identity.csv`: 144,233 unique identity rows and 26,529,680 bytes.
- The one-to-one left join preserves all 590,540 transaction rows. All 144,233 identity rows match a transaction, for 24.423917% overall identity coverage.
- Labels contain 20,663 fraud transactions and 569,877 legitimate transactions, giving 3.499001% fraud prevalence.
- `TransactionDT` ranges from 86,400 to 15,811,131. It is a relative time offset, not a calendar timestamp.
- `TransactionAmt` ranges from 0.251 to 31,937.391, with mean 135.0272, median 68.769, 95th percentile 445, and 99th percentile 1,104.
- Missingness was measured across all 434 columns in the transaction/identity left join; no sampling was used.

## Findings From IEEE-CIS EDA

- The target is strongly imbalanced: 3.499001% fraud. Accuracy will therefore be secondary when modeling eventually begins.
- Fraud transactions have a higher median amount (75.000 versus 68.500) and mean amount (149.2448 versus 134.5117), but the distributions overlap substantially. Amount alone is not a sufficient detector.
- Fraud prevalence varies from 2.030319% to 4.776534% across 20 equal-duration chronological buckets. This is direct evidence of temporal variation and supports chronological evaluation later.
- Identity-table availability differs substantially by actual label: 54.774234% of fraud rows have identity data versus 23.323454% of legitimate rows. This can be predictive, but it may also reflect the data-collection process.
- Twelve of 434 joined columns are more than 90% missing. The most sparse is `id_24` at 99.196159% missing; no joined column is completely missing.
- Minimum-support categorical analysis used at least 591 transactions per category. Examples of observed association include `ProductCD=C` at 11.687269% fraud versus `ProductCD=W` at 2.039939%, `card6=credit` at 6.678480% versus `card6=debit` at 2.426251%, and `DeviceType=mobile` at 10.166232% versus missing `DeviceType` at 2.101705%.
- These are univariate associations from the full labeled dataset, not causal findings, model metrics, or validation evidence.

## Potential Leakage Risks

- `isFraud` is the target and is forbidden from feature matrices and rule conditions.
- `TransactionID` is an identifier aligned with dataset order and `TransactionDT`; it must not be used as a model feature.
- Raw `TransactionDT` is available in the dataset, but it can encode dataset position and regime changes. Any use requires chronological validation, and it must not be converted to an invented calendar timestamp.
- The 31.450780 percentage-point identity-coverage gap between fraud and legitimate rows could let a model learn how the source system selected transactions for identity enrichment. Identity presence and identity fields must be retained only if the same information is available at the intended scoring time.
- Masked `C*`, `D*`, and `V*` meanings do not establish scoring-time availability. Each retained field needs a provenance/availability check before production use.
- No aggregate or velocity feature exists yet. Any later sequential feature must use only transactions strictly earlier than the row being scored and must be tested for future-row contamination.
- Categorical fraud rates in EDA use the target and are descriptive only. They must not be copied into model inputs as target encodings unless such encodings are fit exclusively on training rows with leakage-safe handling.

## Proposed Initial Features

These are experiment candidates, not selected features:

- Core transaction fields: `TransactionAmt`, `ProductCD`, `card4`, `card6`, `P_emaildomain`, and relative `TransactionDT`. The first three core numeric/categorical fields have 0% missingness; `card4`/`card6` are about 0.27% missing and `P_emaildomain` is 15.994852% missing.
- Conservative masked numeric group: `C1`–`C5` and `D1`–`D3`, with explicit imputation and missingness indicators where useful. `C1`–`C5` have 0% missingness, while `D2` and `D3` are 47.549192% and 44.514851% missing.
- Conditional identity group: identity-row availability, `DeviceType`, `DeviceInfo`, and `R_emaildomain`, only after scoring-time availability is confirmed. They are 75%–80% missing after the left join and require a separate ablation.
- Later expanded candidate group: selected `V*` fields only after availability, stability, memory, and incremental-value checks. Masked fields will not be assigned invented business meanings.

## Features Excluded Before Modeling

- `isFraud`, reviewer decisions, chargeback outcomes, and every other future outcome.
- `TransactionID` and any direct transformation that simply reconstructs row order.
- The Kaggle competition test files, because they do not contain labels and are not MerchantShield's held-out evaluation set.
- The twelve fields over 90% missing in the observed left join: `id_24`, `id_25`, `id_07`, `id_08`, `id_21`, `id_26`, `id_27`, `id_23`, `id_22`, `dist2`, `D7`, and `id_18`. They can be reconsidered only through a documented, leakage-safe ablation.
- Target-derived category rates or aggregates computed over validation/test/future rows.
- Invented semantic transformations of masked `C*`, `D*`, `V*`, or `id_*` fields.

## Questions Requiring Experiments

- Does an interpretable feature group outperform the numeric baseline on later chronological validation data?
- Does adding the conditional identity group improve validation PR-AUC and cost without depending on a data-collection artifact?
- Are identity coverage and categorical fraud-rate relationships stable across chronological periods?
- Does `log1p(TransactionAmt)` add value beyond raw amount, and how should extreme values be handled without discarding genuine fraud?
- Which missingness indicators add stable signal, and which only encode source-system behavior?
- Do selected `V*` fields add enough held-out value to justify lower interpretability and higher memory use?
- Which probability calibration method, if any, improves validation Brier score and merchant threshold usefulness?

## Temporal Evaluation Strategy

- All 590,540 labeled rows are sorted by `TransactionDT` ascending and `TransactionID` ascending using a stable sort. `TransactionID` is only a deterministic tie-breaker and audit field, never a model feature.
- Requested fractions are 70% train, 15% validation, and 15% held-out test. The dataset permits exact row shares at clean timestamp boundaries.
- Each cumulative target is moved to the nearest boundary between distinct `TransactionDT` values; a distance tie chooses the earlier cut. This produced strict separation: train maximum 10,437,996 is below validation minimum 10,438,003, and validation maximum 13,151,840 is below test minimum 13,151,880.
- No random split, stratification, label rebalancing, sampling, or forced fraud prevalence was used.
- Processed partitions retain the union of configured candidate features plus `identity_available`. The protected Parquet rows and split metadata remain ignored under `data/processed/ieee-cis/`.

## Train / Validation / Test Characteristics

| Partition | Rows | TransactionDT range | Fraud | Legitimate | Fraud prevalence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 413,378 | 86,400–10,437,996 | 14,538 | 398,840 | 3.516878% |
| Validation | 88,581 | 10,438,003–13,151,840 | 3,042 | 85,539 | 3.434145% |
| Held-out test | 88,581 | 13,151,880–15,811,131 | 3,083 | 85,498 | 3.480430% |

The three partitions account for every source row exactly once. Their `TransactionID` sets are pairwise disjoint. The small observed prevalence difference is retained as part of the temporal problem rather than corrected.

## Observed Temporal Distribution Shift

- `TransactionAmt` medians are similar at 68.95, 67.95, and 68.50 for train, validation, and test. The held-out test has a higher 99th percentile (1,226.894 versus 1,104 in train and validation), so amount-tail behavior still requires monitoring.
- `ProductCD=W` share changes from 72.067454% in train to 81.610052% in validation and 78.423138% in test, a 9.542598 percentage-point spread.
- `card6=debit` changes from 73.334091% to 78.727944% to 75.696820%, a 5.393853 percentage-point spread. The largest `card4` category-share spread is smaller at 1.207531 points.
- Missing `DeviceType` changes from 74.019662% to 82.805568% to 79.474154%, consistent with changing identity enrichment coverage.
- Using a predeclared substantial-missingness threshold of 5 percentage points, `D3`, `DeviceType`, `DeviceInfo`, `D2`, and `R_emaildomain` are flagged. Their maximum spreads are 9.080551, 8.785905, 8.313940, 8.075579, and 7.166564 points respectively.
- These are descriptive comparisons only. They are not target encodings, feature-selection results, or model metrics.

## Identity Availability Across Time

`identity_available` is true only when a transaction's `TransactionID` matched a row in `train_identity.csv`; it is not inferred from `DeviceType`, `DeviceInfo`, or another nullable identity value.

| Partition | Overall | Fraud | Legitimate |
| --- | ---: | ---: | ---: |
| Train | 26.588498% | 55.282707% | 25.542573% |
| Validation | 17.671961% | 49.901381% | 16.525795% |
| Held-out test | 21.074497% | 57.184560% | 19.772392% |

Fraud rows have materially higher identity availability in all three periods, so the full-dataset relationship is directionally stable. Overall coverage nevertheless moves by 8.916536 percentage points, and the fraud-versus-legitimate gap widens from 29.740134 points in train to 37.412168 points in held-out test. The indicator remains an experimental candidate requiring a later validation-only ablation and scoring-time provenance review.

## Evaluation Guardrails

- **TRAIN:** fit preprocessing and model parameters only.
- **VALIDATION:** compare models/features, make calibration decisions, tune thresholds and merchant-cost assumptions, and design rules.
- **HELD-OUT TEST:** final reporting only after every decision is frozen; never use it for selection or tuning.
- `FORBIDDEN_MODEL_FEATURES` rejects `isFraud`, `TransactionID`, and `actual_label` from model feature lists. `TransactionDT` is not automatically forbidden because the later experiment must compare with versus without it.
- All future preprocessing must be fit on train only. Any future aggregate or velocity feature must use strictly earlier transactions.
- The split phase itself generated no predictions, model metrics, thresholds, costs, rules, calibration objects, or model files; those artifacts were produced only in the later modeling phases.

## Model choice

Logistic Regression remains the frozen transparent baseline. CatBoost was selected on validation after a controlled native-categorical comparison on the same conservative features. Only the selected identity-free CatBoost candidate received the one final held-out evaluation; Logistic Regression was not evaluated on test.

## Calibration

Diagnostics and Brier score are implemented. No Platt or isotonic calibrator is retained without a validation experiment demonstrating improved probability usefulness.

## Threshold choice

Two thresholds implement `APPROVE` / `REVIEW` / `BLOCK`. A validation-only grid from 0.05–0.80 for review and 0.10–0.95 for block, in 0.025 steps with review strictly below block, was evaluated. Under the predeclared illustrative Scenario B assumptions, review 0.175 and block 0.250 was selected and frozen for final test reporting. It is still not a production or universal threshold.

## What Didn't Work

The initial SAGA configuration reached `max_iter=1000` and emitted a convergence warning in all seven fits. Those provisional results were discarded. A single solver diagnostic showed `newton-cholesky` converging on the largest design in five iterations without warnings; the complete experiment matrix was then rerun with that solver, and every final fit converged in three to six iterations.

## Logistic Regression Baseline

- Fitting source: frozen TRAIN only (413,378 rows).
- Evaluation source: frozen VALIDATION only (88,581 rows).
- Selected ranking baseline: `lr-07-conservative_combined-none`, using the 17-feature conservative combined set, no class weighting, and 1,696 encoded features.
- Selection used highest validation Average Precision (AP), with F1 and recall only as deterministic tie-breakers. Accuracy was not used.
- Selected validation metrics: precision 0.723333, recall 0.071335, F1 0.129862, AP 0.231434, ROC-AUC 0.793480, TP 217, FP 83, TN 85,456, and FN 2,825 at the fixed descriptive threshold 0.50.
- The best balanced experiment was the same feature set: precision 0.113422, recall 0.591716, F1 0.190355, AP 0.200328, and ROC-AUC 0.783600 at 0.50.
- The unweighted model's AP gain of 0.031106 comes with 0.520381 lower recall and 0.609912 higher precision at 0.50. This demonstrates why 0.50 is not an operational threshold recommendation.
- Brier score for the selected baseline is 0.029640. This is descriptive; no calibrator was fitted.

## Feature Ablation Results

All feature ablations below compare balanced Logistic Regression fits so class weighting is held constant.

| Comparison | Δ Precision | Δ Recall | Δ F1 | Δ AP | Δ ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core + Time minus Core | -0.014639 | +0.074622 | -0.020669 | -0.002046 | -0.000079 |
| Combined + Time minus Combined | -0.034883 | +0.137410 | -0.048553 | -0.000245 | -0.001166 |
| Core + Identity minus Core | +0.009644 | -0.034845 | +0.012721 | +0.024355 | -0.005926 |
| Combined minus Core + Masked | +0.010600 | -0.034845 | +0.013702 | +0.017039 | -0.007311 |
| Core + Masked minus Core | +0.010372 | +0.027613 | +0.016477 | +0.048100 | +0.043664 |

## TransactionDT Experiment

Adding raw `TransactionDT` increased recall at threshold 0.50 but reduced precision, F1, AP, and ROC-AUC in both controlled comparisons. It therefore does not enter the proposed next feature set. This is validation evidence, not a declaration that time is leakage: `TransactionDT` represents relative dataset position and carries temporal-generalization risk that should be revisited only with a clear time-feature hypothesis.

## Identity Availability Experiment

Identity features improved AP by 0.024355 over Core and 0.017039 when added to Core + Masked, while reducing recall and ROC-AUC in both comparisons. They also changed the threshold-0.50 operating behavior materially. Because identity coverage falls from 26.588498% in TRAIN to 17.671961% in VALIDATION, the gain appears useful for ranking but potentially dependent on the enrichment process. Retain the identity group for the next validation ablation, but require scoring-time provenance and compare against a masked-only alternative before freezing a final model.

TRAIN non-null categorical cardinalities were `DeviceInfo` 1,546, `P_emaildomain` 59, and `R_emaildomain` 60. The combined sparse design produced 1,696 encoded features, so no target encoding or rare-category grouping was needed.

## Masked Numeric Experiment

Adding `C1`–`C5` and `D1`–`D3` to Core improved every reported validation measure: precision by 0.010372, recall by 0.027613, F1 by 0.016477, AP by 0.048100, and ROC-AUC by 0.043664. This is the strongest clean feature-group gain in the baseline phase. The fields remain named only by their source identifiers; no business semantics are inferred.

## Baseline Failure Patterns

At threshold 0.50 the selected unweighted baseline misses 2,825 of 3,042 validation fraud rows. Those false negatives total 481,947.732 in `TransactionAmt`, with median 77.000, P90 445.000, P95 653.590, and maximum 5,191.000. Supported zero-recall slices include `ProductCD=W` (1,485 fraud rows), `ProductCD=S` (107), `card4=discover` (122), and the fixed `TransactionAmt >= 500` bucket (224). Identity-unavailable fraud recall is 0.003281 over 1,524 rows, while 98.795% of the 83 false positives have identity available. These are threshold-specific associations, not causal findings or rule proposals.

## Decisions for the Next Model

- Freeze this Logistic Regression result as a validation baseline only; do not expose it as final merchant performance.
- Train one gradient-boosted tree candidate on TRAIN and compare it on VALIDATION against AP 0.231434 and the full fixed-threshold error profile above.
- Start with the conservative combined feature set without raw `TransactionDT`; retain a controlled masked-only comparison to test whether identity gains survive the nonlinear model.
- Do not tune business thresholds, calculate merchant cost, fit calibration, design rules, or access the held-out test until the stronger-model and feature decisions are frozen.

## Gradient Boosting Experiment

Three main `CatBoostClassifier` experiments used the same 17 conservative features, shared parameters, native categorical handling, and validation `PRAUC:type=Classic` early stopping. Numerical NaN values were preserved; categorical missing values were converted consistently to `__MISSING__`. No scaling, one-hot encoding, external target encoding, `TransactionDT`, V features, SHAP, or second model family was used.

| Experiment | Weight | AP | ROC-AUC | Precision | Recall | F1 | FP | FN | Best iteration |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CB-01 | none | 0.427701 | 0.859978 | 0.779396 | 0.246220 | 0.374219 | 212 | 2,293 | 998 |
| CB-02 | Balanced | 0.410351 | 0.862914 | 0.160826 | 0.668310 | 0.259262 | 10,608 | 1,009 | 521 |
| CB-03 | SqrtBalanced | 0.413876 | 0.861801 | 0.492029 | 0.375411 | 0.425881 | 1,179 | 1,900 | 391 |

CB-01 had the highest AP. CB-02 and CB-03 improved recall at the fixed 0.50 threshold but reduced ranking AP; weighting therefore primarily changed the operating point rather than improving the primary ranking metric. CB-01 and the later identity-free ablation selected best iterations 998 and 989, close to the fixed 1,000-tree ceiling. No iteration extension was tested in this phase.

## Nonlinear Model vs Linear Baseline

The selected identity-free CatBoost candidate reached AP 0.426003 and ROC-AUC 0.860332, versus Logistic Regression AP 0.231434 and ROC-AUC 0.793480. Absolute gains are +0.194569 AP and +0.066851 ROC-AUC; relative gains are +84.07% and +8.43% respectively.

At threshold 0.50, CatBoost reached precision 0.769552, recall 0.242604, and F1 0.368908, with 221 FP and 2,304 FN. Relative to Logistic Regression this is +0.046218 precision, +0.171269 recall, +0.239045 F1, 138 additional FP, and 521 fewer FN. The precision-recall curve is also materially above the linear baseline across much of the recall range. This supports the hypothesis that the nonlinear model family captures useful structure absent from the linear baseline, but it does not identify or prove specific causal interactions.

## Class Weight Decision

Use no automatic class weighting for the selected validation candidate. Relative to CB-01, Balanced reduced AP by 0.017349 while increasing FP by 10,396 and reducing FN by 1,284 at 0.50. SqrtBalanced reduced AP by 0.013824 while increasing FP by 967 and reducing FN by 393. These may represent useful later operating points, but class weighting is not preferred for ranking and does not substitute for a future explicit threshold decision.

## Failure Slice Improvements

Using the exact frozen Logistic Regression slice definitions, the selected CatBoost model improved recall on all six requested slices:

| Slice | Support | LR recall | CatBoost recall | Improvement |
| --- | ---: | ---: | ---: | ---: |
| `ProductCD=W` | 1,485 | 0.000000 | 0.047811 | +0.047811 |
| `TransactionAmt>=500` | 224 | 0.000000 | 0.258929 | +0.258929 |
| `card4=discover` | 122 | 0.000000 | 0.196721 | +0.196721 |
| `ProductCD=S` | 107 | 0.000000 | 0.168224 | +0.168224 |
| `identity_available=False` | 1,524 | 0.003281 | 0.056430 | +0.053150 |
| Missing `DeviceType` | 1,553 | 0.004507 | 0.059240 | +0.054733 |

The improvements are meaningful but several slices remain weak, particularly `ProductCD=W` and identity-unavailable/missing-device fraud. The selected model still has 2,304 false negatives totaling 378,333.733 in `TransactionAmt`, which is 521 fewer cases and 103,613.999 less amount than the Logistic Regression baseline at the same descriptive threshold.

## Identity Feature Decision

The best main weighting strategy was rerun without `identity_available`, `DeviceType`, `DeviceInfo`, and `R_emaildomain`, using identical CatBoost parameters. AP changed from 0.427701 to 0.426003, a loss of 0.001697; ROC-AUC improved from 0.859978 to 0.860332. This stayed inside the predeclared maximum losses of 0.005 AP and 0.005 ROC-AUC, so the simpler identity-free model is preferred.

The identity-free model reduced `ProductCD=S` recall by 0.186916 and reduced recall modestly for the high-amount and Discover slices, while improving `ProductCD=W`, identity-unavailable, and missing-device recall. These slice tradeoffs are recorded rather than hidden. Excluding identity reduces dependency on the temporally shifting enrichment process; any reintroduction would require new evidence and scoring-time provenance.

## Selected Validation Candidate

- Model: `CatBoostClassifier`, version `catboost-validation-v1`.
- Experiment: `cb-04-without-identity-none`.
- Features: 13 raw fields—`TransactionAmt`, `ProductCD`, `card4`, `card6`, `P_emaildomain`, `C1`–`C5`, and `D1`–`D3`.
- Class weighting: none.
- Validation AP 0.426003; ROC-AUC 0.860332; precision 0.769552; recall 0.242604; F1 0.368908 at 0.50.
- Brier score: 0.024689; diagnostic only, with no calibration fitted.
- Top native importances are `C1`, `TransactionAmt`, `C5`, `D2`, and `C2`. Importance is associative, not causal, and masked fields are not assigned meanings.
- Status: selected on validation and evaluated once on the later held-out test without retraining. The validation metrics in this section remain development evidence; final metrics are documented separately below.

## Why Threshold 0.50 Is Not Operational

The selected CatBoost candidate's 0.50 cutoff is a descriptive binary classifier threshold, not a merchant policy. On the 88,581-row validation partition it blocks 959 rows (1.082625%), detects 738 of 3,042 fraud rows (24.260355%), and produces 221 false positives. It cannot represent the operational option to review an uncertain transaction.

Under the clearly hypothetical Scenario B inputs below, binary blocking at 0.50 has estimated total cost INR 448,171.18. The provisional three-way point has estimated total cost INR 389,847.95, 40.039448% row-based detection, and a 0.713471% review rate. These are **ESTIMATED BUSINESS COST UNDER USER-SUPPLIED ASSUMPTIONS**, not factual savings or held-out model performance.

## Cost Model Assumptions

Every input in this section is an **ILLUSTRATIVE MERCHANT ASSUMPTION**. None is claimed to be an industry standard.

| Scenario | Fraud loss fraction | Fixed fraud cost | Legitimate margin rate | FP fixed cost | Manual review cost | Review fraud catch | Review legitimate approval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: low margin / low review cost | 60% | INR 0 | 8% | INR 5 | INR 10 | 80% | 97% |
| B: moderate | 85% | INR 50 | 18% | INR 20 | INR 25 | 90% | 98% |
| C: high margin / high fraud cost | 100% | INR 150 | 30% | INR 50 | INR 60 | 92% | 97% |

An approved fraud incurs its amount times the fraud-loss fraction plus the explicitly configured fixed fraud cost; an approved legitimate transaction incurs zero modeled cost. A blocked legitimate transaction incurs amount times legitimate margin plus the fixed false-positive cost; a blocked fraud incurs zero modeled cost. Every review incurs the manual-review cost, plus expected residual fraud or false-positive cost according to the configured review effectiveness. This deliberately simplified model omits lifetime value, delayed fraud labels, reviewer queues, and recovery behavior.

## Three-Way Decision Design

For score `p`, `p >= block_threshold` produces `BLOCK`; otherwise `p >= review_threshold` produces `REVIEW`; all remaining rows receive `APPROVE`. The invariant is `0 <= review_threshold < block_threshold <= 1`, and exact equality belongs to the more restrictive decision band. The reusable cost and decision logic lives in the ML package so analysis and API behavior use one implementation.

The validation search stored all 5,850 evaluated scenario and one-at-a-time sensitivity rows in the machine-readable threshold grid. It also records overall and per-label decision volumes, block precision, row detection, FP/FN, fraud amount routing, expected amount capture, and the complete estimated cost decomposition.

## Validation Threshold Analysis

The saved `cb-04-without-identity-none` prediction artifact was used without retraining. Its 88,581 unique IDs, 3,042 fraud rows, complete `TransactionAmt` join, and fixed-0.50 AP 0.426003, ROC-AUC 0.860332, precision 0.769552, recall 0.242604, F1 0.368908, FP 221, FN 2,304, and Brier score 0.024689 were reproduced before threshold analysis.

| Scenario | Review | Block | Review rate | Block rate | Detected fraud | Amount capture | FP | FN | Estimated total cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.150 | 0.225 | 0.962960% | 2.161863% | 42.044707% | 42.483380% | 800 | 1,763 | INR 195,074.94 |
| B | 0.175 | 0.250 | 0.713471% | 1.960917% | 40.039448% | 41.162812% | 667 | 1,824 | INR 389,847.95 |
| C | 0.175 | 0.250 | 0.713471% | 1.960917% | 40.039448% | 41.262759% | 667 | 1,824 | INR 672,944.76 |

These are the lowest estimated-cost grid rows for their respective assumptions. Matching thresholds in Scenarios B and C do not imply universal optimality; the grid resolution and this one validation period limit the conclusion.

For Scenario B, approve-all costs an estimated INR 574,471.45, selected CatBoost binary blocking at 0.50 costs INR 448,171.18, the prior balanced CatBoost binary artifact at 0.50 costs INR 699,909.39, and three-way decisioning at 0.175/0.250 costs INR 389,847.95. The balanced artifact's 66.831032% row detection comes with 10,608 false positives and therefore is not lowest-cost under these assumptions. These comparisons are estimates, not realized savings.

One-at-a-time sensitivity materially moves the lowest-cost thresholds. With all other Scenario B inputs fixed, fraud-loss fraction 0.60/0.85/1.00 selects 0.175/0.250, 0.175/0.250, and 0.150/0.250. Margin 0.08/0.18/0.30 selects 0.150/0.175, 0.175/0.250, and 0.175/0.375. Manual-review cost INR 10/25/60 selects 0.050/0.450, 0.175/0.250, and 0.175/0.200. Threshold choice is therefore a business decision as well as an ML decision.

## Review Capacity Tradeoff

The lowest-cost Scenario B row reviews 632 transactions (0.713471%), so it is feasible under the predeclared unconstrained, 1%, 2%, and 5% review limits and remains the lowest-cost row under each. The capacity constraint is a maximum, not a target; the search does not force an exact queue share.

Different objectives lead elsewhere on the frontier. Highest detected recall under 1% review uses 0.125/0.150 and reaches 44.575937% detection at 0.685249% review with estimated cost INR 406,646.20. Highest detected recall under 2% uses 0.100/0.150 and reaches 49.441157% at 1.890925% review with estimated cost INR 413,727.26. Highest amount capture under 2% uses 0.100/0.125 and captures an expected 48.911773% of fraud amount at 1.205676% review. These configurations are frontier examples, not additional optima.

## Fraud Amount vs Fraud Count

The validation fraud rows total INR 496,907.588 in `TransactionAmt`. At the provisional point, 1,070 fraud rows totaling INR 182,192.308 are blocked, 148 totaling INR 24,832.033 are reviewed, and 1,824 totaling INR 289,883.247 remain approved. Scenario B's 90% assumed review catch rate yields 41.162812% expected amount capture versus 40.039448% row detection, showing why value-weighted and row-count measures must remain distinct.

## Provisional Operating Point

The continued-development default is Scenario B with review threshold 0.175 and block threshold 0.250. It is the lowest estimated-cost validation grid row feasible under the predeclared 2% review-capacity ceiling: 632 reviews (0.713471%), 1,737 blocks (1.960917%), 667 false positives, 1,824 fraud approvals, 40.039448% row detection, and 41.162812% expected fraud-amount capture. Its estimated decomposition is INR 340,451.48 fraud loss, INR 33,596.47 false-positive cost, and INR 15,800 manual-review cost, totaling INR 389,847.95 under Scenario B assumptions.

This operating point was provisional, validation-only, uncalibrated, and merchant-dependent during selection. The `make evaluate` invocation froze it with zero enabled rules before the held-out test was read. It remains an offline evaluation policy, not a universal production recommendation.

## Final Held-Out Evaluation

The saved TRAIN-fitted `catboost-validation-v1` bundle was evaluated once on the final 88,581 chronological rows. No fit, feature, calibration, threshold, cost-assumption, or rule change occurred during final evaluation. A durable access record prevents accidental reruns.

| Held-out metric | Result |
| --- | ---: |
| Fraud rows | 3,083 |
| Precision at block threshold | 0.382498 |
| Recall at block threshold | 0.351606 |
| F1 | 0.366402 |
| Average precision / PR-AUC | 0.382920 |
| ROC-AUC | 0.851372 |
| TP / FP / TN / FN | 1,084 / 1,750 / 83,748 / 1,999 |
| APPROVE / REVIEW / BLOCK | 84,693 / 1,054 / 2,834 |

Compared with validation at the same frozen policy, block recall stayed nearly flat (0.351742 to 0.351606), but block precision fell from 0.616005 to 0.382498. The review rate increased from 0.713471% to 1.189871%, block rate from 1.960917% to 3.199332%, and false positives from 667 to 1,750. PR-AUC fell from 0.426003 to 0.382920. This temporal degradation is retained as the honest final result and must not trigger post-test tuning.

Under the same explicitly illustrative Scenario B inputs, estimated total cost is INR 454,825.32, including INR 81,533.13 estimated false-positive cost and INR 26,350.00 manual-review cost. Expected fraud-amount capture is 36.206639%. These values are scenario estimates, not realized merchant savings.

## Remaining Failure Cases

At the provisional point, fraud routing in the frozen failure slices is:

| Slice | Fraud support | Approved | Reviewed | Blocked |
| --- | ---: | ---: | ---: | ---: |
| `ProductCD=W` | 1,485 | 1,237 | 72 | 176 |
| `TransactionAmt>=500` | 224 | 119 | 11 | 94 |
| `card4=discover` | 122 | 59 | 11 | 52 |
| `ProductCD=S` | 107 | 58 | 11 | 38 |

The largest approved fraud is TransactionID 3,455,844 at amount INR 3,822.95 with score 0.008379; the next three highest approved amounts are INR 3,260.05 and two rows at INR 2,963.95. All four are `ProductCD=W`. These validation residuals may motivate a later limited rule-candidate analysis, but no rule was created in this phase.
