# Logistic Regression Baseline

## Objective

Establish a transparent validation-only fraud baseline and measure the incremental value of time, masked numeric, and identity feature groups. This is not the final production model.

## Training Data

- Frozen chronological TRAIN: 413,378 rows
- All preprocessing and Logistic Regression parameters were fit on TRAIN only.

## Validation Data

- Frozen chronological VALIDATION: 88,581 rows
- Validation class distribution was not sampled, stratified, or rebalanced.
- HELD-OUT TEST was not loaded and no test predictions were generated.

## Features Tested

- **A — Core:** `TransactionAmt`, `ProductCD`, `card4`, `card6`, `P_emaildomain`
- **B — Core + Time:** `TransactionAmt`, `ProductCD`, `card4`, `card6`, `P_emaildomain`, `TransactionDT`
- **C — Core + Masked Numeric:** `TransactionAmt`, `ProductCD`, `card4`, `card6`, `P_emaildomain`, `C1`, `C2`, `C3`, `C4`, `C5`, `D1`, `D2`, `D3`
- **D — Core + Identity:** `TransactionAmt`, `ProductCD`, `card4`, `card6`, `P_emaildomain`, `identity_available`, `DeviceType`, `DeviceInfo`, `R_emaildomain`
- **E — Conservative Combined:** `TransactionAmt`, `ProductCD`, `card4`, `card6`, `P_emaildomain`, `C1`, `C2`, `C3`, `C4`, `C5`, `D1`, `D2`, `D3`, `identity_available`, `DeviceType`, `DeviceInfo`, `R_emaildomain`
- **F — Combined + Time:** `TransactionAmt`, `ProductCD`, `card4`, `card6`, `P_emaildomain`, `C1`, `C2`, `C3`, `C4`, `C5`, `D1`, `D2`, `D3`, `identity_available`, `DeviceType`, `DeviceInfo`, `R_emaildomain`, `TransactionDT`

## Preprocessing

- Numeric: TRAIN-fitted median imputation followed by `StandardScaler`.
- Categorical: TRAIN-fitted constant `__MISSING__` imputation followed by `OneHotEncoder(handle_unknown="ignore")`.
- Raw `TransactionAmt` was used; no silent transformation or clipping was applied.
- No automatic missingness indicators were added beyond `identity_available`.
- TRAIN non-null cardinalities: `DeviceInfo`=1,546, `P_emaildomain`=59, `R_emaildomain`=60. The sparse encoded dimensionality remained reasonable, so no rare-category grouping was applied.
- Solver diagnostic: the initial SAGA run reached `max_iter=1000` for all seven fits and those provisional results were discarded. `newton-cholesky` was selected once, then the complete experiment set was rerun; all final convergence states and warnings are recorded below.

## Class Imbalance Strategy

Each of the six named feature sets was fit with `class_weight=balanced`. The best balanced feature set was then refit with `class_weight=None` for a direct comparison. No SMOTE, over-sampling, under-sampling, or validation rebalancing was used.

## Experiment Results

Average Precision (AP) is computed with `sklearn.metrics.average_precision_score`. Classification metrics use the fixed descriptive threshold 0.50.

| Experiment | Feature set | Weight | Precision | Recall | F1 | AP | ROC-AUC | FP | FN | Seconds | Converged |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lr-01-core-balanced | A — Core | balanced | 0.092450 | 0.598948 | 0.160176 | 0.135189 | 0.747247 | 17,886 | 1,220 | 0.401 | yes |
| lr-02-core_time-balanced | B — Core + Time | balanced | 0.077811 | 0.673570 | 0.139506 | 0.133143 | 0.747168 | 24,284 | 993 | 0.409 | yes |
| lr-03-core_masked-balanced | C — Core + Masked Numeric | balanced | 0.102821 | 0.626561 | 0.176653 | 0.183289 | 0.790911 | 16,631 | 1,136 | 1.048 | yes |
| lr-04-core_identity-balanced | D — Core + Identity | balanced | 0.102094 | 0.564103 | 0.172897 | 0.159544 | 0.741321 | 15,092 | 1,326 | 0.994 | yes |
| lr-05-conservative_combined-balanced | E — Conservative Combined | balanced | 0.113422 | 0.591716 | 0.190355 | 0.200328 | 0.783600 | 14,070 | 1,242 | 1.947 | yes |
| lr-06-conservative_combined_time-balanced | F — Combined + Time | balanced | 0.078538 | 0.729126 | 0.141802 | 0.200084 | 0.782435 | 26,023 | 824 | 2.024 | yes |
| lr-07-conservative_combined-none | E — Conservative Combined | none | 0.723333 | 0.071335 | 0.129862 | 0.231434 | 0.793480 | 83 | 2,825 | 2.100 | yes |

## Best Baseline

- Selected experiment: `lr-07-conservative_combined-none`
- Feature set: E — Conservative Combined
- Class weight: `none`
- Encoded features: 1,696
- Selection reason: Highest validation Average Precision; F1 and recall are deterministic tie-breakers. Accuracy was not used for selection, and threshold 0.50 was not optimized.
- At threshold 0.50 it trades recall for precision: recall 0.071335, precision 0.723333. This is not an operational threshold recommendation.

## TransactionDT Ablation

- Core + Time versus Core: precision -0.014639, recall +0.074622, F1 -0.020669, AP -0.002046, ROC-AUC -0.000079.
- Combined + Time versus Combined: precision -0.034883, recall +0.137410, F1 -0.048553, AP -0.000245, ROC-AUC -0.001166.
- `TransactionDT` is relative dataset position. Any gain is treated as validation evidence with temporal-generalization risk, not automatically as leakage or proof it belongs in the final model.

## Identity Ablation

- Core + Identity versus Core: precision +0.009644, recall -0.034845, F1 +0.012721, AP +0.024355, ROC-AUC -0.005926.
- Combined versus Core + Masked Numeric: precision +0.010600, recall -0.034845, F1 +0.013702, AP +0.017039, ROC-AUC -0.007311.
- Identity availability shifted materially between TRAIN and VALIDATION, so any improvement remains potentially process-dependent.

## Masked-Feature Ablation

Core + C1–C5 + D1–D3 versus Core: precision +0.010372, recall +0.027613, F1 +0.016477, AP +0.048100, ROC-AUC +0.043664.
Masked variables are reported only by source name; no business meaning is inferred.

## Validation Error Analysis

At threshold 0.50 the best baseline produced 83 false positives and 2,825 false negatives. Detailed supported slices are in `artifacts/reports/baseline_error_analysis.md`.

## High-Value False Negatives

- Count: 2,825
- Total TransactionAmt: 481,947.732
- Median TransactionAmt: 77.000
- P90 / P95: 445.000 / 653.590

## Calibration Diagnostic

Validation Brier score: 0.029640. The reliability plot is descriptive; no Platt or isotonic calibration was fitted.

## Coefficient Associations

Largest positive coefficients:

- `categorical__DeviceInfo_hi6210sft Build/MRA58K`: +5.897584
- `categorical__R_emaildomain_protonmail.com`: +4.251536
- `categorical__DeviceInfo_VS5012 Build/NRD90M`: +4.232173
- `categorical__DeviceInfo_SM-N920A Build/MMB29K`: +3.745887
- `categorical__DeviceInfo_KFFOWI Build/LVY48F`: +3.688314
- `categorical__DeviceInfo_SM-A510M Build/MMB29K`: +3.639732
- `categorical__DeviceInfo_Z835 Build/NMF26V`: +3.560803
- `categorical__DeviceInfo_TA-1039 Build/NMF26F`: +3.522460
- `categorical__DeviceInfo_SM-A510M Build/LMY47X`: +3.457925
- `categorical__DeviceInfo_SM-A300H Build/LRX22G`: +3.343234

Largest negative coefficients:

- `categorical__DeviceInfo_SM-G930V Build/NRD90M`: -1.847228
- `numeric__C4`: -1.808394
- `categorical__DeviceInfo_LG-D693n Build/LRX22G`: -1.761688
- `categorical__DeviceInfo_SM-A520F Build/NRD90M`: -1.759140
- `categorical__DeviceInfo_SM-G930F Build/NRD90M`: -1.684391
- `categorical__DeviceInfo_SAMSUNG-SGH-I337 Build/KOT49H`: -1.581840
- `categorical__DeviceInfo_rv:59.0`: -1.568518
- `categorical__DeviceInfo_PRA-LX3 Build/HUAWEIPRA-LX3`: -1.566937
- `categorical__R_emaildomain_msn.com`: -1.503447
- `categorical__DeviceInfo_SM-G935V Build/NRD90M`: -1.466791

Coefficient magnitude reflects model association after scaling and one-hot encoding, not causation.

## Limitations

- Results describe one later validation period and are not final held-out performance.
- Threshold 0.50 is a comparison convention, not an operational approve/review/block decision.
- Identity signals may encode enrichment-process behavior that changes over time.
- No merchant cost, rule, threshold, or money-saved claim was calculated.

## Next Experiment

Train one stronger gradient-boosted tree model on TRAIN and compare it against this frozen Logistic Regression validation baseline. That phase has not started.
