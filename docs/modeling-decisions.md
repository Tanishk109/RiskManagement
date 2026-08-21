# Modeling Decisions

## Current status

The real IEEE-CIS files are not present, so no model or metric decision has been made from fixtures. The implemented software is ready for the evidence sequence below.

## Observations from EDA

<!-- EDA:START -->
Not evaluated yet. `ml/scripts/run_eda.py` will generate observations from actual local data.
<!-- EDA:END -->

## Potential leakage

- `isFraud` and any future outcome are explicitly forbidden as features and rule conditions.
- All preprocessing is inside fitted sklearn pipelines and is fit on training rows only.
- The primary split is chronological by `TransactionDT`; train, validation, and test `TransactionID` sets must be disjoint.
- No sequential velocity feature is currently implemented. If one is added later, it must count only events strictly before the scored transaction and include leakage tests.

## Features rejected

- Fraud labels and future outcomes: rejected by policy and code.
- Invented semantic transformations of masked `C*`, `D*`, and `V*` fields: rejected.
- Proxy “customer” identities: not implemented because IEEE-CIS identifiers are masked and incomplete.

## Features retained

Candidate sets are declared in `ml/configs/feature_sets.yaml`. Retention is not final until real validation experiments compare interpretable and expanded sets.

## Model choice

<!-- PRIMARY_SELECTION:START -->
Logistic Regression is the transparent baseline. XGBoost is the single stronger tabular candidate. Model and feature-set selection use validation average precision; the held-out test is not read during selection.
<!-- PRIMARY_SELECTION:END -->

## Calibration

Diagnostics and Brier score are implemented. No Platt or isotonic calibrator is retained without a validation experiment demonstrating improved probability usefulness.

## Threshold choice

<!-- THRESHOLD_SELECTION:START -->
Two thresholds are required. The validation search uses the configured merchant assumptions and records the “lowest estimated cost under the currently selected merchant assumptions.” It is not described as universally optimal.
<!-- THRESHOLD_SELECTION:END -->

## What Didn't Work

No actual experiment has been run yet. Failed experiments will be recorded here only when supported by generated validation artifacts.
