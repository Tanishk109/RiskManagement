# MerchantShield

MerchantShield is a defense-only, cost-aware fraud decision engine for merchants. It converts a model risk score and validation-derived rules into one of three actions—`APPROVE`, `REVIEW`, or `BLOCK`—then records human review and measures the estimated merchant cost of that configuration.

Current evidence status: the official local IEEE-CIS labeled training files have passed validation and EDA, the chronological 70/15/15 partitions are frozen, and TRAIN-fitted Logistic Regression and CatBoost candidates have been compared on VALIDATION. Identity-free CatBoost is the selected validation candidate. Final held-out and merchant-facing performance remains **Not evaluated yet**; protected rows and model bundles remain local and ignored by git.

## Problem

Fraud loss, false declines, chargebacks, and manual review all cost money. A detector that maximizes a single classification metric can still make a poor merchant decision: aggressive blocking may catch more fraud while rejecting profitable legitimate payments, and broad review may overwhelm analysts.

MerchantShield makes the decision economics explicit and keeps the analyst in the loop.

## Why Accuracy Is Not Enough

Fraud is rare, so a model can achieve high accuracy while missing the cases that matter. The primary evidence is therefore:

- precision, recall, F1, and average precision / PR-AUC;
- false-positive and false-negative counts;
- decision volumes for approve, review, and block;
- transparent fraud-loss, false-positive, and review costs;
- actual model errors from a later-in-time held-out set.

## System Design

```text
IEEE-CIS labeled train files
  → left join identity on TransactionID
  → chronological 70/15/15 split
  → local Parquet splits under data/processed/ieee-cis
  → train-only preprocessing
  → Logistic Regression baseline vs CatBoost
  → validation model/feature/threshold/rule decisions
  → one final held-out test evaluation
  → FastAPI + PostgreSQL
  → Overview / Transactions / Review Queue / Cost Lab
```

PostgreSQL is the application source of truth for model metadata, final metrics, threshold and cost configurations, selected transaction predictions, explanations, rule hits, reviews, and optional cost-simulation history. The full IEEE-CIS dataset never enters PostgreSQL. The hosted frontend shows `Not evaluated yet` when a remote FastAPI deployment is not configured with real evidence.

## Real Dataset

The only performance dataset is the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) labeled training data:

- `data/raw/ieee-cis/train_transaction.csv`
- `data/raw/ieee-cis/train_identity.csv`

The loader performs a left join on `TransactionID`; transactions without an identity row remain valid data and receive `identity_available=false`. Kaggle raw files remain under `data/raw/`; the chronological ML splits are Parquet files under `data/processed/ieee-cis/`. Credentials, raw/processed rows, models, and prediction-row exports are gitignored and must stay local. See [data/README.md](data/README.md).

## Temporal Evaluation Strategy

Rows are sorted by `TransactionDT`, then split approximately:

- first 70%: training and train-only preprocessing fit;
- next 15%: model comparison, feature selection, calibration decisions, threshold tuning, cost search, and rule design;
- final 15%: held-out test used only after all decisions are frozen.

Automated tests verify chronological ordering and disjoint `TransactionID` sets. Random splitting is not the primary evaluation.

## Models

The comparison is intentionally narrow:

1. Logistic Regression with TRAIN-only imputation, scaling, one-hot encoding, and balanced/unweighted comparisons is complete on VALIDATION.
2. CatBoost with native categorical handling has been compared under none, Balanced, and SqrtBalanced weighting. The selected identity-free candidate remains validation-only.

Masked IEEE-CIS fields such as `V17` or `C1` are never assigned invented business meanings. The pipeline does not use SMOTE by default and does not run a giant hyperparameter search.

## Actual Results

<!-- RESULTS:START -->
**Not evaluated yet.** Run the real-data pipeline before presenting precision, recall, F1, PR-AUC, cost, latency, or savings claims.
<!-- RESULTS:END -->

This section is updated from `artifacts/metrics/final_test_metrics.json` by `ml/scripts/render_readme_results.py`; it is not manually maintained in multiple places.

Technical validation baseline: the selected unweighted conservative combined Logistic Regression reached AP 0.231434 and ROC-AUC 0.793480 on VALIDATION. At the descriptive 0.50 threshold it reached precision 0.723333 and recall 0.071335, exposing an unsuitable default-threshold tradeoff. These are model-development results, not final merchant-facing metrics.

Technical validation candidate: identity-free CatBoost reached AP 0.426003 and ROC-AUC 0.860332 on VALIDATION. At 0.50 it reached precision 0.769552 and recall 0.242604. This is an 84.07% relative AP improvement over the linear baseline, but it remains development evidence—not final held-out or merchant-facing performance.

## Cost Model

For each transaction:

- `APPROVE` fraud incurs the configured fraud fraction of amount plus fixed chargeback cost.
- `BLOCK` legitimate traffic incurs configured lost margin plus fixed false-positive cost.
- `REVIEW` always incurs manual review cost, then residual fraud or false-positive cost based on reviewer-effectiveness assumptions.

The defaults in `ml/configs/cost_assumptions.yaml` are scenario placeholders, not industry facts. Cost Lab visually separates model-derived outcomes from merchant-configurable assumptions. Threshold search uses validation predictions only and reports the **lowest estimated cost under the currently selected merchant assumptions**, never a universal optimum.

## What Didn't Work

The initial SAGA baseline run failed to converge within 1,000 iterations for all seven fits and was discarded. A documented switch to `newton-cholesky` produced converged final fits. The selected unweighted baseline also demonstrates that the conventional 0.50 threshold can produce high precision while missing most fraud; it is not an operational threshold.

## Product Demo

The UI contains only four main sections:

- Overview: real dataset, chronological split, Logistic Regression/CatBoost validation evidence, feature importance, and failure analysis, with final results kept separate and locked.
- Transactions: real validation labels, scores, selected input fields, backend filters, interesting cases, and visible `MODEL ERROR` flags.
- Review Queue: an honest locked state until operational thresholds, rules, governance, and final evaluation are frozen.
- Cost Lab: disabled controls and no monetary claims until operational simulation is valid.

When protected competition rows cannot be deployed, the public site remains in an honest unevaluated state. Synthetic manual-scoring inputs may be added later only if clearly labeled and only after a real model exists.

## Setup

Prerequisites: Node.js 22+, Python 3.11+, a PostgreSQL service (local Docker or managed), and authorized access to the Kaggle dataset.

```bash
cp .env.example .env
make setup
make db-up
make db-migrate
make data-check
make prepare-data
make eda
make train-baseline
make train-primary
make error-analysis
make evaluate
make db-sync
make test
make lint
make typecheck
```

Run the services separately:

```bash
make api   # http://localhost:8000
make web   # http://localhost:3000
```

Or use PostgreSQL, migrated API, and web together:

```bash
docker compose up --build
```

The API container applies Alembic migrations before startup. The generated OpenAPI docs are available at `http://localhost:8000/docs`. Artifact-backed dashboard endpoints are `/api/v1/project/status`, `/api/v1/model-comparison`, `/api/v1/model/feature-importance`, `/api/v1/validation/transactions`, and `/api/v1/validation/interesting-cases`. Operational endpoints remain available for scoring, rules, reviews, and cost configuration.

## Operational Database

`DATABASE_URL` identifies the PostgreSQL service. The checked-in default targets local Docker; standard `postgres://`, `postgresql://`, and `postgresql+psycopg://` URLs are normalized onto psycopg 3, so Neon or Supabase can be configured without code changes.

Alembic owns schema changes:

```bash
make db-migrate  # apply migrations
make db-check    # compare models with migration head on a running database
make db-sync     # copy final model metadata/metrics into runtime tables
```

The operational schema contains `transactions`, `prediction_reasons`, `rule_hits`, `review_cases`, `model_runs`, `threshold_configs`, `cost_configs`, and `cost_simulations`. JSONB is limited to variable model metadata such as the feature-name list and split descriptions; transaction features, rule hits, metrics, thresholds, and costs use typed relational columns.

## Repository Layout

```text
apps/web/                 frontend package boundary / Sites adapter
services/api/             FastAPI, SQLAlchemy, PostgreSQL, rules, reviews
ml/src/merchantshield_ml/ reusable modeling and cost package
ml/scripts/               real-data EDA, training, evaluation, seeding
ml/tests/                 correctness fixtures (never performance evidence)
rules/                    empty-until-evidenced merchant rule config
data/                     ignored raw CSV and processed Parquet boundaries
artifacts/                generated metrics/reports/figures/model metadata
docs/                     architecture, decisions, failures, demo guide
```

## Limitations

- The IEEE-CIS data is anonymized and represents a particular commerce environment and time period.
- Some fields have masked semantics, which limits business interpretation.
- Fraud patterns change over time; offline performance does not equal production performance.
- Cost assumptions vary by merchant and must be reviewed before decisions are operationalized.
- False positives can harm customer experience, and manual review adds operational cost.
- Real-data validation, EDA, chronological splitting, a Logistic Regression baseline, and CatBoost validation selection are complete locally, but there is no held-out performance evidence yet, so the project is not definition-of-done for ML performance.
- The hosted adapter is stateless until `NEXT_PUBLIC_API_URL` points to a deployed FastAPI service; operational persistence belongs to that service's PostgreSQL database.

## Rejected Scope

We deliberately did not implement Kafka, Neo4j, GNNs, an LLM analyst chatbot, fraud-ring visualization, automatic retraining, or complex microservices. The build prioritizes a complete, measurable fraud-decision loop over superficial breadth.

## Future Work

After the four-module loop has genuine held-out evidence, possible future work includes graph-based abuse-ring detection, stream processing, production drift detection, multi-merchant models, automated rule backtesting, analyst copilots, active learning, and real chargeback feedback. None is presented as implemented.

## Safety and Data Governance

MerchantShield detects and mitigates fraud; it does not generate attacks or help evade detection. API inputs are validated, SQLAlchemy produces parameterized queries, credentials live only in ignored environment files, and protected dataset rows are not published automatically.
