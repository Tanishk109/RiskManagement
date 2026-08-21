# MerchantShield

MerchantShield is a defense-only, cost-aware fraud decision engine for merchants. It converts a model risk score and validation-derived rules into one of three actions—`APPROVE`, `REVIEW`, or `BLOCK`—then records human review and measures the estimated merchant cost of that configuration.

Current evidence status: **Not evaluated yet.** The software path is implemented and tested with tiny fixtures, but the protected IEEE-CIS files are not present in this repository. No fixture result is reported as project ML performance.

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
  → local Parquet splits under data/processed
  → train-only preprocessing
  → Logistic Regression baseline vs XGBoost
  → validation model/feature/threshold/rule decisions
  → one final held-out test evaluation
  → FastAPI + PostgreSQL
  → Overview / Transactions / Review Queue / Cost Lab
```

PostgreSQL is the application source of truth for model metadata, final metrics, threshold and cost configurations, selected transaction predictions, explanations, rule hits, reviews, and optional cost-simulation history. The full IEEE-CIS dataset never enters PostgreSQL. The hosted frontend shows `Not evaluated yet` when a remote FastAPI deployment is not configured with real evidence.

## Real Dataset

The only performance dataset is the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) labeled training data:

- `data/raw/train_transaction.csv`
- `data/raw/train_identity.csv`

The loader performs a left join on `TransactionID`; transactions without an identity row remain valid data. Kaggle raw files remain under `data/raw/`; the chronological ML splits are Parquet files under `data/processed/`. Credentials, raw/processed rows, models, and prediction-row exports are gitignored and must stay local. See [data/README.md](data/README.md).

## Temporal Evaluation Strategy

Rows are sorted by `TransactionDT`, then split approximately:

- first 70%: training and train-only preprocessing fit;
- next 15%: model comparison, feature selection, calibration decisions, threshold tuning, cost search, and rule design;
- final 15%: held-out test used only after all decisions are frozen.

Automated tests verify chronological ordering and disjoint `TransactionID` sets. Random splitting is not the primary evaluation.

## Models Tested

The implemented comparison is intentionally narrow:

1. Logistic Regression with imputation, scaling, one-hot encoding, and balanced class weights.
2. XGBoost with train-derived imbalance weighting and two documented feature sets.

Masked IEEE-CIS fields such as `V17` or `C1` are never assigned invented business meanings. The pipeline does not use SMOTE by default and does not run a giant hyperparameter search.

## Actual Results

<!-- RESULTS:START -->
**Not evaluated yet.** Run the real-data pipeline before presenting precision, recall, F1, PR-AUC, cost, latency, or savings claims.
<!-- RESULTS:END -->

This section is updated from `artifacts/metrics/final_test_metrics.json` by `ml/scripts/render_readme_results.py`; it is not manually maintained in multiple places.

## Cost Model

For each transaction:

- `APPROVE` fraud incurs the configured fraud fraction of amount plus fixed chargeback cost.
- `BLOCK` legitimate traffic incurs configured lost margin plus fixed false-positive cost.
- `REVIEW` always incurs manual review cost, then residual fraud or false-positive cost based on reviewer-effectiveness assumptions.

The defaults in `ml/configs/cost_assumptions.yaml` are scenario placeholders, not industry facts. Cost Lab visually separates model-derived outcomes from merchant-configurable assumptions. Threshold search uses validation predictions only and reports the **lowest estimated cost under the currently selected merchant assumptions**, never a universal optimum.

## What Didn't Work

No real experiment has run in this checkout, so there is no evidence-backed failed-experiment claim yet. Validation experiments append to `artifacts/metrics/experiments.csv`, and modeling decisions belong in [docs/modeling-decisions.md](docs/modeling-decisions.md).

## Product Demo

The UI contains only four main sections:

- Overview: held-out provenance, metrics, decision flow, and readiness.
- Transactions: actual held-out labels, scores, rules, factors, and visible `MODEL ERROR` flags.
- Review Queue: persisted analyst decisions with mandatory notes; no automatic retraining.
- Cost Lab: dual thresholds and configurable merchant assumptions calculated over held-out predictions.

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

The API container applies Alembic migrations before startup. The generated OpenAPI docs are available at `http://localhost:8000/docs`. Key endpoints are `/health`, `/api/v1/model`, `/api/v1/metrics/summary`, `/api/v1/transactions`, `/api/v1/score`, `/api/v1/reviews`, and `/api/v1/cost/simulate`.

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
- The current checkout has no real dataset or final evidence artifacts, so the project is not yet definition-of-done for ML performance.
- The hosted adapter is stateless until `NEXT_PUBLIC_API_URL` points to a deployed FastAPI service; operational persistence belongs to that service's PostgreSQL database.

## Rejected Scope

We deliberately did not implement Kafka, Neo4j, GNNs, an LLM analyst chatbot, fraud-ring visualization, automatic retraining, or complex microservices. The build prioritizes a complete, measurable fraud-decision loop over superficial breadth.

## Future Work

After the four-module loop has genuine held-out evidence, possible future work includes graph-based abuse-ring detection, stream processing, production drift detection, multi-merchant models, automated rule backtesting, analyst copilots, active learning, and real chargeback feedback. None is presented as implemented.

## Safety and Data Governance

MerchantShield detects and mitigates fraud; it does not generate attacks or help evade detection. API inputs are validated, SQLAlchemy produces parameterized queries, credentials live only in ignored environment files, and protected dataset rows are not published automatically.
