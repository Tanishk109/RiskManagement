PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTHON_BOOTSTRAP ?= python3
PYTHONPATH := ml/src:services/api

ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: install setup db-up db-status db-logs db-shell db-migrate db-check db-sync data-check prepare-data eda train-baseline train-catboost train-primary threshold-analysis error-analysis evaluate-preflight evaluate benchmark seed-demo api web test lint typecheck docker-up

install: setup

setup:
	$(PYTHON_BOOTSTRAP) -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'
	$(PYTHON_BOOTSTRAP) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e './ml[analysis,test]' -e './services/api[test]'
	npm install

data-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/data_check.py

db-up:
	docker compose up -d postgres

db-status:
	docker compose ps

db-logs:
	docker compose logs postgres

db-shell:
	docker compose exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

db-migrate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m alembic -c services/api/alembic.ini upgrade head

db-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m alembic -c services/api/alembic.ini check

db-sync:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/sync_runtime_evidence.py

prepare-data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/prepare_data.py

eda:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/run_eda.py

train-baseline:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/train_baseline.py

train-catboost:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/train_catboost.py

train-primary: train-catboost

threshold-analysis:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/analyze_thresholds.py

error-analysis:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/error_analysis.py

evaluate-preflight:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/evaluate_final.py --preflight-only

evaluate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/evaluate_final.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/render_readme_results.py

benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/benchmark_inference.py

seed-demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/export_demo_cases.py --local-only

api: db-migrate
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m uvicorn app.main:app --reload --port 8000

web:
	npm run dev

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest --import-mode=importlib ml/tests services/api/tests
	npm test

lint:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ruff check ml services/api
	npm run lint

typecheck:
	npx tsc --noEmit

docker-up:
	docker compose up --build
