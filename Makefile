PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTHON_BOOTSTRAP ?= python3
PYTHONPATH := ml/src:services/api

.PHONY: install setup data-check eda train-baseline train-primary error-analysis evaluate benchmark seed-demo api web test lint typecheck docker-up

install: setup

setup:
	$(PYTHON_BOOTSTRAP) -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'
	$(PYTHON_BOOTSTRAP) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e './ml[test]' -e './services/api[test]'
	npm install

data-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/data_check.py

eda:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/run_eda.py

train-baseline:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/train_baseline.py

train-primary:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/train_primary.py

error-analysis:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/error_analysis.py

evaluate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/evaluate_final.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/render_readme_results.py

benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/benchmark_inference.py

seed-demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) ml/scripts/export_demo_cases.py --local-only

api:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m uvicorn app.main:app --reload --port 8000

web:
	npm run dev

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest ml/tests services/api/tests
	npm test

lint:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ruff check ml services/api
	npm run lint

typecheck:
	npx tsc --noEmit

docker-up:
	docker compose up --build
