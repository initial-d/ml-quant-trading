# ---------------------------------------------------------------------------
# ML-Quant-Trading — convenience make targets
# ---------------------------------------------------------------------------
PY      ?= python
PIP     ?= $(PY) -m pip
PYTEST  ?= $(PY) -m pytest
RUFF    ?= $(PY) -m ruff
CONFIG  ?= configs/small.yaml

.PHONY: help install install-dev lint format test cov benchmark public-validation aggregate-validation audit-validation \
        gen-data features train portfolio backtest \
        paper clean clean-all

help:
	@echo "Targets:"
	@echo "  install       pip install -e ."
	@echo "  install-dev   pip install -e .[dev]"
	@echo "  lint          ruff check ."
	@echo "  format        ruff format ."
	@echo "  test          pytest"
	@echo "  cov           pytest with coverage"
	@echo "  benchmark     benchmark tensor factor primitives"
	@echo "  public-validation  run public-data validation benchmark"
	@echo "  aggregate-validation  aggregate public-data validation reports"
	@echo "  audit-validation  audit a public-data validation report"
	@echo "  gen-data      synthesise GBM-based OCHLV panel  (CONFIG=$(CONFIG))"
	@echo "  features      compute factor matrix"
	@echo "  train         train ML model"
	@echo "  portfolio     run cross-sectional Markowitz optimisation"
	@echo "  backtest      run vectorised backtest + metrics"
	@echo "  paper         end-to-end pipeline (small config)"
	@echo "  clean         remove caches and build artefacts"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e .[dev]

lint:
	$(RUFF) check src tests scripts

format:
	$(RUFF) format src tests

test:
	$(PYTEST)

cov:
	$(PYTEST) --cov=mlquant --cov-report=term-missing

benchmark:
	$(PY) scripts/benchmark_tensor_factors.py --device auto

public-validation:
	$(PY) scripts/public_data_validation.py --source synthetic --models equal_weight,momentum_20,alpha101_mean

aggregate-validation:
	$(PY) scripts/aggregate_validation_reports.py artifacts/public_data_validation

audit-validation:
	$(PY) scripts/audit_validation_report.py artifacts/public_data_validation/summary.json

gen-data:
	$(PY) -m mlquant.cli.main gen-data --config $(CONFIG)

features:
	$(PY) -m mlquant.cli.main features --config $(CONFIG)

train:
	$(PY) -m mlquant.cli.main train --config $(CONFIG)

portfolio:
	$(PY) -m mlquant.cli.main portfolio --config $(CONFIG)

backtest:
	$(PY) -m mlquant.cli.main backtest --config $(CONFIG)

paper: gen-data features train portfolio backtest

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-all: clean
	rm -rf artifacts checkpoints data/processed data/interim
