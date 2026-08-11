PYTHON ?= python3
CONFIG ?= configs/mvp.yaml
COMPOSE ?= docker compose -f compose.rocm.yaml

.PHONY: install lint test smoke metadata-audit rocm-build rocm-check audit train evaluate export rig web-install web-dev web-build

install:
	$(PYTHON) -m pip install -e '.[dev]'

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m drowsiness.smoke --output-dir artifacts/smoke

metadata-audit:
	$(PYTHON) -m drowsiness.audit --config $(CONFIG)

rocm-build:
	$(COMPOSE) build

rocm-check:
	$(COMPOSE) run --rm trainer python -m drowsiness.hardware --require-rocm --expected-gpu '7800 XT'

audit:
	$(COMPOSE) run --rm trainer python -m drowsiness.audit --config $(CONFIG) --full

train:
	$(COMPOSE) run --rm trainer python -m drowsiness.train --config $(CONFIG) --device cuda

evaluate:
	$(COMPOSE) run --rm trainer python -m drowsiness.evaluate --config $(CONFIG) --device cuda

export:
	$(COMPOSE) run --rm trainer python -m drowsiness.export --config $(CONFIG)

rig: rocm-check audit train evaluate export

web-install:
	npm --prefix web ci

web-dev:
	npm --prefix web run dev

web-build:
	npm --prefix web run build
