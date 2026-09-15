PYTHON ?= python3
CONFIG ?= configs/mvp.yaml
COMPOSE ?= docker compose -f compose.rocm.yaml
REPORT_FONT_DIR ?= /usr/share/fonts/truetype/liberation
LEGACY_MODEL_RELEASE ?= https://github.com/yt22-orb/driver-drowsiness-rocm/releases/download/v0.1.0-model

.PHONY: install lint test smoke metadata-audit rocm-build rocm-check audit train evaluate export legacy-model report rig web-install web-dev web-build

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

legacy-model:
	mkdir -p web/public/models
	curl -fL -o web/public/models/drowsiness-mobilenet-v3-small.onnx $(LEGACY_MODEL_RELEASE)/drowsiness-mobilenet-v3-small.onnx
	curl -fL -o web/public/models/mobilenet-v3-small.metadata.json $(LEGACY_MODEL_RELEASE)/model-metadata.json

report:
	$(COMPOSE) run --rm -v $(REPORT_FONT_DIR):/report-fonts:ro trainer sh -lc "python -m pip install -q '.[report]' && python -m drowsiness.report --config $(CONFIG) --pdf-font-dir /report-fonts"

rig: rocm-check audit train evaluate export

web-install:
	npm --prefix web ci

web-dev:
	npm --prefix web run dev

web-build:
	npm --prefix web run build
