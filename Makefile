.PHONY: test lint format install smoke

install:
	pip install -e ".[dev,gpu]"

test:
	pytest tests/test_metrics.py tests/test_pipeline_smoke.py

smoke:
	pytest tests/test_pipeline_smoke.py -v

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests
