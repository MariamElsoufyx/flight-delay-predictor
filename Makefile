# Flight delay predictor — common tasks (requires Poetry: https://python-poetry.org/)
.PHONY: install install-dev test test-verbose cov pipeline-dry-run pipeline clean

install:
	poetry install --no-interaction

install-dev:
	poetry install --no-interaction --with dev

test:
	poetry run pytest

test-verbose:
	poetry run pytest -v

cov:
	poetry run pytest --cov=src --cov-report=html
	@echo "Open htmlcov/index.html in a browser."

pipeline-dry-run:
	poetry run python run_pipeline.py --dry-run

pipeline:
	poetry run python run_pipeline.py

# Data + features + merge + split + EDA only (skip long model training)
pipeline-data:
	poetry run python run_pipeline.py --skip train,classify

clean:
	rm -rf .pytest_cache htmlcov .coverage coverage.xml __pycache__ .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
