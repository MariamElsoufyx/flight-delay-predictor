# Flight delay predictor

## Description

End-to-end workflow for predicting flight delays: **raw data validation**, **airport cleaning**, **flight and weather feature engineering**, **flight–weather merge** (calendar date + departure hour), **imputation**, **IQR outlier capping**, **preprocessing** (encoding, scaling, stratified splits, undersampling), optional **EDA reports**, and **classification models** under `src/models/`.

**`configs/config.toml`** contains the configurations.

## Setup

```bash
# Install all runtime + dev dependencies (pytest, Jupyter, python-dotenv, etc.)
poetry install --with dev

# Optional: copy environment template; run_pipeline loads .env when present
cp .env.example .env
# You can set FLIGHT_DELAY_CONFIG in .env to point at a non-default TOML path.
```

## Run

**Important:** Run scripts with **`poetry run python run_pipeline.py --steps train,classify`** (

**Full pipeline** (needs raw CSV paths defined in `configs/config.toml`):

```bash
poetry run python run_pipeline.py
```

Steps include, in order: `validate` → `clean-airports` → `flight-features` → `weather-features` → `merge` → `split` → `eda` → **`train`** → **`classify`**. To run the data pipeline only (skip model training and evaluation):

```bash
poetry run python run_pipeline.py --skip train,classify
```

**Dry run** (prints steps only; safe in CI without data):

```bash
poetry run python run_pipeline.py --dry-run
```

**Individual modules** (examples):

```bash
poetry run python -m src.data.validate --run-defaults
poetry run python -m src.data.preprocess
poetry run python -m src.features.engineering --pipeline all
poetry run python -m src.reports.eda --report flight
```

## Stakeholder dashboard (Streamlit)

Interactive view of **EDA charts**, **model comparison**, and **plain-language takeaways** (run from project root so paths resolve):

```bash
poetry run streamlit run src/dashboard/app.py
```

Use the sidebar to point at another `config.toml` or project folder if needed. After **train** and **classify**, the dashboard loads `outputs/reports/model_performance_summary.csv` automatically.

## Tests

```bash
poetry run pytest
# HTML coverage report
poetry run pytest --cov=src --cov-report=html
```

## Reproducibility

1. Check in **`poetry.lock`** with the repo (CI installs from it).
2. Place raw inputs where `configs/config.toml` points (`[files.raw]`).
3. Run `poetry install --with dev` then `poetry run python run_pipeline.py`.

## Layout (modular)

| Path | Role |
|------|------|
| `src/data/` | Validation, preprocessing, merge |
| `src/features/` | Flight / weather feature engineering |
| `src/reports/` | EDA figure generation |
| `src/dashboard/` | Streamlit stakeholder dashboard |
| `src/models/` | Training / classification helpers |
| `configs/config.toml` | Paths and hyperparameters |
| `tests/` | `pytest` + coverage |
| `.github/workflows/ci.yml` | CI on `main` pushes and PRs |
