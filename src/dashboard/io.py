"""Load paths and tables for the dashboard (no Streamlit dependency — testable)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pandas as pd

# Human-readable titles for config keys under [files.reports]
EDA_REPORT_ITEMS: list[tuple[str, str]] = [
    ("How often flights are late vs on time", "eda_flight"),
    ("Typical weather patterns in the data", "eda_weather"),
    ("How many flights have detailed weather at the airport", "weather_coverage"),
    ("How features relate to each other and delays", "correlation_heatmap"),
    ("Delay rates under different weather conditions", "eda_weather_delays"),
    ("Weather on delayed flights compared to on-time flights", "eda_weather_boxplots"),
    ("Balancing the training data for fairer models", "class_balance_comparison"),
]


def load_toml_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("rb") as fh:
        return tomllib.load(fh)


def resolve_path(project_root: Path, relative: str) -> Path:
    return (project_root / relative).resolve()


def build_eda_gallery(cfg: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    """Return list of {title, path, exists} for each configured EDA asset."""
    reports = cfg.get("files", {}).get("reports", {})
    out: list[dict[str, Any]] = []
    for title, key in EDA_REPORT_ITEMS:
        rel = reports.get(key)
        if not rel:
            continue
        path = resolve_path(project_root, rel)
        out.append({"title": title, "key": key, "path": path, "exists": path.is_file()})
    return out


def load_model_performance_summary(cfg: dict[str, Any], project_root: Path) -> pd.DataFrame | None:
    reports = cfg.get("files", {}).get("reports", {})
    rel = reports.get("model_performance_summary", "outputs/reports/model_performance_summary.csv")
    path = resolve_path(project_root, rel)
    if not path.is_file():
        return None
    return pd.read_csv(path)


def load_dashboard_context(
    config_path: str | Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Bundle config, EDA file status, and optional model metrics table."""
    cfg = load_toml_config(config_path)
    root = project_root or Path.cwd()
    return {
        "config": cfg,
        "project_root": root,
        "config_path": Path(config_path).resolve(),
        "eda_items": build_eda_gallery(cfg, root),
        "perf_df": load_model_performance_summary(cfg, root),
        "reports_dir": resolve_path(project_root or Path.cwd(), cfg.get("paths", {}).get("reports_dir", "outputs/reports")),
    }
