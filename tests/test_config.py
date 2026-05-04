"""Configuration file is valid TOML and contains expected sections."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_config_toml_loads(project_root: Path) -> None:
    path = project_root / "configs" / "config.toml"
    assert path.is_file(), f"Missing {path}"
    with path.open("rb") as fh:
        cfg = tomllib.load(fh)
    assert "files" in cfg
    assert "raw" in cfg["files"]
    assert "processed" in cfg["files"]
    assert cfg["files"]["raw"]["flight_data"].endswith(".csv")


def test_merge_section(project_root: Path) -> None:
    with (project_root / "configs" / "config.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    merging = cfg.get("merging", {})
    assert merging.get("merge_keys") == ["origin", "date", "dep_hour"]
    assert "weather_cols" in merging


def test_models_section(project_root: Path) -> None:
    with (project_root / "configs" / "config.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    models = cfg.get("models", {})
    assert "output_dir" in models
    assert "final_model" in models
