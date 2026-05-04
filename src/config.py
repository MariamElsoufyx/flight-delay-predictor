"""Project TOML config loader."""

from pathlib import Path
from typing import Any

import tomllib


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return repo_root() / p


def load_toml(rel_path: str = "configs/config.toml") -> dict[str, Any]:
    path = resolve_path(rel_path)
    with path.open("rb") as f:
        return tomllib.load(f)
