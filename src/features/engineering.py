"""Derive calendar / cyclical features from merged CSV for modeling-ready matrix."""

import numpy as np
import pandas as pd

from src.config import load_toml, resolve_path


def add_calendar_and_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out["fl_date"])
    out["month"] = d.dt.month.astype(int)
    out["day_of_week"] = (d.dt.dayofweek + 1).astype(int)
    out["is_weekend"] = out["day_of_week"].isin([6, 7]).astype(int)
    dh = out["dep_hour"].astype(float)
    # hour-of-day cyclic (0–23)
    out["dep_hour_sin"] = np.sin(2 * np.pi * dh / 24.0)
    out["dep_hour_cos"] = np.cos(2 * np.pi * dh / 24.0)
    # day-of-week cyclic (Mon=1 … Sun=7 mapped to angle)
    dow = out["day_of_week"].astype(float)
    out["dow_sin"] = np.sin(2 * np.pi * (dow - 1) / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * (dow - 1) / 7.0)

    out = out.drop(columns=["fl_date"])
    return out


def main() -> None:
    cfg = load_toml()
    paths = cfg["paths"]
    in_path = resolve_path(paths["merged"])
    out_path = resolve_path(paths["processed_merged_enriched"])
    raw = pd.read_csv(in_path, low_memory=False)
    enriched = add_calendar_and_cyclic(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    print(f"Wrote {out_path} shape={enriched.shape}")


if __name__ == "__main__":
    main()
