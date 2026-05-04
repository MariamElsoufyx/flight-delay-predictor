"""Weather imputation logic used after merge (per-date then global fallback)."""

from __future__ import annotations

import pandas as pd


def _impute_weather_like_merge(
    merged: pd.DataFrame,
    weather_cols: list[str],
    impute_group: str = "date",
) -> pd.DataFrame:
    """Mirror merge_flight_weather imputation for isolated unit tests."""
    merged = merged.copy()
    if impute_group not in merged.columns:
        for col in weather_cols:
            if col in merged.columns and merged[col].isna().any():
                merged[col] = merged[col].fillna(merged[col].mean())
        return merged

    fallback_global = {col: merged[col].mean() for col in weather_cols if col in merged.columns}
    for col in weather_cols:
        if col not in merged.columns or not merged[col].isna().any():
            continue
        daily_mean = merged.groupby(impute_group, observed=True)[col].transform("mean")
        merged[col] = merged[col].fillna(daily_mean)
    for col in weather_cols:
        if col not in merged.columns:
            continue
        if merged[col].isna().any():
            merged[col] = merged[col].fillna(fallback_global[col])
    return merged


def test_per_date_imputation_fills_from_same_day() -> None:
    merged = pd.DataFrame(
        {
            "date": ["2024-01-01"] * 3 + ["2024-01-02"] * 2,
            "precipitation": [1.0, float("nan"), float("nan"), 10.0, float("nan")],
            "temperature_c": [20.0, float("nan"), 22.0, float("nan"), float("nan")],
            "humidity_pct": [50.0, 50.0, 50.0, 60.0, 60.0],
            "wind_speed_kmh": [5.0, float("nan"), 7.0, 8.0, 9.0],
        }
    )
    cols = ["precipitation", "temperature_c", "humidity_pct", "wind_speed_kmh"]
    out = _impute_weather_like_merge(merged, cols)
    assert out["precipitation"].isna().sum() == 0
    # 2024-01-01 observed mean for precip = 1.0
    assert out.loc[1, "precipitation"] == 1.0
    assert out.loc[2, "precipitation"] == 1.0
    # 2024-01-02 only 10.0 observed for precip
    assert out.loc[4, "precipitation"] == 10.0
    # temperature: day1 mean = (20+22)/2 = 21
    assert out.loc[1, "temperature_c"] == 21.0


def test_global_fallback_when_date_column_missing() -> None:
    merged = pd.DataFrame({"precipitation": [2.0, float("nan"), 4.0]})
    out = _impute_weather_like_merge(merged, ["precipitation"], impute_group="date")
    assert out["precipitation"].tolist() == [2.0, 3.0, 4.0]
