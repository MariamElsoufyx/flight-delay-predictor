"""Feature-engineering helpers."""

from __future__ import annotations

import pandas as pd

from src.features.engineering import build_weather_profile


def test_build_weather_profile_renames_and_formats_date() -> None:
    hourly = pd.DataFrame(
        {
            "iata": ["DAL", "DAL"],
            "date": ["2024-01-01", "2024-01-02"],
            "hour": [0, 1],
            "precipitation": [2.0, 4.0],
            "temperature_c": [10.0, 12.0],
            "humidity_pct": [50.0, 60.0],
            "wind_speed_kmh": [5.0, 7.0],
        }
    )
    out = build_weather_profile(hourly)
    assert list(out.columns) == [
        "origin",
        "date",
        "dep_hour",
        "precipitation",
        "temperature_c",
        "humidity_pct",
        "wind_speed_kmh",
    ]
    assert out["origin"].tolist() == ["DAL", "DAL"]
    assert out["dep_hour"].tolist() == [0, 1]
    assert out["date"].tolist() == ["2024-01-01", "2024-01-02"]
    assert out["precipitation"].tolist() == [2.0, 4.0]


def test_build_weather_profile_single_row() -> None:
    hourly = pd.DataFrame(
        {
            "iata": ["JFK"],
            "date": ["2024-06-15"],
            "hour": [14],
            "precipitation": [0.1],
            "temperature_c": [22.0],
            "humidity_pct": [55.0],
            "wind_speed_kmh": [12.0],
        }
    )
    out = build_weather_profile(hourly)
    assert len(out) == 1
    assert out["date"].iloc[0] == "2024-06-15"
