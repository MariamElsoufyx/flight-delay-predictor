"""
Fetch real hourly weather observations for the 10 cities used by the
flight-delay-predictor project, in the exact schema produced by the
synthetic data file we are replacing.

Source: Open-Meteo Historical Weather API (ERA5 reanalysis, free, no key).
        https://open-meteo.com/en/docs/historical-weather-api

Output schema (matches data/raw/weather_data.csv):
    Location, Date_Time, Temperature_C, Humidity_pct, Precipitation_mm, Wind_Speed_kmh

Run:
    python scripts/fetch_real_weather.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import requests


# Major airport coordinates for each city in CITY_TO_IATA. Where multiple
# airports share a city (NYC, Houston, Dallas, Chicago) we use the busiest hub.
CITIES: dict[str, tuple[float, float]] = {
    "San Diego":    (32.7335, -117.1897),  # SAN
    "Philadelphia": (39.8729,  -75.2437),  # PHL
    "San Antonio":  (29.5337,  -98.4698),  # SAT
    "San Jose":     (37.3639, -121.9289),  # SJC
    "New York":     (40.6398,  -73.7789),  # JFK
    "Houston":      (29.9844,  -95.3414),  # IAH
    "Dallas":       (32.8998,  -97.0403),  # DFW
    "Chicago":      (41.9742,  -87.9073),  # ORD
    "Los Angeles":  (33.9416, -118.4085),  # LAX
    "Phoenix":      (33.4373, -112.0078),  # PHX
}

START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"
OUTPUT_PATH = Path("data/raw/weather_data.csv")
BACKUP_PATH = Path("data/raw/weather_data.synthetic.csv")


def fetch_city(city: str, lat: float, lon: float) -> pd.DataFrame:
    """One API call per city. Returns a DataFrame in the project schema."""
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "timezone": "America/New_York",
            "wind_speed_unit": "kmh",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
        },
        timeout=120,
    )
    resp.raise_for_status()
    h = resp.json()["hourly"]
    return pd.DataFrame({
        "Location":         city,
        "Date_Time":        h["time"],
        "Temperature_C":    h["temperature_2m"],
        "Humidity_pct":     h["relative_humidity_2m"],
        "Precipitation_mm": h["precipitation"],
        "Wind_Speed_kmh":   h["wind_speed_10m"],
    })


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Back up the existing (synthetic) file by COPY (rename can fail on Windows
    # if a handle is still open). Skip if a backup already exists.
    if OUTPUT_PATH.exists() and not BACKUP_PATH.exists():
        print(f"Backing up existing {OUTPUT_PATH} -> {BACKUP_PATH}")
        try:
            shutil.copy2(str(OUTPUT_PATH), str(BACKUP_PATH))
        except Exception as exc:  # noqa: BLE001
            print(f"   WARNING: backup failed ({exc}); proceeding without backup.")

    frames: list[pd.DataFrame] = []
    for i, (city, (lat, lon)) in enumerate(CITIES.items(), start=1):
        print(f"[{i:>2}/{len(CITIES)}] Fetching {city:<14}  ({lat:.4f}, {lon:.4f}) ...", flush=True)
        try:
            df = fetch_city(city, lat, lon)
        except Exception as exc:  # noqa: BLE001
            print(f"   FAILED: {exc}", file=sys.stderr)
            return 1
        print(f"          got {len(df):>6,} hourly rows  "
              f"(temp range {df['Temperature_C'].min():.1f}..{df['Temperature_C'].max():.1f} C)")
        frames.append(df)
        time.sleep(0.5)  # be polite to the public API

    # Write to a temp path then atomically replace, to avoid leaving the
    # destination half-written if something dies mid-write.
    out = pd.concat(frames, ignore_index=True)
    tmp_path = OUTPUT_PATH.with_suffix(".csv.tmp")
    out.to_csv(tmp_path, index=False)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    tmp_path.rename(OUTPUT_PATH)
    print(f"\nSaved {len(out):,} rows -> {OUTPUT_PATH}")
    print("\nPer-city sanity check (real weather should differ by city/season):")
    out["month"] = pd.to_datetime(out["Date_Time"]).dt.month
    print(out.groupby(["Location", "month"])["Temperature_C"].mean().round(2).unstack().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
