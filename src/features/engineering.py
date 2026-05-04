"""Feature engineering for flight and weather raw datasets.

Flight pipeline  (build_flight_features):
  raw flight CSV  -->  clean_raw_flight_data (preprocessing)
                   -->  dep_hour, is_weekend, is_delayed
                   -->  rename / reorder columns
                   -->  data/processed/flight_features.csv

Weather pipeline (build_weather_features):
  raw weather CSV + city-to-IATA mapping
                   -->  merge, extract date/hour, rename columns
                   -->  hourly aggregation per (iata, date, hour)
                   -->  data/processed/weather_features.csv

Weather profile  (build_weather_profile):
  weather_features.csv
                   -->  aggregate to (origin, month, dep_hour)
                   -->  used by merge_flight_weather in preprocess.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

import pandas as pd

from src.data.preprocess import clean_raw_flight_data, load_toml_config


# ---------------------------------------------------------------------------
# Flight feature engineering
# ---------------------------------------------------------------------------

CITY_TO_IATA: dict[str, list[str]] = {
    "San Diego":    ["SAN"],
    "Philadelphia": ["PHL"],
    "San Antonio":  ["SAT"],
    "San Jose":     ["SJC"],
    "New York":     ["JFK", "LGA", "EWR"],
    "Houston":      ["IAH", "HOU"],
    "Dallas":       ["DFW", "DAL"],
    "Chicago":      ["ORD", "MDW"],
    "Los Angeles":  ["LAX"],
    "Phoenix":      ["PHX"],
}

FLIGHT_COLUMN_ORDER = [
    "airline",
    "origin",
    "dest",
    "distance",
    "dep_hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_delayed",
]


def _engineer_flight_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create dep_hour, is_weekend, is_delayed; rename carrier; reorder columns."""
    flight_cfg = config.get("flight_pipeline", {})
    delay_threshold_min = flight_cfg.get("delay_threshold_minutes", 15)
    weekend_days = flight_cfg.get("weekend_day_of_week_values", [6, 7])

    # dep_hour: scheduled departure hour extracted from HHMM integer
    if "crs_dep_time" in df.columns:
        df["dep_hour"] = (df["crs_dep_time"] // 100).astype(int)

    # is_weekend: 1 if Saturday (6) or Sunday (7) per BTS day-of-week coding
    if "day_of_week" in df.columns:
        df["is_weekend"] = df["day_of_week"].isin(weekend_days).astype(int)

    # is_delayed: 1 if arrival delay exceeds threshold
    if "arr_delay" in df.columns:
        df["is_delayed"] = (df["arr_delay"] > delay_threshold_min).astype(int)

    # Drop source columns consumed during engineering
    cols_to_drop_post = ["crs_dep_time", "crs_arr_time", "arr_delay"]
    existing = [c for c in cols_to_drop_post if c in df.columns]
    df = df.drop(columns=existing)

    # Rename BTS carrier code column to generic name
    if "op_unique_carrier" in df.columns:
        df = df.rename(columns={"op_unique_carrier": "airline"})

    # Keep only the expected feature columns in the defined order
    col_order = flight_cfg.get("output_column_order", FLIGHT_COLUMN_ORDER)
    available = [c for c in col_order if c in df.columns]
    df = df[available]

    return df


def build_flight_features(config: dict) -> None:
    """Run full flight feature-engineering pipeline and save output CSV."""
    raw_path = config["files"]["raw"]["flight_data"]
    output_path = config["files"]["processed"]["flight_features"]

    df = pd.read_csv(raw_path, low_memory=False)
    print(f"Loaded raw flight data: {df.shape}")

    df = clean_raw_flight_data(df, config)
    df = _engineer_flight_features(df, config)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved flight features: {output_path}  shape={df.shape}")


# ---------------------------------------------------------------------------
# Weather feature engineering
# ---------------------------------------------------------------------------

def _build_city_iata_map(config: dict) -> pd.DataFrame:
    """Expand city-to-IATA mapping dict into a flat DataFrame."""
    mapping = config.get("weather_pipeline", {}).get("city_to_iata", CITY_TO_IATA)
    rows = [{"Location": city, "iata": iata} for city, codes in mapping.items() for iata in codes]
    return pd.DataFrame(rows)


def build_weather_features(config: dict) -> None:
    """Run full weather feature-engineering pipeline and save output CSV."""
    raw_path = config["files"]["raw"]["weather_data"]
    output_path = config["files"]["processed"]["weather_features"]

    weather = pd.read_csv(raw_path)
    print(f"Loaded raw weather data: {weather.shape}")

    # Attach IATA codes via city-to-IATA mapping
    city_iata_df = _build_city_iata_map(config)
    weather = weather.merge(city_iata_df, on="Location", how="left")

    # Parse timestamp into date + hour
    weather["date"] = pd.to_datetime(weather["Date_Time"]).dt.date
    weather["hour"] = pd.to_datetime(weather["Date_Time"]).dt.hour

    # Rename to project schema
    weather = weather.rename(
        columns={
            "Precipitation_mm": "precipitation",
            "Temperature_C":    "temperature_c",
            "Humidity_pct":     "humidity_pct",
            "Wind_Speed_kmh":   "wind_speed_kmh",
        }
    )

    # Keep only the columns needed downstream
    weather = weather[["iata", "date", "hour", "precipitation", "temperature_c", "humidity_pct", "wind_speed_kmh"]]

    # Aggregate to one row per (airport, date, hour)
    weather_hourly = (
        weather.groupby(["iata", "date", "hour"])
        .agg(
            precipitation=("precipitation", "mean"),
            temperature_c=("temperature_c", "mean"),
            humidity_pct=("humidity_pct", "mean"),
            wind_speed_kmh=("wind_speed_kmh", "mean"),
        )
        .reset_index()
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    weather_hourly.to_csv(output_path, index=False)
    print(f"Saved weather features: {output_path}  shape={weather_hourly.shape}")


def build_weather_profile(weather_hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly weather to a (origin, month, dep_hour) mean profile.

    The flight dataset has month + dep_hour but no full date, so we collapse
    all daily readings into one representative row per (airport, month, hour).
    The result is renamed to match flight column names so it can be joined
    directly on ['origin', 'month', 'dep_hour'].
    """
    weather_hourly = weather_hourly.copy()
    weather_hourly["date"] = pd.to_datetime(weather_hourly["date"])
    weather_hourly["month"] = weather_hourly["date"].dt.month

    profile = (
        weather_hourly.groupby(["iata", "month", "hour"], as_index=False)
        .agg(
            precipitation=("precipitation", "mean"),
            temperature_c=("temperature_c", "mean"),
            humidity_pct=("humidity_pct", "mean"),
            wind_speed_kmh=("wind_speed_kmh", "mean"),
        )
        .rename(columns={"iata": "origin", "hour": "dep_hour"})
    )
    return profile


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run feature engineering pipelines.")
    parser.add_argument("--config", default="configs/config.toml", help="Path to TOML config file")
    parser.add_argument(
        "--pipeline",
        choices=["flight", "weather", "all"],
        default="all",
        help="Which pipeline to run (default: all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_toml_config(args.config)

    if args.pipeline in ("flight", "all"):
        build_flight_features(cfg)

    if args.pipeline in ("weather", "all"):
        build_weather_features(cfg)
