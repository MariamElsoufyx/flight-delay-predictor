"""Map weather metro locations to IATA hourly aggregates."""

from pathlib import Path
from typing import cast

import pandas as pd

from src.config import load_toml, resolve_path

_CITY_TO_IATA: dict[str, list[str]] = {
    "San Diego": ["SAN"],
    "Philadelphia": ["PHL"],
    "San Antonio": ["SAT"],
    "San Jose": ["SJC"],
    "New York": ["JFK", "LGA", "EWR"],
    "Houston": ["IAH", "HOU"],
    "Dallas": ["DFW", "DAL"],
    "Chicago": ["ORD", "MDW"],
    "Los Angeles": ["LAX"],
    "Phoenix": ["PHX"],
}


def build_city_iata_mapping() -> pd.DataFrame:
    mapping_rows: list[dict[str, str]] = []
    for city_name, airport_codes in _CITY_TO_IATA.items():
        for airport_code in airport_codes:
            mapping_rows.append({"Location": city_name, "iata": airport_code})
    return pd.DataFrame(mapping_rows)


def transform_weather(weather: pd.DataFrame) -> pd.DataFrame:
    city_iata_mapping = build_city_iata_mapping()
    merged_weather = weather.merge(city_iata_mapping, on="Location", how="inner")
    merged_weather["datetime"] = pd.to_datetime(merged_weather["Date_Time"])
    merged_weather["date"] = merged_weather["datetime"].dt.date.astype(str)
    merged_weather["hour"] = merged_weather["datetime"].dt.hour
    merged_weather = merged_weather.rename(
        columns={
            "Precipitation_mm": "precipitation",
            "Temperature_C": "temperature_c",
            "Humidity_pct": "humidity_pct",
            "Wind_Speed_kmh": "wind_speed_kmh",
        },
    )
    merged_weather = merged_weather[
        ["iata", "date", "hour", "precipitation", "temperature_c", "humidity_pct", "wind_speed_kmh"]
    ]
    hourly_aggregates = merged_weather.groupby(["iata", "date", "hour"], as_index=False).agg(
        precipitation=("precipitation", "mean"),
        temperature_c=("temperature_c", "mean"),
        humidity_pct=("humidity_pct", "mean"),
        wind_speed_kmh=("wind_speed_kmh", "mean"),
    )
    return cast(pd.DataFrame, hourly_aggregates)


def main() -> None:
    config = load_toml()
    path_settings = config["paths"]
    input_csv_path = resolve_path(path_settings["raw_weather"])
    output_csv_path = resolve_path(path_settings["processed_weather"])
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    raw_weather = pd.read_csv(input_csv_path)
    processed_weather = transform_weather(raw_weather)
    processed_weather.to_csv(output_csv_path, index=False)
    print(f"Wrote {output_csv_path} shape={processed_weather.shape}")


if __name__ == "__main__":
    main()
