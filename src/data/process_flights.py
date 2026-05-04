"""Clean raw DOT flight CSV and emit booking-time features."""

from pathlib import Path
from typing import cast

import pandas as pd

from src.config import load_toml, resolve_path

_COLUMNS_TO_DROP = [
    "cancellation_code",
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "origin_city_name",
    "origin_state_nm",
    "dest_city_name",
    "dest_state_nm",
    "op_carrier_fl_num",
    "dep_time",
    "wheels_off",
    "wheels_on",
    "taxi_out",
    "taxi_in",
    "arr_time",
    "actual_elapsed_time",
    "air_time",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
    "dep_delay",
    "crs_elapsed_time",
]


def load_raw(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)


def transform_flights(
    flights_dataframe: pd.DataFrame,
    delay_threshold_minutes: int,
) -> pd.DataFrame:
    required_columns = {"fl_date", "crs_dep_time", "arr_delay", "cancelled", "diverted", "op_unique_carrier"}
    missing_columns = sorted(required_columns - set(flights_dataframe.columns))
    if missing_columns:
        raise KeyError(f"Flight CSV missing required columns: {missing_columns}")

    flights_dataframe = flights_dataframe.drop(
        columns=[column_name for column_name in _COLUMNS_TO_DROP if column_name in flights_dataframe.columns],
    )

    flights_dataframe = cast(
        pd.DataFrame,
        flights_dataframe[(flights_dataframe["cancelled"] == 0) & (flights_dataframe["diverted"] == 0)],
    )
    flights_dataframe = flights_dataframe.drop(columns=["cancelled", "diverted"])

    flights_dataframe["fl_date"] = pd.to_datetime(flights_dataframe["fl_date"]).dt.date.astype(str)

    flights_dataframe["dep_hour"] = (flights_dataframe["crs_dep_time"] // 100).astype(int)
    flights_dataframe["is_delayed"] = (flights_dataframe["arr_delay"] > delay_threshold_minutes).astype(int)

    flights_dataframe = flights_dataframe.drop(
        columns=[
            column_name
            for column_name in ("crs_dep_time", "crs_arr_time", "arr_delay")
            if column_name in flights_dataframe.columns
        ],
    )

    flights_dataframe = flights_dataframe.rename(columns={"op_unique_carrier": "airline"})

    output_columns = ["fl_date", "airline", "origin", "dest", "distance", "dep_hour", "is_delayed"]
    return cast(pd.DataFrame, flights_dataframe[output_columns])



def main() -> None:
    config = load_toml()
    path_settings = config["paths"]
    feature_settings = config["features"]
    input_csv_path = resolve_path(path_settings["raw_flights"])
    output_csv_path = resolve_path(path_settings["processed_flights"])
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    raw_flights = load_raw(input_csv_path)
    processed_flights = transform_flights(raw_flights, int(feature_settings["delay_threshold_minutes"]))
    processed_flights.to_csv(output_csv_path, index=False)
    print(f"Wrote {output_csv_path} shape={processed_flights.shape}")


if __name__ == "__main__":
    main()
