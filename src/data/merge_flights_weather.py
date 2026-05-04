"""Merge flights + weather, impute missing weather, then cap numeric outliers."""

import numpy as np
import pandas as pd
from typing import Any, cast

from src.config import load_toml, resolve_path


def cap_iqr_outliers(
    dataframe: pd.DataFrame,
    column_name: str,
    iqr_multiplier: float = 1.5,
) -> tuple[int, float, float]:
    """Clip values outside IQR bounds and return capped count + bounds."""
    column_values = pd.Series(dataframe[column_name], copy=False)
    q1_raw = column_values.quantile(0.25)
    q3_raw = column_values.quantile(0.75)
    q1_value = float(np.asarray(q1_raw).squeeze())
    q3_value = float(np.asarray(q3_raw).squeeze())
    iqr_value = q3_value - q1_value
    if iqr_value == 0:
        return 0, q1_value, q3_value

    lower_bound = float(q1_value - iqr_multiplier * iqr_value)
    upper_bound = float(q3_value + iqr_multiplier * iqr_value)
    is_outlier = (dataframe[column_name] < lower_bound) | (dataframe[column_name] > upper_bound)
    capped_rows = int(is_outlier.sum())
    dataframe[column_name] = dataframe[column_name].clip(lower=lower_bound, upper=upper_bound)
    return capped_rows, lower_bound, upper_bound


def impute_weather_values(
    merged_dataframe: pd.DataFrame,
    weather_cols: list[str],
) -> None:
    """Fill NaNs using (origin, month) means, then (month), then global from observed joins."""
    has_observed_weather = merged_dataframe["precipitation"].notna()
    if not bool(has_observed_weather.any()):
        for column_name in weather_cols:
            merged_dataframe[column_name] = merged_dataframe[column_name].fillna(0.0)
        return

    origin_month_means = (
        merged_dataframe.loc[has_observed_weather]
        .groupby(["origin", "month"])[weather_cols]
        .mean()
    )
    month_means = merged_dataframe.loc[has_observed_weather].groupby("month")[weather_cols].mean()
    global_means_from_observed = merged_dataframe.loc[has_observed_weather, weather_cols].mean()

    for column_name in weather_cols:
        missing_mask = merged_dataframe[column_name].isna()
        missing_index = merged_dataframe.index[missing_mask]
        if len(missing_index) == 0:
            continue

        origin_month_index = pd.MultiIndex.from_frame(
            merged_dataframe.loc[missing_index, ["origin", "month"]],
        )
        merged_dataframe.loc[missing_index, column_name] = (
            origin_month_means[column_name].reindex(origin_month_index).values
        )

        missing_mask = merged_dataframe[column_name].isna()
        missing_index = merged_dataframe.index[missing_mask]
        if len(missing_index) > 0:
            merged_dataframe.loc[missing_index, column_name] = (
                merged_dataframe.loc[missing_index, "month"].map(month_means[column_name]).values
            )

        missing_mask = merged_dataframe[column_name].isna()
        global_fill_value_raw = global_means_from_observed[column_name]
        global_fill_value = 0.0 if pd.isna(global_fill_value_raw) else float(global_fill_value_raw)
        merged_dataframe.loc[missing_mask, column_name] = global_fill_value


def merge_flights_and_weather(
    flights_dataframe: pd.DataFrame,
    weather_dataframe: pd.DataFrame,
    weather_cols: list[str],
) -> pd.DataFrame:
    """Join flights with hourly weather and apply post-merge cleanup."""
    merged_dataframe = flights_dataframe.merge(
        weather_dataframe.rename(columns={"iata": "origin"}),
        left_on=["origin", "fl_date", "dep_hour"],
        right_on=["origin", "date", "hour"],
        how="left",
    )
    merged_dataframe = merged_dataframe.drop(columns=["date", "hour"])
    merged_dataframe["month"] = pd.to_datetime(merged_dataframe["fl_date"]).dt.month.astype(int)

    rows_with_weather = int(cast(Any, merged_dataframe["precipitation"].notna().sum()))
    weather_coverage_pct = 100 * float(cast(Any, merged_dataframe["precipitation"].notna().mean()))
    print("Rows with hourly weather:", rows_with_weather, f"({weather_coverage_pct:.2f}%)")

    impute_weather_values(merged_dataframe, weather_cols)
    merged_dataframe = merged_dataframe.drop(columns=["month"])

    columns_to_cap = ["distance", *weather_cols]
    print(f'{"Column":<22} {"Capped":>12} Bounds')
    print("-" * 56)
    for column_name in columns_to_cap:
        if column_name in merged_dataframe.columns:
            capped_count, lower_bound, upper_bound = cap_iqr_outliers(merged_dataframe, column_name)
            capped_pct = capped_count / len(merged_dataframe) * 100
            print(f"{column_name:<22} {capped_count:>8,} ({capped_pct:.2f}%) [{lower_bound:.2f}, {upper_bound:.2f}]")

    return merged_dataframe


def main() -> None:
    config = load_toml()
    path_settings = config["paths"]
    feature_settings = config["features"]
    weather_columns = list(feature_settings["weather_columns"])

    flights_input_path = resolve_path(path_settings["processed_flights"])
    weather_input_path = resolve_path(path_settings["processed_weather"])
    merged_output_path = resolve_path(path_settings["merged"])

    processed_flights = pd.read_csv(flights_input_path)
    processed_weather = pd.read_csv(weather_input_path)

    merged_dataset = merge_flights_and_weather(processed_flights, processed_weather, weather_columns)
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_dataset.to_csv(merged_output_path, index=False)
    print(f"Wrote {merged_output_path} shape={merged_dataset.shape}")
    print("Columns:", list(merged_dataset.columns))


if __name__ == "__main__":
    main()
