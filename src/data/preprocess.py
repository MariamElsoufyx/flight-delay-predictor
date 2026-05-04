"""Preprocessing pipeline for merged flight-weather features."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import tomllib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample


def load_toml_config(config_path: str | Path) -> dict:
    """Load TOML configuration into a dictionary."""
    with open(config_path, "rb") as file_obj:
        return tomllib.load(file_obj)


def ensure_parent_dirs(paths: list[str]) -> None:
    """Create parent folders for a list of file paths."""
    for file_path in paths:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def clean_airports_dataset(config: dict) -> None:
    """Apply airport string-cleaning logic migrated from data_validation notebook."""
    airports_input = config["files"]["raw"]["airports_data"]
    airports_output = config["files"]["processed"]["airports_clean"]

    airports = pd.read_csv(airports_input)
    str_cols = airports.select_dtypes(include=["object", "string"]).columns

    for col in str_cols:
        airports[col] = airports[col].astype("string").str.strip()

    if "AirportName" in airports.columns:
        airports["AirportName"] = airports["AirportName"].str.title()
    if "City_Name" in airports.columns:
        airports["City_Name"] = airports["City_Name"].str.title()
    if "IATA" in airports.columns:
        airports["IATA"] = airports["IATA"].str.upper()
    if "ICAO" in airports.columns:
        airports["ICAO"] = airports["ICAO"].str.upper()

    Path(airports_output).parent.mkdir(parents=True, exist_ok=True)
    airports.to_csv(airports_output, index=False)
    print(f"Saved cleaned airports to {airports_output}")


def clean_raw_flight_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop leakage/useless columns and remove cancelled/diverted flights.

    This is pure data cleaning — no new columns are created here.
    Feature engineering (dep_hour, is_weekend, is_delayed) lives in
    src/features/engineering.py.
    """
    flight_cfg = config.get("flight_pipeline", {})

    cols_to_drop = flight_cfg.get(
        "cols_to_drop",
        [
            "cancellation_code",  # 98.64% missing
            "year",               # constant value
            "day_of_month",       # not in feature set
            "fl_date",            # month + day_of_week already capture this
            "origin_city_name", "origin_state_nm",
            "dest_city_name", "dest_state_nm",
            "op_carrier_fl_num",  # flight number, not predictive
            # Operational columns unknown at booking time (data leakage)
            "dep_time", "wheels_off", "wheels_on",
            "taxi_out", "taxi_in", "arr_time",
            "actual_elapsed_time", "air_time",
            # Delay breakdown columns (only known after landing)
            "carrier_delay", "weather_delay", "nas_delay",
            "security_delay", "late_aircraft_delay",
            "dep_delay",  # known at departure, not at booking
        ],
    )

    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)

    # Remove flights that have no arrival delay to predict
    if "cancelled" in df.columns and "diverted" in df.columns:
        df = df[(df["cancelled"] == 0) & (df["diverted"] == 0)]
        df = df.drop(columns=["cancelled", "diverted"])

    df = df.reset_index(drop=True)
    print(f"clean_raw_flight_data: shape={df.shape}, columns={list(df.columns)}")
    return df


def _cap_outliers_iqr(df: pd.DataFrame, col: str, multiplier: float = 1.5) -> tuple[pd.DataFrame, int]:
    """Winsorise a single column using the IQR method. Returns modified df and count of capped values."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return df, 0
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    n_capped = int(((df[col] < lower) | (df[col] > upper)).sum())
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df, n_capped


def merge_flight_weather(config: dict) -> None:
    """Join flight features with weather profile, impute, cap outliers, and save merged CSV.

    Steps (from merging_pipeline notebook):
      1. Load flight_features.csv + weather_features.csv
      2. Build (origin, month, dep_hour) weather profile via engineering.build_weather_profile
      3. Left-join flights <- weather profile on ['origin', 'month', 'dep_hour']
      4. Impute missing weather values with global column means
      5. IQR-cap outliers on numerical columns
      6. Save merged_dataset.csv
    """
    # Import here to avoid circular import at module level
    from src.features.engineering import build_weather_profile

    files_cfg = config["files"]
    merge_cfg = config.get("merging", {})

    flight_path = files_cfg["processed"]["flight_features"]
    weather_path = files_cfg["processed"]["weather_features"]
    output_path = files_cfg["processed"]["merged_dataset"]

    flights = pd.read_csv(flight_path)
    weather_hourly = pd.read_csv(weather_path)
    print(f"Loaded flight features : {flights.shape}")
    print(f"Loaded weather features: {weather_hourly.shape}")

    # Build (origin, month, dep_hour) weather profile
    weather_profile = build_weather_profile(weather_hourly)
    print(f"Weather profile shape  : {weather_profile.shape}")

    # Left-join so every flight row is kept
    merged = flights.merge(weather_profile, on=["origin", "month", "dep_hour"], how="left")
    print(f"Merged shape           : {merged.shape}")

    # Impute missing weather values with global column means
    weather_cols = merge_cfg.get(
        "weather_cols",
        ["precipitation", "temperature_c", "humidity_pct", "wind_speed_kmh"],
    )
    for col in weather_cols:
        if col in merged.columns and merged[col].isna().any():
            global_mean = merged[col].mean()
            merged[col] = merged[col].fillna(global_mean)
            print(f"  Imputed {col:<20} with global mean={global_mean:.4f}")

    # IQR-cap outliers on numerical columns
    cols_to_cap = merge_cfg.get(
        "outlier_cap_cols",
        ["distance", "precipitation", "temperature_c", "humidity_pct", "wind_speed_kmh"],
    )
    iqr_multiplier = float(merge_cfg.get("iqr_multiplier", 1.5))
    for col in cols_to_cap:
        if col in merged.columns:
            merged, n_capped = _cap_outliers_iqr(merged, col, iqr_multiplier)
            pct = n_capped / len(merged) * 100
            print(f"  Capped {col:<22}: {n_capped:,} ({pct:.2f}%)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved merged dataset: {output_path}  shape={merged.shape}")


def preprocess_merged_dataset(config: dict) -> None:
    """Run preprocessing pipeline and persist artifacts/splits."""
    files_cfg = config["files"]
    prep_cfg = config["preprocessing"]

    merged_input = files_cfg["processed"]["merged_dataset"]
    split_cfg = files_cfg["splits"]
    encoder_cfg = files_cfg["encoders"]

    ensure_parent_dirs(
        [
            split_cfg["x_train"],
            split_cfg["y_train"],
            split_cfg["x_val"],
            split_cfg["y_val"],
            split_cfg["x_test"],
            split_cfg["y_test"],
            encoder_cfg["label_encoder_airline"],
            encoder_cfg["label_encoder_origin"],
            encoder_cfg["label_encoder_dest"],
            encoder_cfg["scaler"],
            encoder_cfg["class_weights"],
        ]
    )

    df = pd.read_csv(merged_input)
    target_col = prep_cfg.get("target_column", "is_delayed")

    # 1) Label-encode categorical columns
    cat_cols = prep_cfg.get("categorical_label_encode_cols", ["airline", "origin", "dest"])
    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    encoder_path_map = {
        "airline": encoder_cfg["label_encoder_airline"],
        "origin": encoder_cfg["label_encoder_origin"],
        "dest": encoder_cfg["label_encoder_dest"],
    }
    for col, le in encoders.items():
        if col in encoder_path_map:
            with open(encoder_path_map[col], "wb") as file_obj:
                pickle.dump(le, file_obj)
            print(f"Saved: {encoder_path_map[col]}")

    # 2) Scale selected numeric columns
    num_scale_cols = prep_cfg.get(
        "numeric_scale_cols",
        ["distance", "dep_hour", "precipitation", "temperature_c", "humidity_pct", "wind_speed_kmh"],
    )
    available_scale_cols = [col for col in num_scale_cols if col in df.columns]
    scaler = StandardScaler()
    df[available_scale_cols] = scaler.fit_transform(df[available_scale_cols])
    with open(encoder_cfg["scaler"], "wb") as file_obj:
        pickle.dump(scaler, file_obj)
    print(f"Saved: {encoder_cfg['scaler']}")

    # 3) Train/val/test split
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]

    test_size = prep_cfg.get("test_split", 0.15)
    random_state = prep_cfg.get("random_state", 42)
    use_stratify = prep_cfg.get("stratify", True)
    stratify_main = y if use_stratify else None

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_main,
    )

    val_from_remaining = prep_cfg.get("validation_from_remaining", 0.1765)
    stratify_temp = y_temp if use_stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_from_remaining,
        random_state=random_state,
        stratify=stratify_temp,
    )

    # 4) Undersample majority class in training split
    ratio = float(prep_cfg.get("undersampling_majority_to_minority_ratio", 2.0))
    train_data = X_train.copy()
    train_data[target_col] = y_train.values

    majority = train_data[train_data[target_col] == 0]
    minority = train_data[train_data[target_col] == 1]

    target_majority_size = int(len(minority) * ratio)
    majority_downsampled = resample(
        majority,
        replace=False,
        n_samples=target_majority_size,
        random_state=random_state,
    )
    train_balanced = pd.concat([majority_downsampled, minority]).sample(frac=1, random_state=random_state)
    X_train_bal = train_balanced[feature_cols]
    y_train_bal = train_balanced[target_col]

    # 5) Compute and persist class weights from original (pre-undersample) train split
    n_on_time = int((y_train == 0).sum())
    n_delayed = int((y_train == 1).sum())
    n_total = len(y_train)
    w_on_time = n_total / (2 * n_on_time)
    w_delayed = n_total / (2 * n_delayed)
    class_weights = {0: round(w_on_time, 4), 1: round(w_delayed, 4)}
    with open(encoder_cfg["class_weights"], "wb") as file_obj:
        pickle.dump(class_weights, file_obj)
    print(f"Saved: {encoder_cfg['class_weights']}")

    # 6) Save splits (balanced train, untouched val/test)
    X_train_bal.to_csv(split_cfg["x_train"], index=False)
    y_train_bal.to_frame(name=target_col).to_csv(split_cfg["y_train"], index=False)
    X_val.to_csv(split_cfg["x_val"], index=False)
    y_val.to_frame(name=target_col).to_csv(split_cfg["y_val"], index=False)
    X_test.to_csv(split_cfg["x_test"], index=False)
    y_test.to_frame(name=target_col).to_csv(split_cfg["y_test"], index=False)

    print(f"Saved X_train: {split_cfg['x_train']}  shape={X_train_bal.shape}")
    print(f"Saved y_train: {split_cfg['y_train']}  shape={y_train_bal.shape}")
    print(f"Saved X_val  : {split_cfg['x_val']}  shape={X_val.shape}")
    print(f"Saved y_val  : {split_cfg['y_val']}  shape={y_val.shape}")
    print(f"Saved X_test : {split_cfg['x_test']}  shape={X_test.shape}")
    print(f"Saved y_test : {split_cfg['y_test']}  shape={y_test.shape}")
    print(f"Class weights: {class_weights}")


def parse_args() -> argparse.Namespace:
    """Parse command line args for preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline steps.")
    parser.add_argument("--config", default="configs/config.toml", help="Path to TOML config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_toml_config(args.config)
    preprocess_merged_dataset(cfg)
