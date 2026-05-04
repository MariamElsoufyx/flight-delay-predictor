from src.config import load_toml, resolve_path
from src.validator import validate_dataset
import pandas as pd


if __name__ == "__main__":
    ## 0.0 load the config from the toml file
    config = load_toml("configs/config.toml")
    path_settings = config["paths"]
    validation_settings = config["validation"]

    raw_flights_path = resolve_path(path_settings["raw_flights"])
    raw_weather_path = resolve_path(path_settings["raw_weather"])
    raw_airports_path = resolve_path(path_settings["raw_airports"])
    reports_dir = resolve_path(path_settings["reports_dir"])

    validation_flight_dir = reports_dir / "validation_flight"
    validation_weather_dir = reports_dir / "validation_weather"
    validation_airports_dir = reports_dir / "validation_airports"
    missing_threshold = int(validation_settings["missing_threshold"])
    outlier_method = str(validation_settings["outlier_method"])

    ## 1.0 validate the flights dataset
    validate_dataset(
        file_path=str(raw_flights_path),
        output_folder=str(validation_flight_dir),
        target_column=None,          # target not yet created at raw stage
        id_columns=None,
        missing_threshold=missing_threshold,
        outlier_method=outlier_method,
    )

    ## 1.1 print the summary
    with (validation_flight_dir / "summary_report.txt").open("r", encoding="utf-8") as summary_file:
        print(summary_file.read())

    ## 1.2 show the missing columns and quality issues
    flight_missing_df = pd.read_csv(validation_flight_dir / "missing_values_report.csv")
    print("=== Missing Values (Flight Data) ===")
    print(flight_missing_df[flight_missing_df["Missing Count"] > 0].to_string(index=False))

    flight_issues_df = pd.read_csv(validation_flight_dir / "quality_issues_report.csv")
    print("=== Quality Issues (Flight Data) ===")
    print(flight_issues_df.to_string(index=False))

    # 1.3 show outlier report
    flight_outlier_df = pd.read_csv(validation_flight_dir / "outlier_report.csv")
    print("=== Outliers (Flight Data) ===")
    print(flight_outlier_df[flight_outlier_df["Outlier Count"] > 0].to_string(index=False))

    ## 2.0 validate the weather dataset
    validate_dataset(
        file_path=str(raw_weather_path),
        output_folder=str(validation_weather_dir),
        target_column=None,
        id_columns=None,
        missing_threshold=missing_threshold,
        outlier_method=outlier_method,
    )

    ## 2.1 print the summary
    with (validation_weather_dir / "summary_report.txt").open("r", encoding="utf-8") as summary_file:
        print(summary_file.read())

    ## 2.2 show quality issues
    weather_issues_df = pd.read_csv(validation_weather_dir / "quality_issues_report.csv")
    print("=== Quality Issues (Weather Data) ===")
    print(weather_issues_df.to_string(index=False))

    ## 3.0 validate the airports dataset
    validate_dataset(
        file_path=str(raw_airports_path),
        output_folder=str(validation_airports_dir),
        target_column=None,
        id_columns=None,
        missing_threshold=missing_threshold,
        outlier_method=outlier_method,
    )

    ## 3.1 print the summary
    with (validation_airports_dir / "summary_report.txt").open("r", encoding="utf-8") as summary_file:
        print(summary_file.read())

    ## 3.2 show the missing columns and quality issues
    airports_missing_df = pd.read_csv(validation_airports_dir / "missing_values_report.csv")
    print("=== Missing Values (Airports Data) ===")
    print(airports_missing_df[airports_missing_df["Missing Count"] > 0].to_string(index=False))

    airports_issues_df = pd.read_csv(validation_airports_dir / "quality_issues_report.csv")
    print("=== Quality Issues (Airports Data) ===")
    print(airports_issues_df.to_string(index=False))
    