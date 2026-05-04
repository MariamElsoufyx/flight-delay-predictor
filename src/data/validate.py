"""Dataset validation utilities and CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd


def validate_dataset(
    file_path: str | Path,
    output_folder: str | Path,
    target_column: str | None = None,
    id_columns: list[str] | None = None,
    missing_threshold: float = 30.0,
    outlier_method: str = "iqr",
) -> None:
    """Run validation checks on a CSV dataset and save detailed reports."""
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_placeholders = ["", " ", "NA", "N/A", "null", "NULL", "?", "-", "Unknown", "unknown"]
    df = pd.read_csv(file_path, na_values=missing_placeholders, low_memory=False)

    basic_info = pd.DataFrame(
        {
            "Metric": ["Rows", "Columns"],
            "Value": [df.shape[0], df.shape[1]],
        }
    )
    basic_info.to_csv(output_dir / "basic_info.csv", index=False)

    dtype_report = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Non-Null Count": df.notnull().sum().values,
            "Missing Count": df.isnull().sum().values,
            "Unique Values": [df[col].nunique(dropna=False) for col in df.columns],
        }
    )
    dtype_report.to_csv(output_dir / "data_types_report.csv", index=False)

    missing_report = (
        pd.DataFrame(
            {
                "Column": df.columns,
                "Missing Count": df.isnull().sum().values,
                "Missing %": ((df.isnull().sum() / len(df)) * 100).round(2).values,
            }
        )
        .sort_values(by="Missing %", ascending=False)
        .reset_index(drop=True)
    )
    missing_report.to_csv(output_dir / "missing_values_report.csv", index=False)

    total_duplicate_rows = int(df.duplicated().sum())
    duplicate_report = pd.DataFrame(
        {
            "Metric": ["Duplicate Rows Count"],
            "Value": [total_duplicate_rows],
        }
    )
    duplicate_report.to_csv(output_dir / "duplicates_report.csv", index=False)

    duplicate_key_results: list[dict[str, int | str]] = []
    for col in id_columns or []:
        if col in df.columns:
            duplicate_key_results.append({"Column": col, "Duplicate Count": int(df[col].duplicated().sum())})
    if duplicate_key_results:
        pd.DataFrame(duplicate_key_results).to_csv(output_dir / "duplicate_key_columns_report.csv", index=False)

    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        num_summary = num_df.describe().T
        num_summary["median"] = num_df.median()
        num_summary["skewness"] = num_df.skew()
        num_summary["kurtosis"] = num_df.kurt()
        num_summary.to_csv(output_dir / "numerical_summary.csv")

    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    cat_summary_list = []
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False).reset_index()
        vc.columns = [col, "Count"]
        vc["Percentage %"] = (vc["Count"] / len(df) * 100).round(2)
        vc["Column"] = col
        cat_summary_list.append(vc)
    if cat_summary_list:
        pd.concat(cat_summary_list, ignore_index=True).to_csv(output_dir / "categorical_summary.csv", index=False)

    class_imbalance_note = "Not checked"
    if target_column and target_column in df.columns:
        class_report = df[target_column].value_counts(dropna=False).reset_index()
        class_report.columns = [target_column, "Count"]
        class_report["Percentage %"] = (class_report["Count"] / len(df) * 100).round(2)
        class_report.to_csv(output_dir / "class_distribution.csv", index=False)

        if len(class_report) >= 2:
            max_count = class_report["Count"].max()
            min_count = class_report["Count"].min()
            imbalance_ratio = round(max_count / min_count, 2) if min_count != 0 else np.inf
            pd.DataFrame({"Metric": ["Imbalance Ratio"], "Value": [imbalance_ratio]}).to_csv(
                output_dir / "class_imbalance_report.csv", index=False
            )

            if imbalance_ratio <= 3:
                class_imbalance_note = f"Mild imbalance (IR={imbalance_ratio})"
            elif imbalance_ratio < 9:
                class_imbalance_note = f"Moderate imbalance (IR={imbalance_ratio})"
            else:
                class_imbalance_note = f"Severe imbalance (IR={imbalance_ratio})"
        else:
            class_imbalance_note = "Target has only one class"

    one_value_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    high_missing_cols = missing_report[missing_report["Missing %"] > missing_threshold]["Column"].tolist()

    text_cols = df.select_dtypes(include=["object", "string"]).columns
    whitespace_cols: list[str] = []
    case_issue_cols: list[str] = []
    for col in text_cols:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        if series.str.startswith(" ").any() or series.str.endswith(" ").any():
            whitespace_cols.append(col)
        if len(set(series.unique())) != len(set(series.str.lower().unique())):
            case_issue_cols.append(col)

    outlier_results = []
    if not num_df.empty and outlier_method == "iqr":
        for col in num_df.columns:
            series = num_df[col].dropna()
            if len(series) == 0:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr != 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())
            else:
                outlier_count = 0
            outlier_results.append(
                {
                    "Column": col,
                    "Outlier Count": outlier_count,
                    "Outlier %": round((outlier_count / len(df)) * 100, 2),
                }
            )

    if outlier_results:
        outlier_report = pd.DataFrame(outlier_results).sort_values(by="Outlier %", ascending=False).reset_index(drop=True)
        outlier_report.to_csv(output_dir / "outlier_report.csv", index=False)
        high_outlier_cols = outlier_report[outlier_report["Outlier Count"] > 0]["Column"].tolist()
    else:
        high_outlier_cols = []

    quality_issues = pd.DataFrame(
        {
            "Issue Type": [
                "Columns with one unique value",
                f"Columns with >{missing_threshold}% missing",
                "Columns with leading/trailing spaces",
                "Columns with inconsistent casing",
                "Numeric columns with outliers",
            ],
            "Columns": [
                ", ".join(one_value_cols) if one_value_cols else "None",
                ", ".join(high_missing_cols) if high_missing_cols else "None",
                ", ".join(whitespace_cols) if whitespace_cols else "None",
                ", ".join(case_issue_cols) if case_issue_cols else "None",
                ", ".join(high_outlier_cols) if high_outlier_cols else "None",
            ],
        }
    )
    quality_issues.to_csv(output_dir / "quality_issues_report.csv", index=False)

    summary_lines = [
        "DATA VALIDATION SUMMARY",
        "========================",
        f"Dataset file: {file_path}",
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Duplicate rows: {total_duplicate_rows}",
        f"Columns with >{missing_threshold}% missing: {', '.join(high_missing_cols) if high_missing_cols else 'None'}",
        f"Columns with one unique value: {', '.join(one_value_cols) if one_value_cols else 'None'}",
        f"Columns with leading/trailing spaces: {', '.join(whitespace_cols) if whitespace_cols else 'None'}",
        f"Columns with inconsistent casing: {', '.join(case_issue_cols) if case_issue_cols else 'None'}",
        f"Numeric columns with outliers: {', '.join(high_outlier_cols) if high_outlier_cols else 'None'}",
        f"Class imbalance: {class_imbalance_note}",
    ]
    (output_dir / "summary_report.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Validation reports saved in folder: {output_dir}")


def load_toml_config(config_path: str | Path) -> dict:
    """Load TOML configuration into a dictionary."""
    with open(config_path, "rb") as file_obj:
        return tomllib.load(file_obj)


def run_default_validations(config_path: str | Path = "configs/config.toml") -> None:
    """Run validation for the default flight, weather, and airports raw datasets."""
    cfg = load_toml_config(config_path)
    raw_files = cfg.get("files", {}).get("raw", {})
    validation_cfg = cfg.get("validation", {})

    validate_dataset(
        file_path=raw_files["flight_data"],
        output_folder=validation_cfg["flight_output_dir"],
        target_column=None,
        id_columns=None,
        missing_threshold=validation_cfg.get("missing_threshold", 30),
        outlier_method=validation_cfg.get("outlier_method", "iqr"),
    )
    validate_dataset(
        file_path=raw_files["weather_data"],
        output_folder=validation_cfg["weather_output_dir"],
        target_column=None,
        id_columns=None,
        missing_threshold=validation_cfg.get("missing_threshold", 30),
        outlier_method=validation_cfg.get("outlier_method", "iqr"),
    )
    validate_dataset(
        file_path=raw_files["airports_data"],
        output_folder=validation_cfg["airports_output_dir"],
        target_column=None,
        id_columns=validation_cfg.get("airports_id_columns", ["IATA"]),
        missing_threshold=validation_cfg.get("missing_threshold", 30),
        outlier_method=validation_cfg.get("outlier_method", "iqr"),
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for running dataset validation."""
    parser = argparse.ArgumentParser(description="Run dataset validation checks.")
    parser.add_argument("--config", default="configs/config.toml", help="Path to TOML config file")
    parser.add_argument("--input", help="Path to input CSV file")
    parser.add_argument("--output", help="Path to output report folder")
    parser.add_argument("--target-column", default=None, help="Optional target column for class distribution checks")
    parser.add_argument("--id-columns", nargs="*", default=None, help="Optional identifier columns expected to be unique")
    parser.add_argument("--missing-threshold", type=float, default=None, help="High-missingness threshold percentage")
    parser.add_argument("--outlier-method", default=None, choices=["iqr"], help="Outlier detection method")
    parser.add_argument(
        "--run-defaults",
        action="store_true",
        help="Run the standard 3-dataset validation workflow from config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_toml_config(args.config)
    validation_cfg = cfg.get("validation", {})

    if args.run_defaults:
        run_default_validations(args.config)
    elif args.input and args.output:
        validate_dataset(
            file_path=args.input,
            output_folder=args.output,
            target_column=args.target_column,
            id_columns=args.id_columns,
            missing_threshold=args.missing_threshold
            if args.missing_threshold is not None
            else validation_cfg.get("missing_threshold", 30),
            outlier_method=args.outlier_method or validation_cfg.get("outlier_method", "iqr"),
        )
    else:
        raise SystemExit("Provide --run-defaults OR both --input and --output.")
