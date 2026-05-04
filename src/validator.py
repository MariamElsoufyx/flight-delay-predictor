import pandas as pd
import numpy as np
import os


def validate_dataset(
    file_path,
    output_folder="validation_output",
    target_column=None,
    id_columns=None,
    missing_threshold=30,
    outlier_method="iqr"
):
    """
    General dataset validation function.

    Parameters:
        file_path (str): path to dataset CSV file
        output_folder (str): folder to save reports
        target_column (str or None): target column for classification
        id_columns (list or None): columns that should be unique, e.g. ["customer_id"]
        missing_threshold (float): percentage threshold for high missingness
        outlier_method (str): "iqr" only for now
    """

    # ---------------------------
    # 1) Create output folder
    # ---------------------------
    os.makedirs(output_folder, exist_ok=True)

    # ---------------------------
    # 2) Load dataset
    # ---------------------------
    missing_placeholders = ["", " ", "NA", "N/A", "null", "NULL", "?", "-", "Unknown", "unknown"]
    df = pd.read_csv(file_path, na_values=missing_placeholders)

    original_df = df.copy()

    # ---------------------------
    # 3) Basic info
    # ---------------------------
    basic_info = pd.DataFrame({
        "Metric": ["Rows", "Columns"],
        "Value": [df.shape[0], df.shape[1]]
    })
    basic_info.to_csv(f"{output_folder}/basic_info.csv", index=False)

    # ---------------------------
    # 4) Data types report
    # ---------------------------
    dtype_report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null Count": df.notnull().sum().values,
        "Missing Count": df.isnull().sum().values,
        "Unique Values": [df[col].nunique(dropna=False) for col in df.columns]
    })
    dtype_report.to_csv(f"{output_folder}/data_types_report.csv", index=False)

    # ---------------------------
    # 5) Missing values report
    # ---------------------------
    missing_report = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum().values,
        "Missing %": ((df.isnull().sum() / len(df)) * 100).round(2).values
    }).sort_values(by="Missing %", ascending=False)

    missing_report.to_csv(f"{output_folder}/missing_values_report.csv", index=False)

    # ---------------------------
    # 6) Duplicate rows report
    # ---------------------------
    total_duplicate_rows = df.duplicated().sum()

    duplicate_report = pd.DataFrame({
        "Metric": ["Duplicate Rows Count"],
        "Value": [total_duplicate_rows]
    })

    # Duplicate check for key columns if provided
    duplicate_key_results = []
    if id_columns:
        for col in id_columns:
            if col in df.columns:
                dup_count = df[col].duplicated().sum()
                duplicate_key_results.append({
                    "Column": col,
                    "Duplicate Count": dup_count
                })

    duplicate_report.to_csv(f"{output_folder}/duplicates_report.csv", index=False)

    if duplicate_key_results:
        duplicate_key_report = pd.DataFrame(duplicate_key_results)
        duplicate_key_report.to_csv(f"{output_folder}/duplicate_key_columns_report.csv", index=False)

    # ---------------------------
    # 7) Numerical summary
    # ---------------------------
    num_df = df.select_dtypes(include=[np.number])

    if not num_df.empty:
        num_summary = num_df.describe().T
        num_summary["median"] = num_df.median()
        num_summary["skewness"] = num_df.skew()
        num_summary["kurtosis"] = num_df.kurt()
        num_summary.to_csv(f"{output_folder}/numerical_summary.csv")

    # ---------------------------
    # 8) Categorical summary
    # ---------------------------
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    cat_summary_list = []

    for col in cat_cols:
        vc = df[col].value_counts(dropna=False).reset_index()
        vc.columns = [col, "Count"]
        vc["Percentage %"] = (vc["Count"] / len(df) * 100).round(2)
        vc["Column"] = col
        cat_summary_list.append(vc)

    if cat_summary_list:
        cat_summary = pd.concat(cat_summary_list, ignore_index=True)
        cat_summary.to_csv(f"{output_folder}/categorical_summary.csv", index=False)

    # ---------------------------
    # 9) Class distribution
    # ---------------------------
    class_imbalance_note = "Not checked"
    if target_column is not None and target_column in df.columns:
        class_report = df[target_column].value_counts(dropna=False).reset_index()
        class_report.columns = [target_column, "Count"]
        class_report["Percentage %"] = (class_report["Count"] / len(df) * 100).round(2)
        class_report.to_csv(f"{output_folder}/class_distribution.csv", index=False)

        if len(class_report) >= 2:
            max_count = class_report["Count"].max()
            min_count = class_report["Count"].min()
            imbalance_ratio = round(max_count / min_count, 2) if min_count != 0 else np.inf

            imbalance_df = pd.DataFrame({
                "Metric": ["Imbalance Ratio"],
                "Value": [imbalance_ratio]
            })
            imbalance_df.to_csv(f"{output_folder}/class_imbalance_report.csv", index=False)

            if imbalance_ratio <= 3:
                class_imbalance_note = f"Mild imbalance (IR={imbalance_ratio})"
            elif imbalance_ratio < 9:
                class_imbalance_note = f"Moderate imbalance (IR={imbalance_ratio})"
            else:
                class_imbalance_note = f"Severe imbalance (IR={imbalance_ratio})"
        else:
            class_imbalance_note = "Target has only one class"

    # ---------------------------
    # 10) Quality issues
    # ---------------------------
    one_value_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    high_missing_cols = missing_report[missing_report["Missing %"] > missing_threshold]["Column"].tolist()

    text_cols = df.select_dtypes(include=["object"]).columns
    whitespace_cols = []
    case_issue_cols = []

    for col in text_cols:
        series = df[col].dropna().astype(str)

        if not series.empty:
            if series.str.startswith(" ").any() or series.str.endswith(" ").any():
                whitespace_cols.append(col)

            # Case issue means same word appears in multiple casing forms
            unique_original = set(series.unique())
            unique_lower = set(series.str.lower().unique())

            if len(unique_original) != len(unique_lower):
                case_issue_cols.append(col)

    # ---------------------------
    # 11) Outlier detection (numeric only)
    # ---------------------------
    outlier_results = []

    if not num_df.empty and outlier_method == "iqr":
        for col in num_df.columns:
            series = num_df[col].dropna()

            if len(series) > 0:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1

                if iqr != 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
                else:
                    outlier_count = 0

                outlier_results.append({
                    "Column": col,
                    "Outlier Count": int(outlier_count),
                    "Outlier %": round((outlier_count / len(df)) * 100, 2)
                })

    if outlier_results:
        outlier_report = pd.DataFrame(outlier_results).sort_values(by="Outlier %", ascending=False)
        outlier_report.to_csv(f"{output_folder}/outlier_report.csv", index=False)
        high_outlier_cols = outlier_report[outlier_report["Outlier Count"] > 0]["Column"].tolist()
    else:
        high_outlier_cols = []

    # ---------------------------
    # 12) Final quality issues report
    # ---------------------------
    quality_issues = pd.DataFrame({
        "Issue Type": [
            "Columns with one unique value",
            f"Columns with >{missing_threshold}% missing",
            "Columns with leading/trailing spaces",
            "Columns with inconsistent casing",
            "Numeric columns with outliers"
        ],
        "Columns": [
            ", ".join(one_value_cols) if one_value_cols else "None",
            ", ".join(high_missing_cols) if high_missing_cols else "None",
            ", ".join(whitespace_cols) if whitespace_cols else "None",
            ", ".join(case_issue_cols) if case_issue_cols else "None",
            ", ".join(high_outlier_cols) if high_outlier_cols else "None"
        ]
    })
    quality_issues.to_csv(f"{output_folder}/quality_issues_report.csv", index=False)

    # ---------------------------
    # 13) Human-readable text summary
    # ---------------------------
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
        f"Class imbalance: {class_imbalance_note}"
    ]

    with open(f"{output_folder}/summary_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Validation reports saved in folder: {output_folder}")


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    validate_dataset(
        file_path="../data/raw/flight_data_2024.csv",
        output_folder="../outputs/reports/flight_validation",
        target_column=None,
        id_columns=None,
        missing_threshold=30,
        outlier_method="iqr"
    )