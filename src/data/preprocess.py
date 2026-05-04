"""Encode categoricals, scale numerics (train fit), stratified split, optional undersample."""

import argparse
import pickle

from typing import Any, cast

import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


from src.config import load_toml, resolve_path


def undersample_majority(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep all minority rows; sample majority to ratio * minority count."""
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        return X, y
    majority_label = int(cast(Any, class_counts.idxmax()))
    minority_label = int(cast(Any, class_counts.idxmin()))
    minority_count = int(cast(Any, class_counts.min()))
    majority_target_count = min(
        int(ratio * minority_count),
        int(cast(Any, class_counts.loc[majority_label])),
    )
    majority_rows: pd.Series = y.loc[y.eq(majority_label)]
    minority_rows: pd.Series = y.loc[y.eq(minority_label)]
    majority_index = majority_rows.index
    minority_index = minority_rows.index
    sampled_majority_index = X.loc[majority_index].sample(
        n=majority_target_count,
        random_state=random_state,
    ).index
    kept_index = minority_index.union(sampled_majority_index)
    return X.loc[kept_index], y.loc[kept_index]


def class_weights(y: pd.Series) -> dict[int, float]:
    class_counts = y.value_counts()
    total_rows = len(y)
    return {
        int(cast(Any, class_label)): total_rows / (len(class_counts) * class_count)
        for class_label, class_count in class_counts.items()
    }


def run_pipeline(config_path: str) -> None:
    config = load_toml(config_path)
    path_settings = config["paths"]
    feature_settings = config["features"]
    preprocessing_settings = config["preprocessing"]

    input_csv_path = resolve_path(path_settings["processed_merged_enriched"])
    if not input_csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {input_csv_path}. Run src.features.engineering first or point config to an existing CSV.",
        )

    merged_dataframe = pd.read_csv(input_csv_path, low_memory=False)
    target_column = str(feature_settings["target"])
    categorical_columns = list(feature_settings["categorical"])
    target_label = merged_dataframe[target_column]
    if isinstance(target_label, pd.DataFrame):
        target_label = target_label.iloc[:, 0] # make sure the target is of type series and not a dataframe
    feature_matrix = merged_dataframe.drop(columns=[target_column])

    encoders: dict[str, LabelEncoder] = {}
    for column_name in categorical_columns:
        label_encoder = LabelEncoder()
        feature_matrix[column_name] = label_encoder.fit_transform(feature_matrix[column_name].astype(str))
        encoders[column_name] = label_encoder

    feature_columns = list(feature_matrix.columns)
    numeric_columns = [column_name for column_name in feature_columns if column_name not in categorical_columns]

    test_size = float(preprocessing_settings["test_size"])
    validation_fraction = float(preprocessing_settings["val_fraction_of_remainder"])
    random_seed = int(preprocessing_settings["random_state"])

    X_temp, X_test, y_temp, y_test = train_test_split(
        feature_matrix,
        target_label,
        test_size=test_size,
        random_state=random_seed,
        stratify=target_label,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=validation_fraction,
        random_state=random_seed,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_val[numeric_columns] = scaler.transform(X_val[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

    undersample_ratio = float(preprocessing_settings["undersample_ratio"])
    X_train_balanced, y_train_balanced = undersample_majority(
        X_train,
        y_train,
        undersample_ratio,
        random_seed,
    )
    class_weight_mapping = class_weights(y_train_balanced)

    encoder_output_dir = resolve_path(path_settings["encoders_dir"])
    split_output_dir = resolve_path(path_settings["splits_dir"])
    encoder_output_dir.mkdir(parents=True, exist_ok=True)
    split_output_dir.mkdir(parents=True, exist_ok=True)

    for column_name, label_encoder in encoders.items():
        with (encoder_output_dir / f"le_{column_name}.pkl").open("wb") as output_file:
            pickle.dump(label_encoder, output_file)
    with (encoder_output_dir / "scaler.pkl").open("wb") as output_file:
        pickle.dump(scaler, output_file)
    with (encoder_output_dir / "class_weights.pkl").open("wb") as output_file:
        pickle.dump(class_weight_mapping, output_file)

    X_train_balanced.to_csv(split_output_dir / "X_train.csv", index=False)
    y_train_balanced.to_csv(split_output_dir / "y_train.csv", index=False)
    X_val.to_csv(split_output_dir / "X_val.csv", index=False)
    y_val.to_csv(split_output_dir / "y_val.csv", index=False)
    X_test.to_csv(split_output_dir / "X_test.csv", index=False)
    y_test.to_csv(split_output_dir / "y_test.csv", index=False)

    print(f"Splits -> {split_output_dir}")
    print(f"Train (balanced) {X_train_balanced.shape}, val {X_val.shape}, test {X_test.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.toml")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
