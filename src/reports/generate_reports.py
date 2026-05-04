"""EDA figures on merged flight+weather CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import load_toml, resolve_path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def correlation_heatmap(df: pd.DataFrame, num_cols: list[str], out_path: Path) -> None:
    cols = [c for c in num_cols if c in df.columns]
    if len(cols) < 2:
        return
    corr: pd.DataFrame = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def delay_rate_by_bins(df: pd.DataFrame, out_path: Path) -> None:
    d = df.copy()
    cols = []
    if "precipitation" in d.columns:
        d["precip_bin"] = pd.qcut(d["precipitation"], q=5, duplicates="drop")
        cols.append("precip_bin")
    if "temperature_c" in d.columns:
        d["temp_bin"] = pd.qcut(d["temperature_c"], q=5, duplicates="drop")
        cols.append("temp_bin")
    if not cols:
        return
    n = len(cols)
    _, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        rate = pd.Series(d.groupby(col, observed=True)["is_delayed"].mean())
        rate.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(f"Delay rate by {col}")
        ax.set_ylabel("P(delay)")
        plt.sca(ax)
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def boxplots_vs_target(df: pd.DataFrame, weather_cols: list[str], out_path: Path) -> None:
    wcols = [c for c in weather_cols if c in df.columns]
    if not wcols:
        return
    n = len(wcols)
    _, axes = plt.subplots(2, max(2, (n + 1) // 2), figsize=(12, 8))
    axes = axes.flatten()
    for i, c in enumerate(wcols):
        sns.boxplot(data=df, x="is_delayed", y=c, ax=axes[i])
        axes[i].set_title(c)
    for j in range(len(wcols), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def class_balance_snap(y: pd.Series, out_path: Path) -> None:
    s = pd.Series(y)
    vc = s.value_counts()
    plt.figure(figsize=(5, 4))
    vc.plot(kind="bar", color=["seagreen", "coral"], edgecolor="black")
    plt.title("Target class balance (is_delayed)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def weather_coverage_chart(df: pd.DataFrame, weather_cols: list[str], out_path: Path) -> None:
    """Rough split: airports with variance in precipitation vs collapsed imputation-heavy."""
    wc = weather_cols[0] if weather_cols else "precipitation"
    if wc not in df.columns:
        return
    var_by_origin = pd.Series(df.groupby("origin")[wc].var()).fillna(0.0)
    has_signal = var_by_origin > 1e-6
    n_sig = df[df["origin"].isin(var_by_origin[has_signal].index)].shape[0]
    n_weak = len(df) - n_sig
    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Origins with variable\nweather merge", "Origins dominated by\nimputed profile"],
        [n_sig, n_weak],
        color=["steelblue", "lightgray"],
        edgecolor="black",
    )
    plt.ylabel("Flight rows")
    plt.title("Weather signal vs imputation-heavy rows (by origin precip variance)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    cfg = load_toml()
    paths = cfg["paths"]
    feats = cfg["features"]

    merged_path = resolve_path(paths["merged"])
    out_dir = resolve_path(paths["reports_dir"])
    _ensure_dir(out_dir)

    df = pd.read_csv(merged_path, low_memory=False)
    df["_temp_date"] = pd.to_datetime(df["fl_date"])

    target_name = str(feats["target"])
    base_num = list(feats["merged_numeric_before_engineering"])
    numeric_for_corr = base_num + [target_name]
    if feats.get("merged_numeric_engineered_extra"):
        for c in feats["merged_numeric_engineered_extra"]:
            if c == "month":
                numeric_for_corr.append("month")
            elif c == "day_of_week":
                numeric_for_corr.append("day_of_week")
            elif c == "is_weekend":
                numeric_for_corr.append("is_weekend")
    numeric_for_corr = list(dict.fromkeys(numeric_for_corr))
    df["month"] = df["_temp_date"].dt.month
    df["day_of_week"] = df["_temp_date"].dt.dayofweek + 1
    df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype(int)
    numeric_for_corr = [c for c in numeric_for_corr if c in df.columns]

    weather_cols = list(feats["weather_columns"])
    correlation_heatmap(df, numeric_for_corr, out_dir / "correlation_heatmap.png")
    delay_rate_by_bins(df, out_dir / "eda_delay_bins.png")
    boxplots_vs_target(df, weather_cols, out_dir / "eda_weather_delays.png")
    y_series = df[target_name].squeeze()
    if isinstance(y_series, pd.DataFrame):
        y_series = y_series.iloc[:, 0]
    if not isinstance(y_series, pd.Series):
        y_series = pd.Series([y_series], dtype=int)
    class_balance_snap(y_series, out_dir / "class_balance_comparison.png")
    weather_coverage_chart(df, weather_cols, out_dir / "weather_coverage.png")
    df.drop(columns=["_temp_date", "month", "day_of_week", "is_weekend"], inplace=True)
    print(f"EDA images written to {out_dir}")


if __name__ == "__main__":
    main()
