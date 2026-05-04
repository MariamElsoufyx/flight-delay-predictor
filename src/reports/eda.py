"""EDA report generation for the flight-delay-predictor project.

Produces the following PNG reports under outputs/reports/:

  eda_flight.png              -- class distribution, delay rates by airline /
                                 month / day-of-week / hour / weekend
  eda_weather.png             -- raw weather variable distributions
  weather_coverage.png        -- flight volume: weather-covered vs other airports
  correlation_heatmap.png     -- feature correlation matrix on merged dataset
  eda_weather_delays.png      -- delay rate bucketed by weather condition level
  eda_weather_boxplots.png    -- weather feature distributions split by delay class
  class_balance_comparison.png-- class counts before and after undersampling

Run all plots:
    python -m src.reports.eda

Run a specific report:
    python -m src.reports.eda --report flight
    python -m src.reports.eda --report weather
    python -m src.reports.eda --report coverage
    python -m src.reports.eda --report correlation
    python -m src.reports.eda --report weather-delays
    python -m src.reports.eda --report weather-boxplots
    python -m src.reports.eda --report class-balance
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # non-interactive backend; safe for scripts and CI


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_toml(config_path: str) -> dict:
    with open(config_path, "rb") as fh:
        return tomllib.load(fh)


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, path: str, *, dpi: int = 150) -> None:
    _ensure_dir(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _safe_qcut(series: pd.Series, q: int, labels: list[str]) -> pd.Series:
    """pd.qcut that handles duplicate bin edges (common with imputed means)."""
    _, edges = pd.qcut(series, q=q, duplicates="drop", retbins=True)
    n_bins = len(edges) - 1
    return pd.qcut(series, q=q, labels=labels[:n_bins], duplicates="drop")


# ── individual report functions ───────────────────────────────────────────────

def plot_flight_eda(config: dict) -> None:
    """6-panel flight EDA: class distribution + delay rates by various features."""
    flight_path = config["files"]["processed"]["flight_features"]
    out_path = config["files"]["reports"]["eda_flight"]

    df = pd.read_csv(flight_path)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Flight Delay EDA", fontsize=18, fontweight="bold")

    # 1 — class distribution
    ax = axes[0, 0]
    df["is_delayed"].value_counts().plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="black")
    ax.set_title("Class Distribution")
    ax.set_xticklabels(["On-Time (0)", "Delayed (1)"], rotation=0)
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height() / len(df) * 100:.1f}%",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom",
        )

    # 2 — delay rate by airline
    ax = axes[0, 1]
    airline_col = "airline" if "airline" in df.columns else "op_unique_carrier"
    airline_delay = df.groupby(airline_col)["is_delayed"].mean().sort_values(ascending=False) * 100
    airline_delay.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Delay Rate by Airline (%)")
    ax.set_xlabel("Airline")
    ax.set_ylabel("Delay Rate (%)")
    ax.tick_params(axis="x", rotation=45)

    # 3 — delay rate by month
    ax = axes[1, 0]
    month_delay = df.groupby("month")["is_delayed"].mean() * 100
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_delay.index = month_names[: len(month_delay)]
    month_delay.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Delay Rate by Month (%)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Delay Rate (%)")
    ax.tick_params(axis="x", rotation=45)

    # 4 — delay rate by day of week
    ax = axes[1, 1]
    dow_delay = df.groupby("day_of_week")["is_delayed"].mean() * 100
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_delay.index = dow_names[: len(dow_delay)]
    dow_delay.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Delay Rate by Day of Week (%)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Delay Rate (%)")
    ax.tick_params(axis="x", rotation=45)

    # 5 — delay rate by departure hour
    ax = axes[2, 0]
    hour_delay = df.groupby("dep_hour")["is_delayed"].mean() * 100
    hour_delay.plot(kind="line", ax=ax, color="steelblue", marker="o")
    ax.set_title("Delay Rate by Departure Hour (%)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Delay Rate (%)")
    ax.set_xticks(range(0, 25))

    # 6 — delay rate: weekday vs weekend
    ax = axes[2, 1]
    weekend_delay = df.groupby("is_weekend")["is_delayed"].mean() * 100
    weekend_delay.index = ["Weekday", "Weekend"][: len(weekend_delay)]
    weekend_delay.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="black")
    ax.set_title("Delay Rate: Weekday vs Weekend (%)")
    ax.set_ylabel("Delay Rate (%)")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    _save(fig, out_path)


def plot_weather_eda(config: dict) -> None:
    """4-panel histogram of raw weather variable distributions."""
    weather_path = config["files"]["processed"]["weather_features"]
    out_path = config["files"]["reports"]["eda_weather"]

    weather = pd.read_csv(weather_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Weather Data EDA", fontsize=16, fontweight="bold")

    specs = [
        ("precipitation",  "Precipitation Distribution (mm)",  "steelblue"),
        ("temperature_c",  "Temperature Distribution (C)",     "tomato"),
        ("humidity_pct",   "Humidity Distribution (%)",        "seagreen"),
        ("wind_speed_kmh", "Wind Speed Distribution (km/h)",   "orange"),
    ]

    for ax, (col, title, color) in zip(axes.flat, specs):
        ax.hist(weather[col], bins=50, color=color, edgecolor="black")
        ax.set_title(title)
        ax.set_ylabel("Count")

    fig.tight_layout()
    _save(fig, out_path)


def plot_weather_coverage(config: dict) -> None:
    """Bar chart: flight volume from weather-covered vs other origin airports."""
    flight_path = config["files"]["processed"]["flight_features"]
    weather_path = config["files"]["processed"]["weather_features"]
    out_path = config["files"]["reports"]["weather_coverage"]

    flights = pd.read_csv(flight_path)
    weather_hourly = pd.read_csv(weather_path)
    covered_airports = weather_hourly["iata"].unique()

    origin_counts = flights["origin"].value_counts()
    covered_vol   = origin_counts[origin_counts.index.isin(covered_airports)].sum()
    other_vol     = origin_counts[~origin_counts.index.isin(covered_airports)].sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        ["Covered airports\n(weather available)", "Other airports\n(weather imputed)"],
        [covered_vol, other_vol],
        color=["steelblue", "lightgrey"],
        edgecolor="black",
    )
    ax.set_title("Flight Volume: Weather-Covered vs Other Origins")
    ax.set_ylabel("Number of Flights")
    total = len(flights)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height() / total * 100:.1f}%",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontsize=11,
        )

    fig.tight_layout()
    _save(fig, out_path)


def plot_correlation_heatmap(config: dict) -> None:
    """Correlation heatmap of all numeric features in the merged dataset."""
    merged_path = config["files"]["processed"]["merged_dataset"]
    out_path = config["files"]["reports"]["correlation_heatmap"]

    df = pd.read_csv(merged_path)

    num_cols = [
        "distance", "dep_hour", "precipitation", "temperature_c",
        "humidity_pct", "wind_speed_kmh", "day_of_week", "month",
        "is_weekend", "is_delayed",
    ]
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        annot_kws={"size": 9}, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def plot_weather_delay_rates(config: dict) -> None:
    """Delay rate bucketed by low/medium/high weather condition level (4 panels)."""
    merged_path = config["files"]["processed"]["merged_dataset"]
    out_path = config["files"]["reports"]["eda_weather_delays"]

    df = pd.read_csv(merged_path)

    bin_specs = [
        ("precipitation",  "precip_bin",  ["Low", "Medium", "High"],           "steelblue",  "Precipitation Level"),
        ("temperature_c",  "temp_bin",    ["Cold", "Mild", "Hot"],              "tomato",     "Temperature Level"),
        ("wind_speed_kmh", "wind_bin",    ["Calm", "Moderate", "Windy"],        "seagreen",   "Wind Speed Level"),
        ("humidity_pct",   "humid_bin",   ["Dry", "Normal", "Humid"],           "darkorange", "Humidity Level"),
    ]

    for col, bin_col, labels, *_ in bin_specs:
        if col in df.columns:
            df[bin_col] = _safe_qcut(df[col], 3, labels)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Delay Rate by Weather Conditions", fontsize=15, fontweight="bold")

    for ax, (col, bin_col, _, color, title) in zip(axes.flat, bin_specs):
        if bin_col not in df.columns:
            continue
        rates = df.groupby(bin_col, observed=True)["is_delayed"].mean() * 100
        rates.plot(kind="bar", ax=ax, color=color, edgecolor="black")
        ax.set_title(title)
        ax.set_ylabel("Delay Rate (%)")
        ax.tick_params(axis="x", rotation=0)
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.1f}%",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10,
            )

    fig.tight_layout()
    _save(fig, out_path)


def plot_weather_boxplots(config: dict) -> None:
    """Weather feature distributions split by on-time vs delayed (boxplots)."""
    merged_path = config["files"]["processed"]["merged_dataset"]
    out_path = config["files"]["reports"]["eda_weather_boxplots"]

    df = pd.read_csv(merged_path)
    weather_features = ["precipitation", "temperature_c", "humidity_pct", "wind_speed_kmh"]
    available = [c for c in weather_features if c in df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
    if len(available) == 1:
        axes = [axes]
    fig.suptitle("Weather Feature Distribution: On-Time vs Delayed", fontsize=13, fontweight="bold")

    for ax, col in zip(axes, available):
        df.boxplot(
            column=col, by="is_delayed", ax=ax,
            boxprops=dict(color="steelblue"),
            medianprops=dict(color="tomato", linewidth=2),
        )
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("is_delayed  (0 = On-Time, 1 = Delayed)")
        ax.set_ylabel("")

    plt.suptitle("Weather Feature Distribution: On-Time vs Delayed", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def plot_class_balance(config: dict) -> None:
    """Bar charts comparing class distribution before and after undersampling."""
    splits_cfg = config["files"]["splits"]
    out_path = config["files"]["reports"]["class_balance_comparison"]

    y_train_orig_path = splits_cfg.get("y_train_original")
    y_train_bal_path  = splits_cfg["y_train"]
    target_col = config.get("preprocessing", {}).get("target_column", "is_delayed")

    if y_train_orig_path:
        y_orig = pd.read_csv(y_train_orig_path)[target_col]
    else:
        # Reconstruct "before" from balanced + note that we can't perfectly recover it.
        # Use the balanced split as both panels with a clear title difference.
        y_orig = None

    y_bal = pd.read_csv(y_train_bal_path)[target_col]

    panels = []
    if y_orig is not None:
        panels.append((y_orig, "Original Training Set"))
    panels.append((y_bal, "After Random Undersampling (2:1)"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    fig.suptitle("Class Distribution: Before vs After Undersampling", fontsize=13, fontweight="bold")

    for ax, (y_data, title) in zip(axes, panels):
        vc = y_data.value_counts()
        ax.bar(
            ["On-Time (0)", "Delayed (1)"],
            [vc.get(0, 0), vc.get(1, 0)],
            color=["steelblue", "tomato"],
            edgecolor="black",
        )
        ax.set_title(title)
        ax.set_ylabel("Count")
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():,.0f}\n({p.get_height() / len(y_data) * 100:.1f}%)",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=9,
            )

    fig.tight_layout()
    _save(fig, out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

REPORTS: dict[str, object] = {
    "flight":           plot_flight_eda,
    "weather":          plot_weather_eda,
    "coverage":         plot_weather_coverage,
    "correlation":      plot_correlation_heatmap,
    "weather-delays":   plot_weather_delay_rates,
    "weather-boxplots": plot_weather_boxplots,
    "class-balance":    plot_class_balance,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA reports.")
    parser.add_argument("--config", default="configs/config.toml", help="Path to TOML config file")
    parser.add_argument(
        "--report",
        choices=list(REPORTS) + ["all"],
        default="all",
        help="Which report to generate (default: all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = _load_toml(args.config)
    targets = list(REPORTS) if args.report == "all" else [args.report]
    for name in targets:
        print(f"\n--- {name} ---")
        try:
            REPORTS[name](cfg)  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: skipped ({exc})")
