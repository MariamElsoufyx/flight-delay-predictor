"""Full pipeline runner for the flight-delay-predictor project.

Execution order
---------------
1. validate        -- validate all three raw datasets (flight, weather, airports)
2. clean-airports  -- strip whitespace / normalise casing in airports CSV
3. flight-features -- raw flight CSV -> data/processed/flight_features.csv
4. weather-features-- raw weather CSV -> data/processed/weather_features.csv
5. merge           -- flight_features + weather_features -> data/processed/merged_dataset.csv
6. split           -- merged_dataset -> train/val/test splits + label encoders + scaler
7. eda             -- generate all EDA report PNGs -> outputs/reports/
8. train           -- train baseline + tuned models -> outputs/models/
9. classify        -- evaluate models, confusion matrix, feature importance -> outputs/reports/

Run the full pipeline (use Poetry so xgboost and other deps are available):
    poetry run python run_pipeline.py

Run only specific steps (comma-separated):
    poetry run python run_pipeline.py --steps validate,flight-features,weather-features

Skip specific steps:
    poetry run python run_pipeline.py --skip validate

Show what would run without executing:
    poetry run python run_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time


CONFIG = "configs/config.toml"

# ordered step registry 

STEPS: list[tuple[str, str]] = [
    ("validate",         "Validate raw datasets (flight, weather, airports)"),
    ("clean-airports",   "Clean airport strings -> airports_clean.csv"),
    ("flight-features",  "Engineer flight features -> flight_features.csv"),
    ("weather-features", "Engineer weather features -> weather_features.csv"),
    ("merge",            "Merge flight + weather, impute, cap outliers -> merged_dataset.csv"),
    ("split",            "Encode, scale, split, balance -> splits + encoders"),
    ("eda",              "Generate all EDA report PNGs -> outputs/reports/"),
    ("train",            "Train LR / RF / XGB + tuned XGB -> outputs/models/"),
    ("classify",         "Evaluate saved models + plots -> outputs/reports/"),
]

STEP_NAMES = [s[0] for s in STEPS]


# individual step runners

def run_validate(cfg: dict) -> None:
    from src.data.validate import run_default_validations
    run_default_validations.__module__  # noqa: ensure import
    run_default_validations(cfg.get("_config_path", CONFIG))


def run_clean_airports(cfg: dict) -> None:
    from src.data.preprocess import clean_airports_dataset
    clean_airports_dataset(cfg)


def run_flight_features(cfg: dict) -> None:
    from src.features.engineering import build_flight_features
    build_flight_features(cfg)


def run_weather_features(cfg: dict) -> None:
    from src.features.engineering import build_weather_features
    build_weather_features(cfg)


def run_merge(cfg: dict) -> None:
    from src.data.preprocess import merge_flight_weather
    merge_flight_weather(cfg)


def run_split(cfg: dict) -> None:
    from src.data.preprocess import preprocess_merged_dataset
    preprocess_merged_dataset(cfg)


def run_eda(cfg: dict) -> None:
    from src.reports.eda import REPORTS
    for name, fn in REPORTS.items():
        print(f"  plotting: {name}")
        try:
            fn(cfg)  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: {name} skipped ({exc})")


def run_train(cfg: dict) -> None:
    from src.models.train import run_training

    run_training(cfg)


def run_classify(cfg: dict) -> None:
    from src.models.classify import run_classification

    run_classification(cfg)


RUNNERS: dict[str, object] = {
    "validate":         run_validate,
    "clean-airports":   run_clean_airports,
    "flight-features":  run_flight_features,
    "weather-features": run_weather_features,
    "merge":            run_merge,
    "split":            run_split,
    "eda":              run_eda,
    "train":            run_train,
    "classify":         run_classify,
}


# helpers

def _banner(text: str, char: str = "=", width: int = 60) -> None:
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


# main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the flight-delay-predictor pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {n:<20} {d}" for n, d in STEPS),
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("FLIGHT_DELAY_CONFIG", CONFIG),
        help="Path to TOML config (default: env FLIGHT_DELAY_CONFIG or configs/config.toml)",
    )
    parser.add_argument(
        "--steps",
        default=",".join(STEP_NAMES),
        help="Comma-separated list of steps to run (default: all)",
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated list of steps to skip",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print steps without executing them")
    return parser.parse_args()


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    args = parse_args()

    requested = [s.strip() for s in args.steps.split(",") if s.strip()]
    skipped   = {s.strip() for s in args.skip.split(",") if s.strip()}

    # Validate step names
    unknown = [s for s in requested + list(skipped) if s not in STEP_NAMES]
    if unknown:
        print(f"ERROR: unknown step(s): {', '.join(unknown)}")
        print(f"Valid steps: {', '.join(STEP_NAMES)}")
        sys.exit(1)

    # Preserve canonical order regardless of what the user typed
    to_run = [s for s in STEP_NAMES if s in requested and s not in skipped]

    _banner("FLIGHT DELAY PREDICTOR — PIPELINE")
    print(f"Config : {args.config}")
    print(f"Steps  : {', '.join(to_run)}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")

    if args.dry_run:
        print("\n[dry-run] No steps executed.")
        return

    # Load config once for all steps
    from src.data.preprocess import load_toml_config
    cfg = load_toml_config(args.config)
    cfg["_config_path"] = args.config

    results: list[tuple[str, str, float]] = []  # (step, status, duration)

    for step in to_run:
        _banner(f"STEP: {step}", char="-")
        t0 = time.perf_counter()
        try:
            RUNNERS[step](cfg)  # type: ignore[operator]
            duration = time.perf_counter() - t0
            results.append((step, "OK", duration))
            print(f"\n[{step}] done in {_fmt_duration(duration)}")
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - t0
            results.append((step, f"FAILED: {exc}", duration))
            print(f"\n[{step}] FAILED after {_fmt_duration(duration)}: {exc}")
            _banner("PIPELINE ABORTED", char="!")
            _print_summary(results)
            sys.exit(1)

    _banner("PIPELINE COMPLETE")
    _print_summary(results)


def _print_summary(results: list[tuple[str, str, float]]) -> None:
    print(f"\n{'Step':<22} {'Status':<12} {'Duration'}")
    print("-" * 48)
    for step, status, duration in results:
        print(f"  {step:<20} {status:<12} {_fmt_duration(duration)}")
    total = sum(d for _, _, d in results)
    print("-" * 48)
    print(f"  {'Total':<20} {'':12} {_fmt_duration(total)}")


if __name__ == "__main__":
    main()
