"""
Stakeholder dashboard: EDA highlights, model comparison, plain-language insights.

Run from the project root (so paths in configs/config.toml resolve):

    poetry run streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Add project root to Python path for module imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dashboard.io import load_dashboard_context

INTRO = """
This view summarises **flight delay** work for airline and airport stakeholders.
Technical details stay in the background: focus here is **patterns**, **model quality**,
and **what actions** the numbers might support.
"""


def _glossary() -> None:
    with st.expander("What do these terms mean?"):
        st.markdown(
            """
| Term (simple) | Meaning |
|----------------|---------|
| **On-time vs delayed** | We predict whether a flight is likely to be **late** (delayed) using past data. |
| **Accuracy** | Share of flights where the prediction matched what actually happened. |
| **Precision (delays)** | When the model **warns** a delay, how often that warning was correct. |
| **Recall (delays)** | Of all flights that **really were delayed**, how many the model **caught**. |
| **F1 (delays)** | A balance between precision and recall for delay alerts. |
| **ROC-AUC** | How well the model **ranks** delayed flights higher than on-time ones (1.0 is best). |
"""
        )


def _business_takeaways(perf: pd.DataFrame | None) -> None:
    st.subheader("What this means for operations")
    st.markdown(
        """
- **Staffing & gates:** Higher **recall** on delays means fewer surprises for crews and passengers
  (we catch more real delays). Lower recall means more **last-minute** disruptions.
- **Customer messaging:** Higher **precision** means fewer **false delay warnings**
  (less noise in apps and announcements).
- **Planning window:** Weather and time-of-day patterns help decide **where** to invest
  in buffer time, not just **whether** a single flight is late.
"""
    )
    if perf is None or perf.empty:
        st.info("Run the **classify** pipeline step after training to populate model scores here.")
        return
    best = perf.iloc[0]
    name = best["Model"]
    st.success(
        f"**Recommended model to discuss first:** **{name}** "
        f"(highest ROC-AUC on held-out test data among the candidates shown)."
    )
    recall = best.get("Test_Recall_Delayed")
    prec = best.get("Test_Precision_Delayed")
    if recall is not None and pd.notna(recall):
        st.markdown(
            f"- **Catching real delays (recall):** about **{recall:.0%}** of true delays are flagged "
            f"with **{name}** — compare with other models in the chart tab."
        )
    if prec is not None and pd.notna(prec):
        st.markdown(
            f"- **Trust in delay alerts (precision):** when this model predicts a delay, it is right "
            f"about **{prec:.0%}** of the time."
        )


def render() -> None:
    st.set_page_config(
        page_title="Flight delay insights",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Flight delay insights")
    st.caption("Stakeholder view — EDA, models, and plain-language takeaways")
    st.markdown(INTRO)

    with st.sidebar:
        st.header("Settings")
        default_cfg = Path("configs/config.toml")
        cfg_path = st.text_input("Config file", value=str(default_cfg))
        root_input = st.text_input(
            "Project folder (for file paths)",
            value=str(Path.cwd()),
            help="Paths in config.toml are relative to this folder.",
        )
        project_root = Path(root_input).expanduser().resolve()
        if st.button("Reload data"):
            st.cache_data.clear()

    try:
        ctx = load_dashboard_context(cfg_path, project_root=project_root)
    except FileNotFoundError:
        st.error(f"Config not found: {cfg_path}")
        st.stop()
    except OSError as exc:
        st.error(f"Could not read config: {exc}")
        st.stop()

    cfg = ctx["config"]
    delay_def = cfg.get("flight_pipeline", {}).get("delay_threshold_minutes", 15)
    st.sidebar.metric("Delay definition", f">{int(delay_def)} min late")

    tab_overview, tab_eda, tab_models, tab_help = st.tabs(
        ["Overview", "Data stories (charts)", "Model comparison", "Help"]
    )

    with tab_overview:
        st.subheader("At a glance")
        st.markdown(
            f"""
We study **whether a flight is likely to be delayed**, using features such as **route**,
**time of day**, **calendar effects**, and **weather** where available.
A flight counts as **delayed** when arrival is more than **{int(delay_def)} minutes** late
(configurable in `configs/config.toml`).
"""
        )
        _business_takeaways(ctx["perf_df"])
        _glossary()

    with tab_eda:
        st.subheader("Charts from the exploration phase")
        st.markdown(
            "Each picture below was produced by the project’s **EDA** scripts. "
            "If a file is missing, run **`poetry run python -m src.reports.eda`** (or the full pipeline)."
        )
        for item in ctx["eda_items"]:
            st.markdown(f"#### {item['title']}")
            if item["exists"]:
                st.image(str(item["path"]), use_container_width=True)
            else:
                st.warning(f"File not found: `{item['path']}`")

    with tab_models:
        st.subheader("Model performance (test data)")
        perf = ctx["perf_df"]
        if perf is None or perf.empty:
            st.info(
                "No `model_performance_summary.csv` yet. "
                "Run **`poetry run python run_pipeline.py --steps train,classify`** "
                "or **`poetry run python -m src.models.classify`** after training."
            )
        else:
            show = perf.copy()
            rename_map = {
                "Model": "Model",
                "Test_Accuracy": "Test accuracy",
                "Test_Precision_Delayed": "Test precision (delays)",
                "Test_Recall_Delayed": "Test recall (delays)",
                "Test_F1_Delayed": "Test F1 (delays)",
                "Test_ROC_AUC": "Test ROC-AUC",
            }
            cols = [c for c in rename_map if c in show.columns]
            st.dataframe(
                show[cols].rename(columns={k: rename_map[k] for k in cols}),
                use_container_width=True,
                hide_index=True,
            )
            chart_cols = [c for c in ("Test_ROC_AUC", "Test_F1_Delayed", "Test_Recall_Delayed") if c in show.columns]
            if chart_cols:
                melted = show.melt(id_vars=["Model"], value_vars=chart_cols, var_name="Metric", value_name="Score")
                fig = px.bar(
                    melted,
                    x="Model",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title="Side-by-side model scores (higher is better for all three)",
                    labels={"Model": "", "Score": ""},
                )
                fig.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Extra visuals (after evaluation)")
        rep = ctx["reports_dir"]
        cm_paths = sorted(rep.glob("confusion_matrix_*.png"))
        if cm_paths:
            st.markdown("**Confusion matrix** for the best-performing candidate model (test set).")
            st.image(str(cm_paths[0]), use_container_width=True)
        fi = rep / "feature_importance.png"
        if fi.is_file():
            st.markdown("**Drivers of delay risk** (top factors from the final model, when available).")
            st.image(str(fi), use_container_width=True)

    with tab_help:
        st.markdown(
            """
### How to refresh this dashboard
1. Run the data and EDA steps (or full pipeline) so PNGs exist under `outputs/reports/`.
2. Run **train** then **classify** so `model_performance_summary.csv` is written.
3. Click **Reload data** in the sidebar.

### Where numbers come from
- Charts: `src/reports/eda.py` and related outputs.
- Tables: `src/models/classify.py` writes `model_performance_summary.csv`.

### Privacy & scope
This demo reads **aggregated** files from your machine only; it does not send data externally.
"""
        )


def main() -> None:
    render()


if __name__ == "__main__":
    main()
