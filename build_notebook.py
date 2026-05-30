from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "analysis.ipynb"


def build_interpretation(metrics: dict, coefficients: pd.DataFrame, rankings: pd.DataFrame) -> str:
    coefficient_lookup = coefficients.set_index("term")
    recent_vol_coef = coefficient_lookup.loc["recent_volatility_7d", "coef"]
    category_terms = coefficients[coefficients["term"].str.contains("C(category", regex=False, na=False)].copy()
    significant_terms = category_terms[category_terms["p_value"] < 0.05].sort_values("p_value")

    highest_category = rankings.iloc[0]
    lowest_category = rankings.iloc[-1]
    reference_category = metrics["ols_reference_category"]
    if significant_terms.empty:
        category_sentence = (
            f"Relative to {reference_category}, no alternate category clears the 5% significance threshold "
            "in this run after adding the control variables."
        )
    else:
        lead_term = significant_terms.iloc[0]
        lead_direction = "higher" if lead_term["coef"] > 0 else "lower"
        category_sentence = (
            f"Relative to {reference_category}, the clearest category effect is "
            f"{lead_term['term']} with a {lead_direction} forward-volatility coefficient of "
            f"{lead_term['coef']:.4f} (p={lead_term['p_value']:.3g}). "
            f"{int(metrics['significant_category_terms'])} category term(s) are significant at the 5% level."
        )

    coverage_snapshot = ", ".join(
        f"{row.category} ({int(row.market_count)} markets)" for row in rankings.head(4).itertuples()
    )

    return dedent(
        f"""
        The MVP sample contains {metrics['n_markets']} markets and {metrics['n_rows']:,} training rows after filtering. The headline pattern is that category matters unevenly rather than uniformly. {category_sentence} The OLS adjusted R² is {metrics['ols_adj_r2']:.3f}, so the controls explain a meaningful share of the variation even in a rough one-day build.

        The volatility story itself is intuitive. {highest_category['category']} has the highest median realized 7-day volatility ({highest_category['median_volatility']:.4f}), while {lowest_category['category']} has the lowest ({lowest_category['median_volatility']:.4f}). Recent volatility remains a strong positive predictor of future volatility ({recent_vol_coef:.3f}), but the random forest indicates that bid-ask spread and volume carry most of the non-linear predictive weight, with category contributing about {metrics['category_importance_share']:.1%} of total feature importance. The important caveat is data coverage: the live Polymarket open-market sample is concentrated in a handful of long-duration categories, especially {coverage_snapshot}, and historical spread is approximated with the latest observed bid/ask spread because the ingestion path does not expose a clean daily spread history.
        """
    ).strip()


def create_notebook() -> nbf.NotebookNode:
    metrics = json.loads((DATA_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    coefficients = pd.read_csv(DATA_DIR / "ols_coefficients.csv")
    rankings = pd.read_csv(DATA_DIR / "category_rankings.csv")
    interpretation = build_interpretation(metrics, coefficients, rankings)

    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.9"}

    cells = [
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # Modeling Volatility Risk Across Event Categories in Prediction Markets

                **Research question:** Can market category predict future probability volatility in prediction markets, even after controlling for liquidity, volume, and time-to-expiry?

                This notebook uses Polymarket data pulled from the live Gamma + CLOB APIs, with daily probability histories and market-level microstructure controls.
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Setup

                This notebook assumes it is being run from either the project root or the `notebooks/` directory.
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                from pathlib import Path
                import json

                import numpy as np
                import pandas as pd
                import plotly.express as px
                import plotly.graph_objects as go
                from IPython.display import display

                project_root = Path.cwd()
                if project_root.name == "notebooks":
                    project_root = project_root.parent
                data_dir = project_root / "data"

                analytical = pd.read_csv(data_dir / "analytical_dataset.csv", parse_dates=["date"])
                scoring = pd.read_csv(data_dir / "scoring_dataset.csv", parse_dates=["date"])
                coefficients = pd.read_csv(data_dir / "ols_coefficients.csv")
                rankings = pd.read_csv(data_dir / "category_rankings.csv")
                importances = pd.read_csv(data_dir / "rf_feature_importances.csv")
                volatility_over_time = pd.read_csv(data_dir / "volatility_over_time.csv", parse_dates=["date"])
                with open(data_dir / "model_metrics.json", "r", encoding="utf-8") as handle:
                    metrics = json.load(handle)

                analytical.head()
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## Data Coverage"),
        nbf.v4.new_code_cell(
            dedent(
                """
                category_summary = (
                    scoring.groupby("category", as_index=False)
                    .agg(
                        markets=("market_id", "nunique"),
                        observations=("market_id", "size"),
                        median_volatility=("rolling_volatility_7d", "median"),
                        mean_volatility=("rolling_volatility_7d", "mean"),
                    )
                    .sort_values("markets", ascending=False)
                )
                category_summary["underrepresented_lt_20"] = category_summary["markets"] < 20
                display(category_summary)

                print("Training rows:", metrics["n_rows"])
                print("Markets:", metrics["n_markets"])
                print("OLS adjusted R²:", round(metrics["ols_adj_r2"], 3))
                print("Random forest mean CV R²:", round(metrics["rf_mean_r2"], 3))
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## Volatility by Category"),
        nbf.v4.new_code_cell(
            dedent(
                """
                order = (
                    scoring.groupby("category")["rolling_volatility_7d"]
                    .median()
                    .sort_values(ascending=False)
                    .index
                )

                fig = px.box(
                    scoring,
                    x="category",
                    y="rolling_volatility_7d",
                    category_orders={"category": order.tolist()},
                    points=False,
                    title="Distribution of 7-day rolling volatility by category",
                )
                fig.update_layout(xaxis_title="", yaxis_title="7d rolling volatility")
                fig.show()
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## Volatility Over Time"),
        nbf.v4.new_code_cell(
            dedent(
                """
                fig = px.line(
                    volatility_over_time,
                    x="date",
                    y="avg_rolling_volatility_7d",
                    color="category",
                    title="Average category-level volatility over time",
                )
                fig.update_layout(xaxis_title="", yaxis_title="Average 7d rolling volatility")
                fig.show()
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## Correlation Matrix"),
        nbf.v4.new_code_cell(
            dedent(
                """
                corr_columns = [
                    "rolling_volatility_7d",
                    "future_volatility_7d",
                    "log_volume",
                    "bid_ask_spread",
                    "time_to_expiry_days",
                    "prob_level",
                    "recent_volatility_7d",
                ]
                corr = analytical[corr_columns].corr(numeric_only=True)
                heatmap = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale="Blues",
                    title="Feature correlation matrix",
                )
                heatmap.show()
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## Volatility vs Time to Expiry"),
        nbf.v4.new_code_cell(
            dedent(
                """
                sample = analytical.sample(min(len(analytical), 6000), random_state=42)
                fig = px.scatter(
                    sample,
                    x="time_to_expiry_days",
                    y="rolling_volatility_7d",
                    color="category",
                    opacity=0.35,
                    trendline="lowess",
                    title="Does realized volatility rise near expiry?",
                )
                fig.update_layout(xaxis_title="Days to expiry", yaxis_title="7d rolling volatility")
                fig.show()
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## Volatility vs Probability Level"),
        nbf.v4.new_code_cell(
            dedent(
                """
                sample = analytical.sample(min(len(analytical), 6000), random_state=7)
                fig = px.scatter(
                    sample,
                    x="prob_level",
                    y="rolling_volatility_7d",
                    color="category",
                    opacity=0.35,
                    trendline="lowess",
                    title="Volatility vs implied probability level",
                )
                fig.update_layout(xaxis_title="Implied probability", yaxis_title="7d rolling volatility")
                fig.show()
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## OLS Regression Results"),
        nbf.v4.new_code_cell(
            dedent(
                """
                display(coefficients.sort_values("p_value"))

                significant_categories = coefficients[
                    coefficients["term"].str.contains("C(category", regex=False, na=False)
                    & (coefficients["p_value"] < 0.05)
                ][["term", "coef", "p_value"]]
                significant_categories
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## Random Forest Feature Importance"),
        nbf.v4.new_code_cell(
            dedent(
                """
                fig = px.bar(
                    importances.sort_values("importance", ascending=True),
                    x="importance",
                    y="feature_group",
                    orientation="h",
                    title="Aggregated random forest feature importances",
                    color="importance",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(coloraxis_showscale=False, xaxis_title="Importance", yaxis_title="")
                fig.show()

                print("Top feature:", metrics["top_feature"])
                print("Second feature:", metrics["second_feature"])
                print("Category importance share:", round(metrics["category_importance_share"] * 100, 2), "%")
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## Interpretation"),
        nbf.v4.new_markdown_cell(interpretation),
    ]

    notebook["cells"] = cells
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = create_notebook()
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    print(f"[done] wrote notebook to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
