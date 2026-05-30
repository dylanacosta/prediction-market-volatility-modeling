from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA_DIR = Path("data")
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}
PALETTE = [
    "#6EE7F9",
    "#F6C177",
    "#7EE081",
    "#F78FB3",
    "#A78BFA",
    "#4ADE80",
    "#F97316",
    "#2DD4BF",
]
RANKING_SORT_OPTIONS = {
    "Median volatility": "median_volatility",
    "Mean volatility": "mean_volatility",
    "Market count": "market_count",
    "Average volume": "avg_volume",
}
TIME_RANGE_OPTIONS = ["Last 30 days", "Year to date", "Last 2 years", "All data", "Custom range"]


@st.cache_data
def load_data() -> dict[str, pd.DataFrame | dict]:
    analytical = pd.read_csv(DATA_DIR / "analytical_dataset.csv", parse_dates=["date"])
    scoring = pd.read_csv(DATA_DIR / "scoring_dataset.csv", parse_dates=["date"])
    latest_scores = pd.read_csv(DATA_DIR / "latest_market_scores.csv", parse_dates=["date"])
    importances = pd.read_csv(DATA_DIR / "rf_feature_importances.csv")
    coefficients = pd.read_csv(DATA_DIR / "ols_coefficients.csv")
    volatility_over_time = pd.read_csv(DATA_DIR / "volatility_over_time.csv", parse_dates=["date"])
    category_rankings = pd.read_csv(DATA_DIR / "category_rankings.csv")
    with open(DATA_DIR / "model_metrics.json", "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return {
        "analytical": analytical,
        "scoring": scoring,
        "latest_scores": latest_scores,
        "importances": importances,
        "coefficients": coefficients,
        "volatility_over_time": volatility_over_time,
        "category_rankings": category_rankings,
        "metrics": metrics,
    }


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        :root {
            --bg: #07111d;
            --bg-elevated: #0c1828;
            --surface: #0f1c2f;
            --surface-strong: #14253a;
            --surface-soft: #112033;
            --border: rgba(126, 148, 177, 0.18);
            --border-strong: rgba(110, 231, 249, 0.28);
            --text: #e6edf7;
            --muted: #94a7c2;
            --accent: #6ee7f9;
            --accent-warm: #f6c177;
            --accent-green: #7ee081;
            --shadow: 0 18px 42px rgba(2, 8, 18, 0.34);
        }

        html, body, [class*="css"]  {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .stApp {
            background:
                linear-gradient(180deg, rgba(9, 18, 30, 0.98) 0%, rgba(7, 17, 29, 1) 100%);
            color: var(--text);
        }

        .main .block-container {
            max-width: 1480px;
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stAppViewContainer"] {
            overflow-x: hidden;
        }

        h1, h2, h3, h4, .section-title, .hero-title {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text);
            letter-spacing: 0;
        }

        p, li, label, .stMarkdown, .stCaption {
            color: var(--muted);
        }

        [data-testid="stSidebar"] > div:first-child {
            background:
                linear-gradient(180deg, rgba(13, 24, 40, 0.98) 0%, rgba(9, 18, 31, 0.98) 100%);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1rem;
        }

        .sidebar-title {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0 0 0.25rem 0;
        }

        .sidebar-copy {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.45;
            margin-bottom: 1rem;
        }

        .section-shell {
            margin-bottom: 0.75rem;
        }

        .section-kicker {
            color: var(--accent);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .section-title {
            font-size: 1.18rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.2rem;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.5;
            margin-bottom: 0.3rem;
        }

        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, rgba(110, 231, 249, 0.28), rgba(110, 231, 249, 0));
            margin: 0.9rem 0 1.2rem 0;
        }

        .hero-card, .info-card, .stat-card {
            background: linear-gradient(180deg, rgba(17, 32, 51, 0.96) 0%, rgba(12, 24, 40, 0.98) 100%);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: var(--shadow);
        }

        .hero-card {
            padding: 1.35rem 1.45rem 1.15rem 1.45rem;
            min-height: 188px;
        }

        .hero-eyebrow {
            color: var(--accent);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .hero-title {
            font-size: 2.2rem;
            line-height: 1;
            margin-bottom: 0.65rem;
            font-weight: 700;
        }

        .hero-copy {
            color: var(--muted);
            max-width: 58rem;
            line-height: 1.6;
            font-size: 0.97rem;
            margin-bottom: 0.9rem;
        }

        .hero-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            align-items: stretch;
        }

        .hero-pill {
            border: 1px solid rgba(110, 231, 249, 0.22);
            border-radius: 999px;
            padding: 0.34rem 0.65rem;
            color: var(--text);
            font-size: 0.8rem;
            background: rgba(17, 32, 51, 0.68);
            max-width: 100%;
        }

        .info-card {
            padding: 1.15rem 1.2rem;
            min-height: 188px;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 0.9rem;
        }

        .info-label {
            color: var(--muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.2rem;
        }

        .info-value {
            color: var(--text);
            font-size: 1.02rem;
            font-weight: 600;
        }

        .stat-card {
            padding: 0.95rem 1rem 0.9rem 1rem;
            border-left: 4px solid var(--accent);
            min-height: 116px;
            margin-bottom: 0.35rem;
        }

        .stat-card.warm {
            border-left-color: var(--accent-warm);
        }

        .stat-card.green {
            border-left-color: var(--accent-green);
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .stat-value {
            color: var(--text);
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.65rem;
            line-height: 1;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }

        .stat-sub {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.45;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(17, 32, 51, 0.96) 0%, rgba(12, 24, 40, 0.98) 100%);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: 16px;
            box-shadow: var(--shadow);
            padding: 0.75rem 1rem;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-family: 'Space Grotesk', sans-serif;
        }

        div[data-testid="stMetricDelta"] {
            color: var(--accent);
        }

        .stTabs [data-baseweb="tab-list"] {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            background: rgba(9, 18, 31, 0.72);
            border: 1px solid var(--border);
            padding: 0.35rem;
            border-radius: 14px;
        }

        .stTabs [data-baseweb="tab"] {
            min-height: 44px;
            height: auto;
            background: transparent;
            border-radius: 10px;
            color: var(--muted);
            padding: 0 1rem;
            font-family: 'Space Grotesk', sans-serif;
            flex: 1 1 0;
            text-align: center;
            justify-content: center;
            white-space: normal;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(110, 231, 249, 0.12);
            color: var(--text);
        }

        .stTabs [data-baseweb="tab"] p {
            white-space: normal;
            text-align: center;
            line-height: 1.2;
            margin: 0;
        }

        .stButton > button,
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-testid="stDateInputField"] {
            background: rgba(17, 32, 51, 0.92);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        .stButton > button {
            padding: 0.6rem 0.95rem;
            transition: all 0.18s ease;
        }

        .stButton > button:hover {
            border-color: var(--border-strong);
            color: var(--text);
            background: rgba(23, 42, 67, 0.98);
        }

        div[data-baseweb="select"] > div {
            min-height: 56px;
            height: auto !important;
            padding-top: 0.2rem;
            padding-bottom: 0.2rem;
        }

        [data-baseweb="tag"] {
            background: rgba(247, 143, 179, 0.14) !important;
            border: 1px solid rgba(247, 143, 179, 0.3) !important;
            border-radius: 999px !important;
            max-width: 100%;
        }

        [data-baseweb="tag"] span {
            max-width: 110px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input,
        div[data-baseweb="input"] input,
        .stMultiSelect label,
        .stSlider label,
        .stSelectbox label {
            color: var(--text);
        }

        .stSlider [data-baseweb="slider"] > div div {
            background: var(--accent);
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            background: rgba(12, 24, 40, 0.84);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
        }

        [data-testid="stPlotlyChart"],
        [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] .js-plotly-plot,
        [data-testid="stPlotlyChart"] .plot-container {
            width: 100% !important;
            max-width: 100% !important;
        }

        [data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }

        [data-testid="stMarkdownContainer"] a {
            color: var(--accent);
        }

        @media (max-width: 1200px) {
            .main .block-container {
                width: 100% !important;
                max-width: 100% !important;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            [data-testid="stHorizontalBlock"] {
                display: flex !important;
                flex-direction: column !important;
                gap: 1rem;
            }

            [data-testid="stColumn"],
            [data-testid="column"] {
                width: 100% !important;
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }
        }

        @media (max-width: 900px) {
            .main .block-container {
                padding-top: 1.4rem;
            }

            .hero-title {
                font-size: 1.78rem;
            }

            .hero-copy {
                font-size: 0.92rem;
            }

            .info-grid {
                grid-template-columns: 1fr;
            }

            .hero-card,
            .info-card,
            .stat-card {
                width: 100%;
                min-height: auto;
            }

            .stTabs [data-baseweb="tab-list"] {
                overflow: visible;
                gap: 0.4rem;
            }

            .stTabs [data-baseweb="tab"] {
                flex: 1 1 calc(50% - 0.4rem);
                min-width: calc(50% - 0.4rem);
                padding: 0.65rem 0.75rem;
            }
        }

        @media (max-width: 640px) {
            .main .block-container {
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }

            .hero-title {
                font-size: 1.56rem;
            }

            .hero-pill {
                width: 100%;
                border-radius: 14px;
            }

            .stat-value {
                font-size: 1.42rem;
            }

            .stTabs [data-baseweb="tab"] {
                flex: 1 1 100%;
                min-width: 100%;
            }

            div[data-baseweb="select"] > div {
                min-height: 50px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_compact_number(value: float | int) -> str:
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def format_decimal(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def section_header(icon: str, eyebrow: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-kicker">{icon} {eyebrow}</div>
            <div class="section-title">{title}</div>
            <div class="section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_divider() -> None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def stat_card(label: str, value: str, subtext: str, accent_class: str = "") -> str:
    extra_class = f" {accent_class}" if accent_class else ""
    return (
        f'<div class="stat-card{extra_class}">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-sub">{subtext}</div>'
        "</div>"
    )


def style_figure(fig: go.Figure, *, height: int = 360, legend_title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 12, "r": 12, "t": 44, "b": 84},
        font={"family": "IBM Plex Sans, sans-serif", "color": "#E6EDF7", "size": 13},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "title": {"text": legend_title or ""},
            "font": {"size": 11},
        },
        title={"font": {"family": "Space Grotesk, sans-serif", "size": 18}},
        hoverlabel={"bgcolor": "#0F1C2F", "bordercolor": "rgba(110, 231, 249, 0.28)"},
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(126, 148, 177, 0.10)",
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont={"color": "#9FB2CC"},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(126, 148, 177, 0.10)",
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont={"color": "#9FB2CC"},
    )
    return fig


def build_color_map(categories: list[str]) -> dict[str, str]:
    return {category: PALETTE[index % len(PALETTE)] for index, category in enumerate(categories)}


def filter_frame(df: pd.DataFrame, categories: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    return df[df["category"].isin(categories) & df["date"].between(start_date, end_date)].copy()


def resolve_date_window(
    time_range: str,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if time_range == "Last 30 days":
        start = max_date - pd.DateOffset(months=1)
    elif time_range == "Year to date":
        start = pd.Timestamp(year=max_date.year, month=1, day=1)
    elif time_range == "Last 2 years":
        start = max_date - pd.DateOffset(years=2)
    else:
        start = min_date
    return max(start.normalize(), min_date.normalize()), max_date.normalize()


def format_date_span(start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
    return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"


def build_sidebar(scoring: pd.DataFrame) -> tuple[list[str], pd.Timestamp, pd.Timestamp, str, int, str]:
    categories = sorted(scoring["category"].dropna().unique().tolist())
    min_date = scoring["date"].min().normalize()
    max_date = scoring["date"].max().normalize()

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Control Room</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-copy">Expanded categories split the open-market sample into cleaner buckets such as '
            'US Elections, Basketball, Football / Soccer, Hockey, Crypto Airdrops, and Crypto Corporate.</div>',
            unsafe_allow_html=True,
        )
        selected_categories = st.multiselect(
            "Category scope",
            options=categories,
            default=categories,
        )
        time_range = st.selectbox("Time range", options=TIME_RANGE_OPTIONS, index=3)
        if time_range == "Custom range":
            selected_start, selected_end = st.slider(
                "Custom dates",
                min_value=min_date.date(),
                max_value=max_date.date(),
                value=(min_date.date(), max_date.date()),
            )
            selected_start = pd.Timestamp(selected_start)
            selected_end = pd.Timestamp(selected_end)
        else:
            selected_start, selected_end = resolve_date_window(time_range, min_date, max_date)
            st.caption(f"Showing {format_date_span(selected_start, selected_end)}")
        ranking_sort = st.selectbox("Category ranking", options=list(RANKING_SORT_OPTIONS.keys()), index=0)
        risk_limit = st.select_slider("High-risk rows", options=[10, 25, 50, 100], value=50)
    return selected_categories, pd.Timestamp(selected_start), pd.Timestamp(selected_end), ranking_sort, risk_limit, time_range


def hero_block(
    metrics: dict,
    scoring: pd.DataFrame,
    filtered_latest: pd.DataFrame,
    selected_categories: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    time_range: str,
) -> None:
    latest_date = scoring["date"].max().strftime("%Y-%m-%d")
    first_date = scoring["date"].min().strftime("%Y-%m-%d")
    category_count = filtered_latest["category"].nunique() if not filtered_latest.empty else 0
    market_count = filtered_latest["market_id"].nunique() if not filtered_latest.empty else 0
    current_median_vol = filtered_latest["recent_volatility_7d"].median() if not filtered_latest.empty else 0.0
    current_mean_risk = filtered_latest["risk_score"].mean() if not filtered_latest.empty else 0.0

    left_col, right_col = st.columns([1.65, 1], gap="large")
    with left_col:
        st.markdown(
            f"""
            <div class="hero-card">
                <div class="hero-eyebrow">Data-Rich / Dense Dashboard</div>
                <div class="hero-title">Prediction Market Volatility Lens</div>
                <div class="hero-copy">
                    A dark control surface for comparing volatility risk across long-duration Polymarket contracts.
                    The category taxonomy is expanded beyond the original coarse labels so the dashboard can separate
                    US Elections, Football / Soccer, Basketball, Hockey, Crypto Airdrops, and Crypto Corporate setups.
                </div>
                <div class="hero-pill-row">
                    <div class="hero-pill">Tracked markets: {market_count}</div>
                    <div class="hero-pill">Active categories: {category_count}</div>
                    <div class="hero-pill">Median current vol: {current_median_vol:.4f}</div>
                    <div class="hero-pill">Mean risk score: {current_mean_risk:.2f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_col:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="section-kicker">Coverage Snapshot</div>
                <div class="section-title">Sample and model context</div>
                <div class="info-grid">
                    <div>
                        <div class="info-label">Training rows</div>
                        <div class="info-value">{format_compact_number(metrics['n_rows'])}</div>
                    </div>
                    <div>
                        <div class="info-label">OLS adj. R²</div>
                        <div class="info-value">{metrics['ols_adj_r2']:.3f}</div>
                    </div>
                    <div>
                        <div class="info-label">RF mean R²</div>
                        <div class="info-value">{metrics['rf_mean_r2']:.3f}</div>
                    </div>
                    <div>
                        <div class="info-label">Date span</div>
                        <div class="info-value">{first_date} to {latest_date}</div>
                    </div>
                </div>
                <div class="section-copy" style="margin-top: 0.95rem;">
                    Current filter: {len(selected_categories)} category buckets in scope. Category importance share in the
                    random forest is {metrics['category_importance_share']:.1%}. Time range: {time_range}
                    ({format_date_span(start_date, end_date)}).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def summary_cards(
    filtered_scoring: pd.DataFrame,
    filtered_latest: pd.DataFrame,
    metrics: dict,
) -> None:
    if filtered_scoring.empty or filtered_latest.empty:
        return

    hottest_category = (
        filtered_scoring.groupby("category")["rolling_volatility_7d"].median().sort_values(ascending=False).index[0]
    )
    avg_days_to_expiry = filtered_latest["time_to_expiry_days"].mean()
    top_risk_market = filtered_latest.sort_values("risk_score", ascending=False).iloc[0]["title"]
    cards = [
        ("Filtered markets", format_compact_number(filtered_latest["market_id"].nunique()), f"{filtered_latest['category'].nunique()} category buckets", ""),
        ("Current risk ceiling", f"{filtered_latest['risk_score'].max():.2f}", top_risk_market[:42], "warm"),
        ("Category hotspot", hottest_category, "Highest median realized volatility", "green"),
        ("Expiry profile", f"{avg_days_to_expiry:.0f}d", f"Reference category: {metrics['ols_reference_category']}", ""),
    ]

    first_row = st.columns(2, gap="large")
    second_row = st.columns(2, gap="large")
    for column, (label, value, subtext, accent) in zip(first_row + second_row, cards):
        with column:
            st.markdown(stat_card(label, value, subtext, accent), unsafe_allow_html=True)


def render_category_tab(
    filtered_scoring: pd.DataFrame,
    color_map: dict[str, str],
    ranking_sort: str,
) -> None:
    section_header("◈", "Category structure", "Volatility by expanded category", "The category lens is now split into finer buckets so the distribution comparison is more informative than Politics versus Sports versus Crypto.")
    if filtered_scoring.empty:
        st.info("No data for the current filter selection.")
        return

    ranking_df = (
        filtered_scoring.groupby("category", as_index=False)
        .agg(
            median_volatility=("rolling_volatility_7d", "median"),
            mean_volatility=("rolling_volatility_7d", "mean"),
            market_count=("market_id", "nunique"),
            observation_count=("market_id", "size"),
            avg_volume=("volume", "mean"),
        )
        .sort_values(RANKING_SORT_OPTIONS[ranking_sort], ascending=False)
    )
    order = ranking_df["category"].tolist()

    top_left, top_right = st.columns([1.1, 0.9], gap="large")
    with top_left:
        fig = px.bar(
            ranking_df,
            x="category",
            y="median_volatility",
            color="category",
            color_discrete_map=color_map,
            category_orders={"category": order},
            title="Median 7-day volatility ranking",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Median 7d volatility")
        st.plotly_chart(style_figure(fig, height=390), use_container_width=True, config=PLOTLY_CONFIG)

    with top_right:
        fig = px.box(
            filtered_scoring,
            x="category",
            y="rolling_volatility_7d",
            color="category",
            color_discrete_map=color_map,
            category_orders={"category": order},
            points=False,
            title="Distribution spread by category",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="7d volatility")
        st.plotly_chart(style_figure(fig, height=390), use_container_width=True, config=PLOTLY_CONFIG)

    bottom_left, bottom_right = st.columns([1.15, 0.85], gap="large")
    with bottom_left:
        st.dataframe(
            ranking_df.assign(
                median_volatility=lambda df: df["median_volatility"].round(4),
                mean_volatility=lambda df: df["mean_volatility"].round(4),
                avg_volume=lambda df: df["avg_volume"].map(format_compact_number),
            ),
            width="stretch",
            hide_index=True,
            height=340,
        )

    with bottom_right:
        mix_df = ranking_df.sort_values("observation_count", ascending=False)
        fig = px.bar(
            mix_df,
            x="observation_count",
            y="category",
            orientation="h",
            color="category",
            color_discrete_map=color_map,
            title="Observation depth by category",
        )
        fig.update_layout(showlegend=False, xaxis_title="Daily observations", yaxis_title="")
        st.plotly_chart(style_figure(fig, height=340), use_container_width=True, config=PLOTLY_CONFIG)


def render_time_tab(
    filtered_scoring: pd.DataFrame,
    filtered_analytical: pd.DataFrame,
    color_map: dict[str, str],
) -> None:
    section_header("◆", "Time structure", "Volatility over time and regime shape", "Average volatility is plotted through time, then linked back to expiry distance and probability level so you can see whether the same categories are simply close to resolution or structurally noisier.")
    if filtered_scoring.empty or filtered_analytical.empty:
        st.info("No data for the current filter selection.")
        return

    time_df = (
        filtered_scoring.groupby(["date", "category"], as_index=False)["rolling_volatility_7d"]
        .mean()
        .rename(columns={"rolling_volatility_7d": "avg_rolling_volatility_7d"})
    )
    latest_snapshot = time_df.sort_values("date").groupby("category", as_index=False).tail(1)
    sample = filtered_analytical.sample(min(len(filtered_analytical), 5000), random_state=42)

    top_left, top_right = st.columns([1.35, 0.65], gap="large")
    with top_left:
        fig = px.line(
            time_df,
            x="date",
            y="avg_rolling_volatility_7d",
            color="category",
            color_discrete_map=color_map,
            title="Average category volatility over calendar time",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Average 7d volatility")
        st.plotly_chart(style_figure(fig, height=390), use_container_width=True, config=PLOTLY_CONFIG)

    with top_right:
        fig = px.bar(
            latest_snapshot.sort_values("avg_rolling_volatility_7d", ascending=True),
            x="avg_rolling_volatility_7d",
            y="category",
            orientation="h",
            color="category",
            color_discrete_map=color_map,
            title="Latest average volatility snapshot",
        )
        fig.update_layout(showlegend=False, xaxis_title="Current average 7d volatility", yaxis_title="")
        st.plotly_chart(style_figure(fig, height=390), use_container_width=True, config=PLOTLY_CONFIG)

    bottom_left, bottom_right = st.columns(2, gap="large")
    with bottom_left:
        fig = px.scatter(
            sample,
            x="time_to_expiry_days",
            y="rolling_volatility_7d",
            color="category",
            color_discrete_map=color_map,
            opacity=0.28,
            trendline="lowess",
            title="Volatility versus time to expiry",
        )
        fig.update_traces(marker={"size": 7})
        fig.update_layout(xaxis_title="Days to expiry", yaxis_title="7d volatility")
        st.plotly_chart(style_figure(fig, height=360), use_container_width=True, config=PLOTLY_CONFIG)

    with bottom_right:
        fig = px.scatter(
            sample,
            x="prob_level",
            y="rolling_volatility_7d",
            color="category",
            color_discrete_map=color_map,
            opacity=0.28,
            trendline="lowess",
            title="Volatility versus implied probability",
        )
        fig.update_traces(marker={"size": 7})
        fig.update_layout(xaxis_title="Implied probability", yaxis_title="7d volatility")
        st.plotly_chart(style_figure(fig, height=360), use_container_width=True, config=PLOTLY_CONFIG)


def render_risk_tab(
    filtered_latest: pd.DataFrame,
    color_map: dict[str, str],
    risk_limit: int,
) -> None:
    section_header("▣", "Risk surface", "Highest-risk markets right now", "The risk table is anchored to the current predicted forward-volatility score. The right-side chart shows whether the current top end is concentrated in one bucket or distributed across several setups.")
    active_latest = filtered_latest[filtered_latest["active"] == True].copy()
    if active_latest.empty:
        st.info("No active markets match the current filter selection.")
        return

    top_risk = active_latest.sort_values("risk_score", ascending=False).head(risk_limit)
    risk_mix = (
        active_latest.groupby("category", as_index=False)
        .agg(avg_risk_score=("risk_score", "mean"), top_decile_count=("risk_score", lambda s: int((s >= s.quantile(0.9)).sum())))
        .sort_values("avg_risk_score", ascending=False)
    )

    top_left, top_right = st.columns([1.25, 0.75], gap="large")
    with top_left:
        st.dataframe(
            top_risk[
                [
                    "title",
                    "category",
                    "implied_probability",
                    "risk_score",
                    "time_to_expiry_days",
                    "recent_volatility_7d",
                ]
            ]
            .rename(
                columns={
                    "title": "market_title",
                    "implied_probability": "current_probability",
                    "time_to_expiry_days": "days_to_expiry",
                    "recent_volatility_7d": "current_volatility",
                }
            )
            .round(
                {
                    "current_probability": 3,
                    "risk_score": 2,
                    "days_to_expiry": 0,
                    "current_volatility": 4,
                }
            ),
            width="stretch",
            hide_index=True,
            height=400,
        )

    with top_right:
        fig = px.bar(
            risk_mix.sort_values("avg_risk_score", ascending=True),
            x="avg_risk_score",
            y="category",
            orientation="h",
            color="category",
            color_discrete_map=color_map,
            title="Average risk score by category",
        )
        fig.update_layout(showlegend=False, xaxis_title="Average risk score", yaxis_title="")
        st.plotly_chart(style_figure(fig, height=400), use_container_width=True, config=PLOTLY_CONFIG)

    fig = px.scatter(
        active_latest,
        x="implied_probability",
        y="risk_score",
        size="recent_volatility_7d",
        color="category",
        color_discrete_map=color_map,
        hover_name="title",
        title="Risk score against current probability level",
    )
    fig.update_layout(xaxis_title="Current probability", yaxis_title="Risk score")
    st.plotly_chart(style_figure(fig, height=360), use_container_width=True, config=PLOTLY_CONFIG)


def render_feature_tab(
    importances: pd.DataFrame,
    coefficients: pd.DataFrame,
    metrics: dict,
) -> None:
    section_header("✦", "Model diagnostics", "Feature importance and category coefficients", "This view keeps the model logic readable: non-linear feature importance on the left, category-specific OLS effects on the right, with the control-variable fit metrics summarized above.")

    category_coefficients = coefficients[
        coefficients["term"].str.contains("C(category", regex=False, na=False)
    ].copy()
    category_coefficients["direction"] = np.where(category_coefficients["coef"] >= 0, "Positive", "Negative")
    top_left, top_right = st.columns([0.95, 1.05], gap="large")

    with top_left:
        fig = px.bar(
            importances.sort_values("importance", ascending=True),
            x="importance",
            y="feature_group",
            orientation="h",
            color="importance",
            color_continuous_scale=["#112033", "#6EE7F9"],
            title="Random forest feature importance",
        )
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Importance share", yaxis_title="")
        st.plotly_chart(style_figure(fig, height=420), use_container_width=True, config=PLOTLY_CONFIG)

    with top_right:
        if category_coefficients.empty:
            st.info("No category coefficients are available.")
        else:
            fig = px.bar(
                category_coefficients.sort_values("coef"),
                x="coef",
                y="term",
                orientation="h",
                color="direction",
                color_discrete_map={"Positive": "#7EE081", "Negative": "#F78FB3"},
                title=f"OLS category coefficients vs {metrics['ols_reference_category']}",
            )
            fig.update_layout(xaxis_title="Coefficient", yaxis_title="", legend_title="")
            st.plotly_chart(style_figure(fig, height=420), use_container_width=True, config=PLOTLY_CONFIG)

    metric_row_one = st.columns(2, gap="large")
    metric_row_two = st.columns(2, gap="large")
    metric_row_one[0].metric("OLS adjusted R²", f"{metrics['ols_adj_r2']:.3f}", delta=f"R² {metrics['ols_r2']:.3f}")
    metric_row_one[1].metric("RF mean CV R²", f"{metrics['rf_mean_r2']:.3f}", delta=f"sd {metrics['rf_std_r2']:.3f}")
    metric_row_two[0].metric(
        "Top predictor",
        str(metrics["top_feature"]).replace("_", " ").title(),
        delta=str(metrics["second_feature"]).replace("_", " ").title(),
    )
    metric_row_two[1].metric(
        "Category share",
        f"{metrics['category_importance_share']:.1%}",
        delta=f"{metrics['significant_category_terms']} sig. terms",
    )

    st.markdown(
        f"""
        <div class="section-copy">
            The current model fit is driven mostly by market structure rather than taxonomy alone. Bid-ask spread,
            recent volatility, probability level, and volume explain most of the predictive lift, while category still
            contributes signal through the OLS category terms and a smaller aggregate random-forest share.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_lookup_tab(
    filtered_scoring: pd.DataFrame,
    filtered_latest: pd.DataFrame,
    color_map: dict[str, str],
) -> None:
    section_header("⬢", "Single-market view", "Risk lookup and probability path", "Use this to pull one market out of the broader surface and inspect how its probability path and realized volatility evolved into the current score.")
    lookup_df = (
        filtered_latest[filtered_latest["active"] == True]
        .sort_values(["title", "market_id"])
        .copy()
    )
    if lookup_df.empty:
        st.info("No active markets are available for lookup.")
        return

    lookup_df["market_label"] = lookup_df["title"] + " | " + lookup_df["market_id"].astype(str)
    selector_col, summary_col = st.columns([1.1, 0.9], gap="large")
    with selector_col:
        selected_label = st.selectbox("Market selection", options=lookup_df["market_label"].tolist())
    selected_market = lookup_df.loc[lookup_df["market_label"] == selected_label].iloc[0]
    history = filtered_scoring[filtered_scoring["market_id"] == selected_market["market_id"]].sort_values("date")
    probability_delta = history["implied_probability"].iloc[-1] - history["implied_probability"].iloc[max(0, len(history) - 8)]
    volatility_delta = history["recent_volatility_7d"].iloc[-1] - history["recent_volatility_7d"].median()

    with summary_col:
        summary_row_one = st.columns(2, gap="small")
        summary_row_two = st.columns(2, gap="small")
        summary_row_one[0].metric(
            "Risk score",
            f"{selected_market['risk_score']:.2f}",
            delta=f"{selected_market['predicted_future_volatility_7d']:.4f} fwd vol",
        )
        summary_row_one[1].metric(
            "Category",
            selected_market["category"],
            delta=f"{selected_market.get('category_base', selected_market['category'])}",
        )
        summary_row_two[0].metric(
            "Probability",
            f"{selected_market['implied_probability']:.3f}",
            delta=f"{probability_delta:+.3f} over 7d",
        )
        summary_row_two[1].metric(
            "Current vol",
            f"{selected_market['recent_volatility_7d']:.4f}",
            delta=f"{volatility_delta:+.4f} vs median",
        )

    chart_col, detail_col = st.columns([1.35, 0.65], gap="large")
    with chart_col:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["implied_probability"],
                mode="lines",
                name="Probability",
                line={"color": color_map.get(selected_market["category"], "#6EE7F9"), "width": 2.6},
                hovertemplate="%{x|%Y-%m-%d}<br>Probability=%{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["recent_volatility_7d"],
                mode="lines",
                name="Recent 7d vol",
                line={"color": "#F6C177", "width": 2, "dash": "dot"},
                yaxis="y2",
                hovertemplate="%{x|%Y-%m-%d}<br>7d vol=%{y:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Probability path with realized volatility overlay",
            yaxis={"title": "Probability"},
            yaxis2={"title": "7d volatility", "overlaying": "y", "side": "right", "showgrid": False},
            xaxis_title="",
            legend_title="",
        )
        st.plotly_chart(style_figure(fig, height=420), use_container_width=True, config=PLOTLY_CONFIG)

    with detail_col:
        detail_rows = pd.DataFrame(
            [
                ("Market", selected_market["title"]),
                ("Expanded category", selected_market["category"]),
                ("Base category", selected_market.get("category_base", selected_market["category"])),
                ("Days to expiry", f"{selected_market['time_to_expiry_days']:.0f}"),
                ("Avg volume", format_compact_number(selected_market["volume"])),
                ("Bid / ask spread", format_decimal(float(selected_market["bid_ask_spread"]), 4)),
            ],
            columns=["Field", "Value"],
        )
        st.dataframe(detail_rows, width="stretch", hide_index=True, height=320)

    recent_rows = history.tail(12)[["date", "implied_probability", "prob_return", "recent_volatility_7d"]].copy()
    recent_rows["date"] = recent_rows["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        recent_rows.round(
            {
                "implied_probability": 3,
                "prob_return": 4,
                "recent_volatility_7d": 4,
            }
        ),
        width="stretch",
        hide_index=True,
        height=260,
    )


def main() -> None:
    st.set_page_config(
        page_title="Volatility Lens | Prediction Markets",
        page_icon="📉",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    data = load_data()
    analytical = data["analytical"]
    scoring = data["scoring"]
    latest_scores = data["latest_scores"]
    importances = data["importances"]
    coefficients = data["coefficients"]
    metrics = data["metrics"]

    categories, start_date, end_date, ranking_sort, risk_limit, time_range = build_sidebar(scoring)
    filtered_scoring = filter_frame(scoring, categories, start_date, end_date)
    filtered_analytical = filter_frame(analytical, categories, start_date, end_date)
    filtered_latest = latest_scores[
        latest_scores["category"].isin(categories)
        & latest_scores["date"].between(start_date, end_date)
    ].copy()

    color_map = build_color_map(sorted(scoring["category"].dropna().unique().tolist()))
    hero_block(metrics, scoring, filtered_latest, categories, start_date, end_date, time_range)
    section_divider()
    summary_cards(filtered_scoring, filtered_latest, metrics)
    section_divider()

    category_tab, time_tab, risk_tab, feature_tab, lookup_tab = st.tabs(
        [
            "Category Rankings",
            "Volatility Over Time",
            "High-Risk Markets",
            "Feature Importance",
            "Risk Score Lookup",
        ]
    )

    with category_tab:
        render_category_tab(filtered_scoring, color_map, ranking_sort)
    with time_tab:
        render_time_tab(filtered_scoring, filtered_analytical, color_map)
    with risk_tab:
        render_risk_tab(filtered_latest, color_map, risk_limit)
    with feature_tab:
        render_feature_tab(importances, coefficients, metrics)
    with lookup_tab:
        render_lookup_tab(filtered_scoring, filtered_latest, color_map)


if __name__ == "__main__":
    main()
