from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from taxonomy import expand_market_category


REQUIRED_SCORE_FEATURES = [
    "category",
    "log_volume",
    "bid_ask_spread",
    "time_to_expiry_days",
    "recent_volatility_7d",
    "prob_level",
]


def load_raw_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    markets = pd.read_csv(data_dir / "raw_markets.csv")
    prices = pd.read_csv(data_dir / "raw_prices.csv")
    return markets, prices


def expand_categories(markets_df: pd.DataFrame) -> pd.DataFrame:
    markets_df = markets_df.copy()
    markets_df["category_base"] = markets_df["category"]
    markets_df["category_expanded"] = markets_df.apply(
        lambda row: expand_market_category(
            row.get("category"),
            row.get("title"),
            row.get("tag_labels"),
        ),
        axis=1,
    )
    markets_df["category"] = markets_df["category_expanded"]
    return markets_df


def merge_sparse_categories(markets_df: pd.DataFrame, min_markets_per_category: int) -> pd.DataFrame:
    category_counts = (
        markets_df.groupby("category")["market_id"].nunique().sort_values(ascending=False)
    )
    sparse_categories = category_counts[category_counts < min_markets_per_category].index
    markets_df = markets_df.copy()
    markets_df["category"] = markets_df["category"].where(
        ~markets_df["category"].isin(sparse_categories),
        "Other",
    )
    return markets_df


def prepare_daily_panel(markets_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    markets = markets_df.copy()
    prices = prices_df.copy()
    markets["start_date"] = pd.to_datetime(markets["start_date"], utc=True, errors="coerce").dt.tz_localize(None)
    markets["expiry_date"] = pd.to_datetime(markets["expiry_date"], utc=True, errors="coerce").dt.tz_localize(None)
    prices["date"] = pd.to_datetime(prices["date"], utc=True, errors="coerce").dt.tz_localize(None)

    metadata_columns = [
        "market_id",
        "title",
        "category",
        "category_base",
        "category_expanded",
        "source_category",
        "tag_labels",
        "token_id",
        "outcome_label",
        "volume",
        "liquidity",
        "bid",
        "ask",
        "spread",
        "active",
        "closed",
        "start_date",
        "expiry_date",
        "created_at",
        "url_slug",
    ]
    merged = prices.merge(markets[metadata_columns], on=["market_id", "token_id"], how="left")
    merged = merged.dropna(subset=["date", "price", "category"]).sort_values(["market_id", "date"])
    merged = merged.rename(columns={"price": "implied_probability"})
    merged["implied_probability"] = merged["implied_probability"].clip(0, 1)

    market_lengths = merged.groupby("market_id")["date"].nunique()
    eligible_market_ids = market_lengths[market_lengths >= 14].index
    merged = merged[merged["market_id"].isin(eligible_market_ids)].copy()

    daily_frames = []
    static_columns = [column for column in merged.columns if column not in {"date", "implied_probability"}]

    for _, group in merged.groupby("market_id", sort=False):
        ordered = group.sort_values("date")
        panel = (
            ordered.set_index("date")[["implied_probability"]]
            .resample("D")
            .last()
            .ffill()
            .reset_index()
        )
        for column in static_columns:
            panel[column] = ordered.iloc[0][column]
        daily_frames.append(panel)

    panel_df = pd.concat(daily_frames, ignore_index=True)
    panel_df = panel_df.sort_values(["market_id", "date"])
    return panel_df


def engineer_features(
    data_dir: Path,
    *,
    min_markets_per_category: int,
    rolling_window: int,
    forecast_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    markets_df, prices_df = load_raw_data(data_dir)
    markets_df = expand_categories(markets_df)
    markets_df = merge_sparse_categories(markets_df, min_markets_per_category=min_markets_per_category)
    daily_panel = prepare_daily_panel(markets_df, prices_df)

    daily_panel["log_volume"] = np.log1p(daily_panel["volume"].fillna(0))
    daily_panel["bid_ask_spread"] = (
        daily_panel["ask"].fillna(np.nan) - daily_panel["bid"].fillna(np.nan)
    )
    daily_panel["bid_ask_spread"] = daily_panel["bid_ask_spread"].where(
        daily_panel["bid_ask_spread"].notna(),
        daily_panel["spread"],
    )
    spread_by_category = daily_panel.groupby("category")["bid_ask_spread"].transform("median")
    daily_panel["bid_ask_spread"] = daily_panel["bid_ask_spread"].fillna(spread_by_category).fillna(0)

    daily_panel["time_to_expiry_days"] = (
        daily_panel["expiry_date"] - daily_panel["date"]
    ).dt.days.clip(lower=0)
    daily_panel["market_age_days"] = (
        daily_panel["date"] - daily_panel["start_date"]
    ).dt.days.clip(lower=0)
    daily_panel["prob_level"] = daily_panel["implied_probability"]

    daily_panel["prob_return"] = daily_panel.groupby("market_id")["implied_probability"].diff()
    daily_panel["rolling_volatility_7d"] = (
        daily_panel.groupby("market_id")["prob_return"]
        .transform(lambda series: series.rolling(rolling_window, min_periods=rolling_window).std())
    )
    daily_panel["rolling_volatility_14d"] = (
        daily_panel.groupby("market_id")["prob_return"]
        .transform(lambda series: series.rolling(14, min_periods=14).std())
    )
    daily_panel["future_volatility_7d"] = (
        daily_panel.groupby("market_id")["rolling_volatility_7d"].shift(-forecast_horizon)
    )
    daily_panel["recent_volatility_7d"] = daily_panel["rolling_volatility_7d"]

    for column in ["rolling_volatility_7d", "future_volatility_7d", "recent_volatility_7d", "rolling_volatility_14d"]:
        upper = daily_panel[column].quantile(0.99)
        daily_panel[column] = daily_panel[column].clip(upper=upper)

    scoring_dataset = daily_panel.dropna(subset=REQUIRED_SCORE_FEATURES).copy()
    analytical_dataset = scoring_dataset.dropna(
        subset=["rolling_volatility_7d", "future_volatility_7d", "prob_return"]
    ).copy()
    analytical_dataset = analytical_dataset.sort_values(["date", "market_id"]).reset_index(drop=True)
    scoring_dataset = scoring_dataset.sort_values(["date", "market_id"]).reset_index(drop=True)

    scoring_dataset.to_csv(data_dir / "scoring_dataset.csv", index=False)
    analytical_dataset.to_csv(data_dir / "analytical_dataset.csv", index=False)
    return analytical_dataset, scoring_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build analytical features from raw Polymarket data.")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--min-markets-per-category", default=3, type=int)
    parser.add_argument("--rolling-window", default=7, type=int)
    parser.add_argument("--forecast-horizon", default=7, type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    analytical_dataset, scoring_dataset = engineer_features(
        args.data_dir,
        min_markets_per_category=args.min_markets_per_category,
        rolling_window=args.rolling_window,
        forecast_horizon=args.forecast_horizon,
    )
    print(f"[done] analytical rows: {len(analytical_dataset):,}")
    print(f"[done] scoring rows: {len(scoring_dataset):,}")


if __name__ == "__main__":
    main()
