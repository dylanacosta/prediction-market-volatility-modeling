from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MODEL_FEATURES = [
    "category",
    "log_volume",
    "bid_ask_spread",
    "time_to_expiry_days",
    "recent_volatility_7d",
    "prob_level",
]
TARGET = "future_volatility_7d"


def load_datasets(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    analytical = pd.read_csv(data_dir / "analytical_dataset.csv", parse_dates=["date"])
    scoring = pd.read_csv(data_dir / "scoring_dataset.csv", parse_dates=["date"])
    return analytical, scoring


def choose_reference_category(df: pd.DataFrame, preferred: str | None = "Politics") -> str:
    for candidate in [preferred, "US Elections", "Politics"]:
        if candidate and candidate in set(df["category"]):
            return candidate
    return df["category"].value_counts().idxmax()


def fit_ols_model(df: pd.DataFrame, reference_category: str) -> tuple[Any, pd.DataFrame]:
    safe_reference = reference_category.replace("'", "\\'")
    formula = (
        f"{TARGET} ~ C(category, Treatment(reference='{safe_reference}'))"
        " + log_volume + bid_ask_spread + time_to_expiry_days + recent_volatility_7d + prob_level"
    )
    model = smf.ols(formula=formula, data=df).fit()
    coefficient_table = (
        pd.DataFrame(
            {
                "term": model.params.index,
                "coef": model.params.values,
                "std_err": model.bse.values,
                "t_value": model.tvalues.values,
                "p_value": model.pvalues.values,
                "conf_low": model.conf_int()[0].values,
                "conf_high": model.conf_int()[1].values,
            }
        )
        .sort_values("term")
        .reset_index(drop=True)
    )
    return model, coefficient_table


def fit_random_forest(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, list[float]]:
    numeric_features = [feature for feature in MODEL_FEATURES if feature != "category"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["category"]),
            ("numeric", "passthrough", numeric_features),
        ]
    )
    forest = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=5,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", forest)])

    ordered = df.sort_values("date").reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=5)
    scores: list[float] = []
    for train_index, test_index in splitter.split(ordered):
        train_df = ordered.iloc[train_index]
        test_df = ordered.iloc[test_index]
        pipeline.fit(train_df[MODEL_FEATURES], train_df[TARGET])
        predictions = pipeline.predict(test_df[MODEL_FEATURES])
        scores.append(r2_score(test_df[TARGET], predictions))

    pipeline.fit(ordered[MODEL_FEATURES], ordered[TARGET])
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    feature_importances = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    feature_importances["feature_group"] = feature_importances["feature"].apply(
        lambda name: "category" if name.startswith("category__category_") else name.split("__", 1)[-1]
    )
    grouped_importances = (
        feature_importances.groupby("feature_group", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return pipeline, grouped_importances, scores


def normalize_scores(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or pd.isna(max_value) or np.isclose(max_value, min_value):
        return pd.Series(np.full(len(series), 5.0), index=series.index)
    return ((series - min_value) / (max_value - min_value) * 10).clip(0, 10)


def build_summary_outputs(
    analytical: pd.DataFrame,
    scoring: pd.DataFrame,
    ols_model: Any,
    grouped_importances: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scoring = scoring.copy()
    scoring["predicted_future_volatility_7d"] = ols_model.predict(scoring)

    latest_scores = (
        scoring.sort_values("date")
        .groupby("market_id", as_index=False)
        .tail(1)
        .copy()
        .sort_values("predicted_future_volatility_7d", ascending=False)
    )
    latest_scores["risk_score"] = normalize_scores(latest_scores["predicted_future_volatility_7d"]).round(2)

    category_rankings = (
        scoring.groupby("category", as_index=False)
        .agg(
            median_volatility=("rolling_volatility_7d", "median"),
            mean_volatility=("rolling_volatility_7d", "mean"),
            market_count=("market_id", "nunique"),
            observation_count=("market_id", "size"),
            avg_volume=("volume", "mean"),
        )
        .sort_values("median_volatility", ascending=False)
    )

    volatility_over_time = (
        scoring.groupby(["date", "category"], as_index=False)["rolling_volatility_7d"]
        .mean()
        .rename(columns={"rolling_volatility_7d": "avg_rolling_volatility_7d"})
    )

    corr_columns = [
        "rolling_volatility_7d",
        "future_volatility_7d",
        "log_volume",
        "bid_ask_spread",
        "time_to_expiry_days",
        "prob_level",
        "recent_volatility_7d",
    ]
    correlation_matrix = analytical[corr_columns].corr(numeric_only=True)
    return latest_scores, category_rankings, volatility_over_time, correlation_matrix


def train_and_save_outputs(data_dir: Path) -> dict[str, float | str]:
    analytical, scoring = load_datasets(data_dir)
    reference_category = choose_reference_category(analytical)
    ols_model, coefficient_table = fit_ols_model(analytical, reference_category)
    _, grouped_importances, cv_scores = fit_random_forest(analytical)
    latest_scores, category_rankings, volatility_over_time, correlation_matrix = build_summary_outputs(
        analytical,
        scoring,
        ols_model,
        grouped_importances,
    )

    coefficient_table.to_csv(data_dir / "ols_coefficients.csv", index=False)
    grouped_importances.to_csv(data_dir / "rf_feature_importances.csv", index=False)
    latest_scores.to_csv(data_dir / "latest_market_scores.csv", index=False)
    category_rankings.to_csv(data_dir / "category_rankings.csv", index=False)
    volatility_over_time.to_csv(data_dir / "volatility_over_time.csv", index=False)
    correlation_matrix.to_csv(data_dir / "correlation_matrix.csv")

    category_importance = grouped_importances.loc[
        grouped_importances["feature_group"] == "category", "importance"
    ]
    top_feature = grouped_importances.iloc[0]["feature_group"]
    second_feature = grouped_importances.iloc[1]["feature_group"] if len(grouped_importances) > 1 else top_feature
    metrics = {
        "n_rows": int(len(analytical)),
        "n_markets": int(analytical["market_id"].nunique()),
        "ols_adj_r2": float(ols_model.rsquared_adj),
        "ols_r2": float(ols_model.rsquared),
        "ols_reference_category": reference_category,
        "rf_mean_r2": float(np.mean(cv_scores)),
        "rf_std_r2": float(np.std(cv_scores)),
        "category_importance_share": float(category_importance.iloc[0]) if not category_importance.empty else 0.0,
        "top_feature": str(top_feature),
        "second_feature": str(second_feature),
        "significant_category_terms": int(
            coefficient_table["term"].str.contains("C\\(category").fillna(False)
            .mul(coefficient_table["p_value"] < 0.05)
            .sum()
        ),
    }

    with open(data_dir / "model_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train OLS and random forest models for volatility risk.")
    parser.add_argument("--data-dir", default="data", type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = train_and_save_outputs(args.data_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
