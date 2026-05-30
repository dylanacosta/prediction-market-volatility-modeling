# Prediction Market Volatility Modeling

This repository is a one-day MVP for modeling volatility risk in prediction markets.

The core research question is:

Can market category predict future probability volatility in prediction markets after controlling for liquidity, volume, probability level, and time to expiry?

The project uses Polymarket data, builds a daily market panel, engineers forward-looking volatility targets, fits both an OLS regression and a random forest, and exposes the results in a Streamlit dashboard plus a Jupyter notebook.

## What is in the repo

- `ingest.py`
  Pulls market metadata and price history from Polymarket Gamma + CLOB endpoints.
- `features.py`
  Builds the daily analytical panel and volatility features.
- `taxonomy.py`
  Expands coarse categories into more specific buckets.
- `model.py`
  Fits the OLS and random forest models and writes dashboard-ready outputs.
- `build_notebook.py`
  Generates `notebooks/analysis.ipynb` from the saved outputs.
- `app.py`
  Streamlit dashboard for category rankings, time-series volatility, high-risk markets, feature importance, and single-market lookup.
- `data/`
  Raw extracts plus derived modeling outputs used by the dashboard.
- `notebooks/analysis.ipynb`
  Notebook version of the analysis and findings.

## Current committed sample

The committed artifacts in `data/` currently reflect this run:

- `160` markets
- `40,678` analytical training rows
- date span: `2025-05-10` to `2026-05-30`
- OLS adjusted R²: `0.686`
- random forest mean time-series CV R²: `0.590`
- OLS reference category: `US Elections`
- aggregated category feature-importance share: `1.1%`

The current expanded category mix in the committed dataset is concentrated in:

- `US Elections`
- `Football / Soccer`
- `Basketball`
- `Hockey`
- `Crypto Airdrops`
- `Crypto Corporate`
- `Other`

## Modeling approach

Each row in the analytical dataset represents one market on one date.

Key engineered fields:

- `implied_probability`
- `prob_return`
- `rolling_volatility_7d`
- `future_volatility_7d`
- `log_volume`
- `bid_ask_spread`
- `time_to_expiry_days`
- `recent_volatility_7d`
- `prob_level`

Primary model:

- OLS regression on `future_volatility_7d`
- predictors: `category`, `log_volume`, `bid_ask_spread`, `time_to_expiry_days`, `recent_volatility_7d`, `prob_level`

Secondary model:

- random forest regressor on the same features
- evaluated with time-series cross-validation

Risk score:

- the dashboard converts the OLS predicted forward volatility into a normalized `0-10` market risk score

## Category expansion

The ingestion layer starts with broad market categories, then `taxonomy.py` expands them into more specific buckets when the title and tag text support it.

Examples:

- `Politics` -> `US Elections`, `Global Politics`
- `Sports` -> `Basketball`, `Hockey`, `Football / Soccer`
- `Crypto` -> `Crypto Airdrops`, `Crypto Corporate`
- additional buckets exist for `Macro / Business`, `Weather`, `Tech / Science`, and `Culture / Gaming`

Sparse buckets are merged back into `Other` during feature engineering.

## Important assumptions and caveats

- This is an MVP, not a production research pipeline.
- The sample is built from live Polymarket public endpoints and is currently concentrated in open long-duration markets.
- Historical bid-ask spread is approximated from the latest observed market-level bid/ask or spread fields because a clean daily spread history is not exposed in the same path as price history.
- Markets with fewer than 14 days of history are dropped.
- Volatility targets are clipped at the 99th percentile to reduce extreme outlier impact.
- Recent volatility is expected to be a strong predictor; the question is whether category still adds signal after controls.

## Setup

Use Python `3.10+`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the full pipeline

1. Ingest raw Polymarket data:

```bash
python ingest.py --output-dir data --max-markets 320 --min-history-days 14 --sleep-seconds 0.35
```

2. Build analytical features:

```bash
python features.py --data-dir data --min-markets-per-category 3 --rolling-window 7 --forecast-horizon 7
```

3. Train models and write outputs:

```bash
python model.py --data-dir data
```

4. Regenerate the notebook:

```bash
python build_notebook.py
```

5. Launch the dashboard:

```bash
streamlit run app.py
```

If you only want to inspect the committed MVP, `data/` is already populated, so you can go straight to:

```bash
streamlit run app.py
```

## Dashboard

The Streamlit app includes five views:

1. `Category Rankings`
2. `Volatility Over Time`
3. `High-Risk Markets`
4. `Feature Importance`
5. `Risk Score Lookup`

It also includes:

- category filters
- time-range presets
- current risk rankings
- expanded-category comparisons
- Plotly-based dark-theme charts

## Data outputs

Main artifacts written to `data/`:

- `raw_markets.csv`
- `raw_prices.csv`
- `analytical_dataset.csv`
- `scoring_dataset.csv`
- `latest_market_scores.csv`
- `category_rankings.csv`
- `volatility_over_time.csv`
- `correlation_matrix.csv`
- `ols_coefficients.csv`
- `rf_feature_importances.csv`
- `model_metrics.json`

## Notebook

`notebooks/analysis.ipynb` covers:

- category coverage
- volatility distributions
- volatility over time
- correlation analysis
- volatility vs expiry
- volatility vs probability level
- OLS coefficient review
- random forest feature importance
- a short written interpretation of the findings

## Notes on rate limiting

`ingest.py` includes a simple rate limiter plus retry/backoff behavior for `429`, `500`, `502`, `503`, and `504` responses.

Relevant knobs:

- `--sleep-seconds`
- `--max-markets`
- `--candidate-multiplier`
- `--event-page-size`
- `--fidelity`

If Polymarket throttles aggressively, lower `--max-markets` or increase `--sleep-seconds`.
