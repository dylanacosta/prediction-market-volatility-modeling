from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; prediction-market-volatility-modeling/1.0)",
}


def parse_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return loaded if isinstance(loaded, list) else []
    return []


def normalize_category(raw_category: str | None, title: str, tag_labels: list[str]) -> str | None:
    text = " ".join(
        part for part in [raw_category or "", title or "", " ".join(tag_labels)] if part
    ).lower()
    if not text.strip():
        return None

    category_rules = [
        (
            "Politics",
            [
                "politic",
                "election",
                "president",
                "senate",
                "house",
                "governor",
                "congress",
                "democrat",
                "republican",
                "white house",
            ],
        ),
        (
            "Sports",
            [
                "sports",
                "nba",
                "nfl",
                "mlb",
                "nhl",
                "ncaa",
                "soccer",
                "football",
                "baseball",
                "basketball",
                "tennis",
                "golf",
                "f1",
                "ufc",
                "boxing",
                "olympic",
                "world cup",
            ],
        ),
        (
            "Crypto",
            [
                "crypto",
                "bitcoin",
                "btc",
                "ethereum",
                "eth",
                "solana",
                "doge",
                "memecoin",
                "token",
                "airdrop",
                "blockchain",
            ],
        ),
        (
            "Economics/Macro",
            [
                "econom",
                "macro",
                "inflation",
                "cpi",
                "gdp",
                "fed",
                "fomc",
                "interest rate",
                "recession",
                "treasury",
                "jobs report",
                "nonfarm payroll",
                "oil",
            ],
        ),
        (
            "Weather",
            [
                "weather",
                "temperature",
                "snow",
                "rain",
                "storm",
                "hurricane",
                "tornado",
                "forecast",
                "climate",
            ],
        ),
        (
            "Pop Culture",
            [
                "pop culture",
                "movie",
                "album",
                "box office",
                "oscar",
                "grammy",
                "celebrity",
                "tv",
                "series",
                "rihanna",
                "taylor swift",
                "super bowl halftime",
            ],
        ),
        (
            "Science/Tech",
            [
                "technology",
                "tech",
                "science",
                "ai",
                "artificial intelligence",
                "openai",
                "spacex",
                "nasa",
                "launch",
                "robot",
                "chip",
                "semiconductor",
            ],
        ),
    ]

    for normalized, keywords in category_rules:
        if any(keyword in text for keyword in keywords):
            return normalized

    raw = (raw_category or "").strip()
    return raw or "Other"


@dataclass
class RateLimiter:
    sleep_seconds: float
    last_request_at: float = 0.0

    def wait(self) -> None:
        if self.sleep_seconds <= 0:
            return
        elapsed = time.monotonic() - self.last_request_at
        remaining = self.sleep_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def mark(self) -> None:
        self.last_request_at = time.monotonic()


def request_json(
    session: requests.Session,
    url: str,
    rate_limiter: RateLimiter,
    *,
    params: dict[str, Any] | None = None,
    max_retries: int = 5,
) -> Any:
    backoff = max(rate_limiter.sleep_seconds, 0.5)
    for attempt in range(max_retries):
        rate_limiter.wait()
        response = session.get(url, params=params, timeout=45)
        rate_limiter.mark()
        if response.status_code in {429, 500, 502, 503, 504}:
            if attempt == max_retries - 1:
                response.raise_for_status()
            time.sleep(backoff * (2**attempt))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Request failed after retries: {url}")


def choose_outcome_index(outcomes: list[str]) -> int:
    lowered = [outcome.lower() for outcome in outcomes]
    if "yes" in lowered:
        return lowered.index("yes")
    return 0


def is_binary_market(market: dict[str, Any]) -> bool:
    outcomes = parse_json_list(market.get("outcomes"))
    token_ids = parse_json_list(market.get("clobTokenIds"))
    return len(outcomes) == 2 and len(token_ids) == 2 and all(token_ids)


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def flatten_event_markets(event: dict[str, Any]) -> list[dict[str, Any]]:
    title = event.get("title", "")
    tag_labels = [tag.get("label", "") for tag in event.get("tags", []) if isinstance(tag, dict)]
    event_category = event.get("category")
    markets = []
    for market in event.get("markets", []):
        if not is_binary_market(market):
            continue
        outcomes = parse_json_list(market.get("outcomes"))
        token_ids = parse_json_list(market.get("clobTokenIds"))
        outcome_index = choose_outcome_index(outcomes)
        category = normalize_category(event_category or market.get("category"), market.get("question", title), tag_labels)
        if category is None:
            continue

        best_bid = parse_float(market.get("bestBid"))
        best_ask = parse_float(market.get("bestAsk"))
        spread = parse_float(market.get("spread"))
        if spread is None and best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid

        markets.append(
            {
                "market_id": str(market.get("id")),
                "event_id": str(event.get("id")),
                "event_title": title,
                "title": market.get("question", title),
                "source_category": event_category or market.get("category"),
                "category": category,
                "tag_labels": ", ".join([tag for tag in tag_labels if tag]),
                "token_id": str(token_ids[outcome_index]),
                "outcome_label": outcomes[outcome_index],
                "volume": parse_float(market.get("volume") or event.get("volume")),
                "liquidity": parse_float(market.get("liquidity") or event.get("liquidity")),
                "bid": best_bid,
                "ask": best_ask,
                "spread": spread,
                "active": bool(market.get("active")),
                "closed": bool(market.get("closed")),
                "start_date": market.get("startDate") or event.get("startDate"),
                "expiry_date": market.get("endDate") or event.get("endDate"),
                "created_at": market.get("createdAt") or event.get("createdAt"),
                "url_slug": market.get("slug"),
            }
        )
    return markets


def fetch_event_pages(
    session: requests.Session,
    rate_limiter: RateLimiter,
    *,
    active: bool | None,
    closed: bool | None,
    candidate_limit: int,
    page_size: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_market_ids: set[str] = set()
    offset = 0

    while len(candidates) < candidate_limit:
        params: dict[str, Any] = {"limit": page_size, "offset": offset}
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()

        payload = request_json(session, f"{GAMMA_BASE_URL}/events", rate_limiter, params=params)
        if not isinstance(payload, list) or not payload:
            break

        for event in payload:
            for market in flatten_event_markets(event):
                market_id = market["market_id"]
                if market_id in seen_market_ids:
                    continue
                candidates.append(market)
                seen_market_ids.add(market_id)
                if len(candidates) >= candidate_limit:
                    break
            if len(candidates) >= candidate_limit:
                break

        if len(payload) < page_size:
            break
        offset += page_size

    return candidates


def sample_balanced_markets(
    candidates: list[dict[str, Any]],
    *,
    max_markets: int,
    per_category_cap: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[row["category"]].append(row)

    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                int(bool(row["active"])),
                row["volume"] or 0.0,
                row["liquidity"] or 0.0,
            ),
            reverse=True,
        )

    category_order = sorted(
        grouped,
        key=lambda category: (category == "Other", -len(grouped[category]), category),
    )
    sampled: list[dict[str, Any]] = []
    taken_by_category: Counter[str] = Counter()

    while len(sampled) < max_markets and any(grouped.values()):
        progress = False
        for category in category_order:
            rows = grouped[category]
            if not rows or taken_by_category[category] >= per_category_cap:
                continue
            sampled.append(rows.pop(0))
            taken_by_category[category] += 1
            progress = True
            if len(sampled) >= max_markets:
                break
        if not progress:
            break

    return sampled


def filter_candidate_markets(
    candidates: list[dict[str, Any]],
    *,
    min_history_days: int,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    filtered = []
    for row in candidates:
        start_dt = parse_iso_datetime(row.get("start_date"))
        expiry_dt = parse_iso_datetime(row.get("expiry_date"))

        if expiry_dt is not None and expiry_dt < now:
            continue
        if start_dt is not None and (now - start_dt).days < min_history_days:
            continue
        if start_dt is not None and expiry_dt is not None and (expiry_dt - start_dt).days < min_history_days:
            continue
        filtered.append(row)
    return filtered


def fetch_price_history(
    session: requests.Session,
    rate_limiter: RateLimiter,
    *,
    market_id: str,
    token_id: str,
    fidelity: int,
) -> list[dict[str, Any]]:
    payload = request_json(
        session,
        f"{CLOB_BASE_URL}/prices-history",
        rate_limiter,
        params={"market": token_id, "interval": "max", "fidelity": fidelity},
    )
    history = payload.get("history", []) if isinstance(payload, dict) else []
    rows: list[dict[str, Any]] = []
    for point in history:
        timestamp = point.get("t")
        price = parse_float(point.get("p"))
        if timestamp is None or price is None:
            continue
        dt = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        rows.append(
            {
                "market_id": market_id,
                "token_id": token_id,
                "date": dt.date().isoformat(),
                "price": price,
            }
        )
    return rows


def ingest_polymarket_data(
    output_dir: Path,
    *,
    max_markets: int,
    candidate_multiplier: int,
    event_page_size: int,
    min_history_days: int,
    sleep_seconds: float,
    fidelity: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    rate_limiter = RateLimiter(sleep_seconds=sleep_seconds)

    candidate_limit = max_markets * candidate_multiplier
    active_candidates = fetch_event_pages(
        session,
        rate_limiter,
        active=True,
        closed=False,
        candidate_limit=candidate_limit,
        page_size=event_page_size,
    )
    combined_candidates = filter_candidate_markets(
        active_candidates,
        min_history_days=min_history_days,
    )
    print(
        f"[info] candidate markets after open-market and age filters: "
        f"{len(combined_candidates):,} from {len(active_candidates):,} fetched"
    )
    per_category_cap = max(60, max_markets // 2)
    selected_markets = sample_balanced_markets(
        combined_candidates,
        max_markets=max_markets,
        per_category_cap=per_category_cap,
    )

    fetched_at = datetime.now(timezone.utc).isoformat()
    for row in selected_markets:
        row["fetched_at"] = fetched_at

    price_rows: list[dict[str, Any]] = []
    retained_markets: list[dict[str, Any]] = []

    for index, market in enumerate(selected_markets, start=1):
        try:
            history = fetch_price_history(
                session,
                rate_limiter,
                market_id=market["market_id"],
                token_id=market["token_id"],
                fidelity=fidelity,
            )
        except requests.HTTPError as exc:
            print(f"[warn] history request failed for {market['market_id']}: {exc}")
            continue

        deduped = {}
        for row in history:
            deduped[row["date"]] = row
        history = list(sorted(deduped.values(), key=lambda row: row["date"]))
        if len(history) < min_history_days:
            continue

        retained_markets.append(market)
        price_rows.extend(history)
        if index % 25 == 0:
            print(f"[info] fetched histories for {index} / {len(selected_markets)} candidate markets")

    markets_df = pd.DataFrame(retained_markets)
    prices_df = pd.DataFrame(price_rows)

    if not markets_df.empty:
        markets_df = markets_df.sort_values(["category", "volume", "market_id"], ascending=[True, False, True])
    if not prices_df.empty:
        prices_df = prices_df.sort_values(["market_id", "date"])

    markets_df.to_csv(output_dir / "raw_markets.csv", index=False)
    prices_df.to_csv(output_dir / "raw_prices.csv", index=False)

    print(f"[done] saved {len(markets_df):,} markets to {output_dir / 'raw_markets.csv'}")
    print(f"[done] saved {len(prices_df):,} price rows to {output_dir / 'raw_prices.csv'}")
    print("[done] category counts:")
    if not markets_df.empty:
        for category, count in markets_df["category"].value_counts().sort_index().items():
            print(f"  - {category}: {count}")
    return markets_df, prices_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest historical Polymarket data for volatility modeling.")
    parser.add_argument("--output-dir", default="data", type=Path)
    parser.add_argument("--max-markets", default=320, type=int)
    parser.add_argument("--candidate-multiplier", default=3, type=int)
    parser.add_argument("--event-page-size", default=100, type=int)
    parser.add_argument("--min-history-days", default=14, type=int)
    parser.add_argument("--sleep-seconds", default=0.35, type=float)
    parser.add_argument("--fidelity", default=1440, type=int, help="History fidelity in minutes.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ingest_polymarket_data(
        output_dir=args.output_dir,
        max_markets=args.max_markets,
        candidate_multiplier=args.candidate_multiplier,
        event_page_size=args.event_page_size,
        min_history_days=args.min_history_days,
        sleep_seconds=args.sleep_seconds,
        fidelity=args.fidelity,
    )


if __name__ == "__main__":
    main()
