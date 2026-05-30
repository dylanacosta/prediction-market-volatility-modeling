from __future__ import annotations


def _normalize_text(*parts: object) -> str:
    text = " ".join("" if part is None else str(part) for part in parts)
    return " ".join(text.lower().split())


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def expand_market_category(
    base_category: str | None,
    title: str | None,
    tag_labels: str | None,
) -> str:
    text = _normalize_text(base_category, title, tag_labels)

    if _contains_any(text, ["airdrop", "pre-market", "premarket"]):
        return "Crypto Airdrops"

    if _contains_any(
        text,
        [
            "microstrategy",
            "kraken ipo",
            "ipo",
            "crypto treasury",
            "stocks",
            "public listing",
        ],
    ):
        return "Crypto Corporate"

    if _contains_any(
        text,
        [
            "democratic presidential nomination",
            "republican presidential nomination",
            "u.s. presidential election",
            "us presidential election",
            "u.s. election",
            "us election",
            "united states",
            "primaries",
            "white house",
        ],
    ):
        return "US Elections"

    if _contains_any(
        text,
        [
            "global elections",
            "world elections",
            "uk election",
            "macron",
            "starmer",
            "military clash",
            "prime minister",
            "president of france",
            "china x india",
        ],
    ):
        return "Global Politics"

    if _contains_any(text, ["nba", "wnba", "ncaa basketball", "basketball", "finals"]):
        return "Basketball"

    if _contains_any(text, ["nhl", "stanley cup", "hockey", "rangers", "canucks"]):
        return "Hockey"

    if _contains_any(
        text,
        [
            "fifa",
            "world cup",
            "soccer",
            "football",
            "uzbekistan",
            "curaçao",
            "new zealand",
            "usa win the 2026 fifa world cup",
        ],
    ):
        return "Football / Soccer"

    if _contains_any(
        text,
        [
            "macro",
            "inflation",
            "gdp",
            "cpi",
            "fed",
            "fomc",
            "economy",
            "business",
            "finance",
            "treasury",
            "interest rate",
        ],
    ):
        return "Macro / Business"

    if _contains_any(
        text,
        [
            "weather",
            "storm",
            "rain",
            "snow",
            "temperature",
            "hurricane",
            "climate",
        ],
    ):
        return "Weather"

    if _contains_any(
        text,
        [
            "openai",
            "artificial intelligence",
            "ai ",
            " ai",
            "technology",
            "tech",
            "science",
            "nasa",
            "spacex",
            "launch",
            "semiconductor",
        ],
    ):
        return "Tech / Science"

    if _contains_any(
        text,
        [
            "gta",
            "culture",
            "movie",
            "album",
            "celebrity",
            "tv",
            "box office",
            "gaming",
        ],
    ):
        return "Culture / Gaming"

    raw = (base_category or "").strip()
    if raw:
        return raw
    return "Other"
