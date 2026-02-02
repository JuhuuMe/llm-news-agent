"""Alpha Vantage News Sentiment adapter — news with built-in sentiment scores."""

from __future__ import annotations

from datetime import datetime

import httpx

from news_agent.models import Article
from news_agent.sources.base import NewsSource

_BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageSource(NewsSource):
    name = "AlphaVantage"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def fetch(self, query: str, max_results: int = 20) -> list[Article]:
        params = {
            "function": "NEWS_SENTIMENT",
            "topics": _map_query_to_topics(query),
            "limit": min(max_results, 50),
            "apikey": self._api_key,
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        articles: list[Article] = []
        for item in data.get("feed", []):
            published = None
            raw_time = item.get("time_published", "")
            if raw_time:
                try:
                    published = datetime.strptime(raw_time, "%Y%m%dT%H%M%S")
                except ValueError:
                    pass

            # Extract sentiment score (Alpha Vantage gives -1 to 1)
            sentiment = None
            score_str = item.get("overall_sentiment_score")
            if score_str is not None:
                try:
                    sentiment = float(score_str)
                except ValueError:
                    pass

            tickers = [
                ts.get("ticker", "")
                for ts in item.get("ticker_sentiment", [])
                if ts.get("ticker")
            ]

            articles.append(
                Article(
                    title=item.get("title") or "",
                    summary=item.get("summary") or "",
                    url=item.get("url") or "",
                    source_name=item.get("source", "AlphaVantage"),
                    published_at=published,
                    tickers=tickers,
                    raw_sentiment=sentiment,
                )
            )
        return articles


def _map_query_to_topics(query: str) -> str:
    """Map free-text query to Alpha Vantage topic labels."""
    topic_map = {
        "stock": "earnings",
        "earnings": "earnings",
        "crypto": "blockchain",
        "bitcoin": "blockchain",
        "oil": "energy",
        "energy": "energy",
        "fed": "economy_monetary",
        "central bank": "economy_monetary",
        "interest rate": "economy_monetary",
        "inflation": "economy_macro",
        "gdp": "economy_macro",
        "ipo": "ipo",
        "merger": "mergers_and_acquisitions",
        "acquisition": "mergers_and_acquisitions",
        "tech": "technology",
    }
    query_lower = query.lower()
    matched = set()
    for keyword, topic in topic_map.items():
        if keyword in query_lower:
            matched.add(topic)
    # Default to broad financial topics if nothing matched
    if not matched:
        matched = {"economy_macro", "earnings", "economy_monetary"}
    return ",".join(sorted(matched))
