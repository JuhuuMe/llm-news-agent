"""Finnhub adapter — financial-specific news with sentiment."""

from __future__ import annotations

from datetime import datetime, timedelta

import httpx

from news_agent.models import Article
from news_agent.sources.base import NewsSource

_BASE_URL = "https://finnhub.io/api/v1"


class FinnhubSource(NewsSource):
    name = "Finnhub"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def fetch(self, query: str, max_results: int = 20) -> list[Article]:
        # Finnhub general news endpoint — category=general covers broad market news.
        # For ticker-specific news use /company-news instead.
        now = datetime.utcnow()
        params = {
            "category": "general",
            "minId": 0,
            "token": self._api_key,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{_BASE_URL}/news", params=params)
            resp.raise_for_status()
            items = resp.json()

        # Finnhub returns a flat list; filter by rough keyword match on headline.
        keywords = {kw.lower() for kw in query.split()}
        articles: list[Article] = []
        for item in items:
            headline = (item.get("headline") or "").lower()
            if not keywords or any(kw in headline for kw in keywords):
                published = None
                if item.get("datetime"):
                    try:
                        published = datetime.fromtimestamp(item["datetime"])
                    except (ValueError, OSError):
                        pass
                articles.append(
                    Article(
                        title=item.get("headline") or "",
                        summary=item.get("summary") or "",
                        url=item.get("url") or "",
                        source_name=item.get("source", "Finnhub"),
                        published_at=published,
                    )
                )
            if len(articles) >= max_results:
                break
        return articles

    async def fetch_for_ticker(self, ticker: str, max_results: int = 10) -> list[Article]:
        """Fetch news for a specific stock ticker."""
        now = datetime.utcnow()
        params = {
            "symbol": ticker.upper(),
            "from": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
            "to": now.strftime("%Y-%m-%d"),
            "token": self._api_key,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{_BASE_URL}/company-news", params=params)
            resp.raise_for_status()
            items = resp.json()

        articles: list[Article] = []
        for item in items[:max_results]:
            published = None
            if item.get("datetime"):
                try:
                    published = datetime.fromtimestamp(item["datetime"])
                except (ValueError, OSError):
                    pass
            articles.append(
                Article(
                    title=item.get("headline") or "",
                    summary=item.get("summary") or "",
                    url=item.get("url") or "",
                    source_name=item.get("source", "Finnhub"),
                    published_at=published,
                    tickers=[ticker.upper()],
                )
            )
        return articles
