"""NewsAPI.org adapter — broad coverage, 80k+ sources."""

from __future__ import annotations

from datetime import datetime, timedelta

import httpx

from news_agent.models import Article
from news_agent.sources.base import NewsSource

_BASE_URL = "https://newsapi.org/v2"


class NewsAPISource(NewsSource):
    name = "NewsAPI"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def fetch(self, query: str, max_results: int = 20) -> list[Article]:
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(max_results, 100),
            "from": (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "apiKey": self._api_key,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{_BASE_URL}/everything", params=params)
            resp.raise_for_status()
            data = resp.json()

        articles: list[Article] = []
        for item in data.get("articles", []):
            published = None
            if item.get("publishedAt"):
                try:
                    published = datetime.fromisoformat(
                        item["publishedAt"].replace("Z", "+00:00")
                    )
                except ValueError:
                    pass
            articles.append(
                Article(
                    title=item.get("title") or "",
                    summary=item.get("description") or "",
                    url=item.get("url") or "",
                    source_name=item.get("source", {}).get("name", "NewsAPI"),
                    published_at=published,
                )
            )
        return articles
