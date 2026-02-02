"""Reddit adapter — public JSON API, no auth required for read-only access.

Fetches top/hot posts from financial and macro subreddits.
Rate limit: ~60 req/min without auth (Reddit tolerates moderate scraping
of the public JSON endpoints).
"""

from __future__ import annotations

import logging
from datetime import datetime

import httpx

from news_agent.models import Article
from news_agent.sources.base import NewsSource

logger = logging.getLogger(__name__)

# Curated subreddits by category
SUBREDDITS: dict[str, list[str]] = {
    "markets": ["wallstreetbets", "stocks", "investing", "stockmarket"],
    "macro": ["economics", "economy", "finance"],
    "politics": ["geopolitics", "worldnews", "politics"],
    "crypto": ["cryptocurrency", "bitcoin"],
    "commodities": ["commodities"],
}

_TOPIC_TO_SUBS: dict[str, str] = {
    "stock": "markets",
    "market": "markets",
    "sp500": "markets",
    "invest": "markets",
    "macro": "macro",
    "economy": "macro",
    "inflation": "macro",
    "fed": "macro",
    "interest rate": "macro",
    "gdp": "macro",
    "politic": "politics",
    "election": "politics",
    "geopolit": "politics",
    "war": "politics",
    "crypto": "crypto",
    "bitcoin": "crypto",
    "oil": "commodities",
    "gold": "commodities",
    "commodit": "commodities",
}


class RedditSource(NewsSource):
    """Fetches top posts from financial/macro subreddits. No API key required."""

    name = "Reddit"

    async def fetch(self, query: str, max_results: int = 20) -> list[Article]:
        subreddits = self._resolve_subreddits(query)
        articles: list[Article] = []

        async with httpx.AsyncClient(
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "llm-news-agent/0.1"},
        ) as client:
            for sub in subreddits:
                try:
                    fetched = await self._fetch_subreddit(client, sub, query, max_results)
                    articles.extend(fetched)
                except Exception as exc:
                    logger.debug("Reddit r/%s failed: %s", sub, exc)

        # Sort by score (popularity) then by time
        articles.sort(
            key=lambda a: a.published_at or datetime.min,
            reverse=True,
        )
        return articles[:max_results]

    def _resolve_subreddits(self, query: str) -> list[str]:
        query_lower = query.lower()
        matched_categories: set[str] = set()
        for keyword, category in _TOPIC_TO_SUBS.items():
            if keyword in query_lower:
                matched_categories.add(category)

        if not matched_categories:
            matched_categories = {"markets", "macro"}

        subs: list[str] = []
        for cat in matched_categories:
            subs.extend(SUBREDDITS.get(cat, []))
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for s in subs:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    async def _fetch_subreddit(
        self,
        client: httpx.AsyncClient,
        subreddit: str,
        query: str,
        max_results: int,
    ) -> list[Article]:
        """Fetch hot posts from a subreddit via public JSON API."""
        url = f"https://www.reddit.com/r/{subreddit}/hot.json"
        resp = await client.get(url, params={"limit": min(max_results, 25)})
        resp.raise_for_status()
        data = resp.json()

        articles: list[Article] = []
        query_keywords = {kw.lower() for kw in query.split() if len(kw) > 2}

        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            if not post:
                continue

            # Skip pinned/stickied posts
            if post.get("stickied"):
                continue

            title = post.get("title", "")
            selftext = post.get("selftext", "")[:500]
            url_link = post.get("url", "")
            permalink = post.get("permalink", "")
            score = post.get("score", 0)
            created_utc = post.get("created_utc")

            # Minimum quality filter: at least 10 upvotes
            if score < 10:
                continue

            # Keyword filter
            if query_keywords:
                text_lower = (title + " " + selftext).lower()
                if not any(kw in text_lower for kw in query_keywords):
                    continue

            published = None
            if created_utc:
                try:
                    published = datetime.fromtimestamp(created_utc)
                except (ValueError, OSError):
                    pass

            # Use permalink as URL for discussion, or external link if it's a link post
            article_url = url_link
            if not article_url or "reddit.com" in (article_url or ""):
                article_url = f"https://www.reddit.com{permalink}" if permalink else ""

            # Build summary from selftext or indicate it's a link post
            summary = selftext.strip() if selftext.strip() else f"[Link post with {score} upvotes]"

            articles.append(
                Article(
                    title=title,
                    summary=summary,
                    url=article_url,
                    source_name=f"r/{subreddit}",
                    published_at=published,
                )
            )

        return articles
