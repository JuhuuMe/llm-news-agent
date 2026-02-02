"""Abstract base class for news source adapters."""

from __future__ import annotations

import abc

from news_agent.models import Article


class NewsSource(abc.ABC):
    """All news sources implement this interface."""

    name: str = "unknown"

    @abc.abstractmethod
    async def fetch(self, query: str, max_results: int = 20) -> list[Article]:
        """Fetch articles matching *query*. Returns newest first."""
        ...
