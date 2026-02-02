"""News source adapters."""

from news_agent.sources.base import NewsSource
from news_agent.sources.newsapi import NewsAPISource
from news_agent.sources.finnhub import FinnhubSource
from news_agent.sources.alphavantage import AlphaVantageSource

__all__ = ["NewsSource", "NewsAPISource", "FinnhubSource", "AlphaVantageSource"]
