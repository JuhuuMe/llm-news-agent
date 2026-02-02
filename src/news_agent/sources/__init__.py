"""News source adapters."""

from news_agent.sources.base import NewsSource
from news_agent.sources.newsapi import NewsAPISource
from news_agent.sources.finnhub import FinnhubSource
from news_agent.sources.alphavantage import AlphaVantageSource
from news_agent.sources.rss import RSSSource
from news_agent.sources.reddit import RedditSource

__all__ = [
    "NewsSource",
    "NewsAPISource",
    "FinnhubSource",
    "AlphaVantageSource",
    "RSSSource",
    "RedditSource",
]
