"""Data models for news articles, user interests, and analysis results."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Urgency(str, Enum):
    """How urgently a piece of news requires attention."""

    BREAKING = "breaking"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class Article(BaseModel):
    """A single news article from any source."""

    title: str
    summary: str = ""
    url: str = ""
    source_name: str = ""
    published_at: datetime | None = None
    tickers: list[str] = Field(default_factory=list)
    raw_sentiment: float | None = None  # -1.0 (bearish) to 1.0 (bullish), if source provides it


class UserInterests(BaseModel):
    """What the user cares about."""

    topics: list[str] = Field(
        default_factory=lambda: ["stock market", "crypto", "commodities", "central banks"]
    )
    tickers: list[str] = Field(default_factory=list)  # e.g. ["AAPL", "TSLA", "BTC"]
    regions: list[str] = Field(default_factory=lambda: ["US", "EU"])


class MarketImpact(BaseModel):
    """Expected market impact of a news cluster."""

    affected_assets: list[str] = Field(default_factory=list)
    direction: str = ""  # "bullish", "bearish", "neutral", "mixed"
    confidence: str = ""  # "high", "medium", "low"
    reasoning: str = ""


class NewsDigest(BaseModel):
    """A single digest item — a cluster of related articles summarized."""

    headline: str
    summary: str
    urgency: Urgency = Urgency.NORMAL
    impact: MarketImpact = Field(default_factory=MarketImpact)
    source_articles: list[Article] = Field(default_factory=list)


class DailyOverview(BaseModel):
    """The full daily overview returned to the user."""

    generated_at: datetime = Field(default_factory=datetime.utcnow)
    market_mood: str = ""  # one-line overall sentiment
    digests: list[NewsDigest] = Field(default_factory=list)


class BreakingAlert(BaseModel):
    """An alert for news that needs immediate attention."""

    digest: NewsDigest
    reason: str = ""  # why this is breaking / urgent


# ---------------------------------------------------------------------------
# Graph RAG models
# ---------------------------------------------------------------------------


class EntityType(str, Enum):
    TICKER = "ticker"
    COMMODITY = "commodity"
    INDEX = "index"
    INSTITUTION = "institution"
    PERSON = "person"
    SECTOR = "sector"
    COUNTRY = "country"
    CURRENCY = "currency"


class EventType(str, Enum):
    RATE_DECISION = "rate_decision"
    EARNINGS = "earnings"
    GEOPOLITICAL = "geopolitical"
    REGULATORY = "regulatory"
    DATA_RELEASE = "data_release"
    IPO = "ipo"
    MERGER = "merger"
    COMMODITY_MOVE = "commodity_move"
    OTHER = "other"


class Entity(BaseModel):
    """A named entity in the knowledge graph."""

    name: str
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)


class Event(BaseModel):
    """A market event extracted from articles."""

    description: str
    event_type: EventType = EventType.OTHER
    timestamp: datetime | None = None


class Relationship(BaseModel):
    """An edge in the knowledge graph."""

    source: str  # entity or event name
    target: str  # entity or event name
    relation: str  # MENTIONS, REPORTS_ON, AFFECTS, RELATED_TO, CAUSED_BY
    direction: str = ""  # bullish / bearish / neutral (for AFFECTS edges)
    weight: float = 1.0


class ExtractionResult(BaseModel):
    """Structured output from the LLM entity/relationship extraction step."""

    entities: list[Entity] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
