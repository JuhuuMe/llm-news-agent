"""LLM-based entity and relationship extraction from news articles."""

from __future__ import annotations

import logging

from news_agent.llm import LLM, extract_json
from news_agent.models import (
    Article,
    Entity,
    EntityType,
    Event,
    EventType,
    ExtractionResult,
    Relationship,
)

logger = logging.getLogger(__name__)

_ENTITY_TYPES = ", ".join(t.value for t in EntityType)
_EVENT_TYPES = ", ".join(t.value for t in EventType)

_EXTRACTION_PROMPT = """You are a financial data extraction engine. Given a news article, extract structured entities, events, and relationships.

Article title: {title}
Article summary: {summary}
Source: {source}
Published: {published}

Extract:
1. **Entities** — companies, tickers, commodities, indices, institutions, people, sectors, countries, currencies mentioned.
2. **Events** — specific market events described (rate decisions, earnings reports, geopolitical events, data releases, etc.).
3. **Relationships** — how entities and events are connected:
   - AFFECTS: an event affects an entity (include direction: bullish/bearish/neutral)
   - RELATED_TO: two entities are related
   - CAUSED_BY: one event caused another

Entity types: {entity_types}
Event types: {event_types}

Respond ONLY with valid JSON:
```json
{{
  "entities": [
    {{"name": "FED", "entity_type": "institution", "aliases": ["Federal Reserve", "FOMC"]}}
  ],
  "events": [
    {{"description": "Fed holds interest rates at 5.25-5.50%", "event_type": "rate_decision"}}
  ],
  "relationships": [
    {{"source": "Fed holds interest rates at 5.25-5.50%", "target": "USD", "relation": "AFFECTS", "direction": "neutral"}},
    {{"source": "USD", "target": "GOLD", "relation": "RELATED_TO"}}
  ]
}}
```

Be concise. Only extract what is explicitly stated or strongly implied. Do not invent information."""


async def extract_from_article(llm: LLM, article: Article) -> ExtractionResult:
    """Run entity/relationship extraction on a single article."""
    prompt = _EXTRACTION_PROMPT.format(
        title=article.title,
        summary=article.summary[:500],
        source=article.source_name,
        published=article.published_at or "unknown",
        entity_types=_ENTITY_TYPES,
        event_types=_EVENT_TYPES,
    )

    try:
        raw = await llm.generate(prompt, max_tokens=1024)
        data = extract_json(raw)
    except Exception as exc:
        logger.warning("Extraction failed for '%s': %s", article.title[:60], exc)
        return ExtractionResult()

    if not isinstance(data, dict):
        logger.debug("No valid JSON from extraction for '%s'", article.title[:60])
        return ExtractionResult()

    entities = _parse_entities(data.get("entities", []))
    events = _parse_events(data.get("events", []), article)
    relationships = _parse_relationships(data.get("relationships", []))

    return ExtractionResult(entities=entities, events=events, relationships=relationships)


async def extract_batch(llm: LLM, articles: list[Article]) -> list[ExtractionResult]:
    """Extract from multiple articles. Runs sequentially to respect rate limits."""
    results: list[ExtractionResult] = []
    for article in articles:
        result = await extract_from_article(llm, article)
        results.append(result)
    return results


def _parse_entities(raw_list: list) -> list[Entity]:
    entities = []
    for item in raw_list:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        etype_str = item.get("entity_type", "ticker")
        try:
            etype = EntityType(etype_str)
        except ValueError:
            etype = EntityType.TICKER
        entities.append(
            Entity(
                name=item["name"],
                entity_type=etype,
                aliases=item.get("aliases", []),
            )
        )
    return entities


def _parse_events(raw_list: list, article: Article) -> list[Event]:
    events = []
    for item in raw_list:
        if not isinstance(item, dict) or not item.get("description"):
            continue
        evtype_str = item.get("event_type", "other")
        try:
            evtype = EventType(evtype_str)
        except ValueError:
            evtype = EventType.OTHER
        events.append(
            Event(
                description=item["description"],
                event_type=evtype,
                timestamp=article.published_at,
            )
        )
    return events


def _parse_relationships(raw_list: list) -> list[Relationship]:
    relationships = []
    for item in raw_list:
        if not isinstance(item, dict) or not item.get("source") or not item.get("target"):
            continue
        relation = item.get("relation", "RELATED_TO").upper()
        if relation not in ("AFFECTS", "RELATED_TO", "CAUSED_BY", "MENTIONS", "REPORTS_ON"):
            relation = "RELATED_TO"
        relationships.append(
            Relationship(
                source=item["source"],
                target=item["target"],
                relation=relation,
                direction=item.get("direction", ""),
                weight=float(item.get("weight", 1.0)),
            )
        )
    return relationships
