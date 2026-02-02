"""Q&A pipeline: question → entity extraction → graph traversal → context assembly → LLM answer.

Answers include a consensus view from the majority of sources plus any
outlier / contrarian perspectives found in the data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from news_agent.graph import GraphStore
from news_agent.llm import LLM, extract_json

logger = logging.getLogger(__name__)


async def answer_question(
    question: str,
    llm: LLM,
    graph: GraphStore,
    since: datetime | None = None,
) -> str:
    """End-to-end Q&A: parse question → find graph context → generate answer."""
    # Step 1: Extract entities/keywords from the question
    entities = await _extract_question_entities(question, llm)
    logger.debug("Extracted question entities: %s", entities)

    # Step 2: Gather context from graph
    context = _gather_context(entities, graph, since)

    # Step 3: Generate answer with consensus + outliers
    return await _generate_answer(question, context, llm)


async def _extract_question_entities(question: str, llm: LLM) -> list[str]:
    """Use LLM to pull entity names / keywords from the user question."""
    prompt = f"""Extract the key financial entities, assets, or topics from this question.
Return a JSON array of strings, nothing else.

Question: {question}

Examples:
- "Why did oil prices drop today?" → ["OIL", "CRUDE"]
- "What happened with NVIDIA earnings?" → ["NVDA", "NVIDIA"]
- "How are emerging markets reacting to the Fed?" → ["EMERGING MARKETS", "FED", "FEDERAL RESERVE"]

```json
"""
    raw = await llm.generate(prompt, max_tokens=128)
    data = extract_json(raw)
    if isinstance(data, list):
        return [str(e).upper() for e in data if e]
    # Fallback: split question into significant words
    stopwords = {
        "what", "why", "how", "did", "does", "is", "are", "was", "were",
        "the", "a", "an", "to", "of", "in", "on", "for", "and", "or",
        "today", "yesterday", "this", "that", "with", "about", "do",
        "has", "have", "had", "will", "would", "could", "should",
    }
    return [w.upper() for w in question.split() if w.lower() not in stopwords and len(w) > 2]


def _gather_context(
    entities: list[str],
    graph: GraphStore,
    since: datetime | None = None,
) -> str:
    """Query the graph for all context related to the extracted entities."""
    since = since or datetime.utcnow() - timedelta(days=2)
    sections: list[str] = []

    all_articles: list[dict] = []
    all_events: list[dict] = []
    all_related: list[dict] = []

    for entity_name in entities:
        subgraph = graph.get_subgraph_for_entity(entity_name, since)
        all_articles.extend(subgraph["articles"])
        all_events.extend(subgraph["events"])
        all_related.extend(subgraph["related_entities"])

    # Deduplicate articles by id
    seen_ids: set[int] = set()
    unique_articles: list[dict] = []
    for a in all_articles:
        if a["id"] not in seen_ids:
            seen_ids.add(a["id"])
            unique_articles.append(a)

    if unique_articles:
        article_lines = []
        for a in unique_articles[:25]:
            sentiment_str = ""
            if a.get("sentiment") is not None:
                sentiment_str = f" [sentiment={a['sentiment']:.2f}]"
            article_lines.append(
                f"- {a['title']} ({a['source_name']}, {a['published_at'] or 'unknown'}){sentiment_str}"
                + (f"\n  {a['summary'][:200]}" if a.get("summary") else "")
            )
        sections.append("RELATED ARTICLES:\n" + "\n".join(article_lines))

    if all_events:
        # Deduplicate events by description
        seen_desc: set[str] = set()
        event_lines = []
        for ev in all_events:
            desc = ev["description"][:100]
            if desc not in seen_desc:
                seen_desc.add(desc)
                dir_str = f" → {ev['direction'].upper()}" if ev.get("direction") else ""
                event_lines.append(
                    f"- [{ev['event_type']}] {ev['description']}{dir_str}"
                )
        if event_lines:
            sections.append("RELATED EVENTS:\n" + "\n".join(event_lines[:15]))

    if all_related:
        # Deduplicate
        seen_names: set[str] = set()
        related_lines = []
        for r in all_related:
            if r["name"] not in seen_names:
                seen_names.add(r["name"])
                related_lines.append(
                    f"- {r['name']} ({r['entity_type']}) — {r['relation']}"
                    + (f" [{r['direction']}]" if r.get("direction") else "")
                )
        if related_lines:
            sections.append("RELATED ENTITIES:\n" + "\n".join(related_lines[:15]))

    if not sections:
        # Fall back to recent articles if graph has no matches
        recent = graph.get_recent_articles(since, limit=20)
        if recent:
            lines = [
                f"- {a['title']} ({a['source_name']})" for a in recent
            ]
            sections.append(
                "No specific graph matches found. Here are recent articles:\n"
                + "\n".join(lines)
            )

    return "\n\n".join(sections) if sections else "No relevant data found in the knowledge graph."


async def _generate_answer(question: str, context: str, llm: LLM) -> str:
    """Generate a structured answer with consensus + outlier views."""
    prompt = f"""You are a financial market analyst answering a user's question using data from multiple news sources.

USER QUESTION: {question}

AVAILABLE CONTEXT FROM KNOWLEDGE GRAPH:
{context}

INSTRUCTIONS:
1. Provide a clear, direct answer to the question.
2. Structure your answer as:

   **Consensus View**: What most sources and data points agree on. Cite specific articles or events from the context.

   **Outlier / Contrarian Views**: Any minority perspectives, dissenting analyst opinions, or data points that contradict the consensus. If no outliers exist, note that consensus is strong.

   **Key Drivers**: List the 2-4 main factors/events driving the situation.

   **Watch List**: What to monitor next — upcoming events or data releases that could change the picture.

3. Be specific — reference actual events and data from the context, not generic advice.
4. If the context doesn't contain enough information to answer fully, say so clearly and explain what data is available.
5. Keep the answer concise but complete.

ANSWER:"""

    return await llm.generate(prompt, max_tokens=1500)
