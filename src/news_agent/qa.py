"""Q&A pipeline: question → entity extraction → graph traversal → ReAct tool loop → answer.

The LLM can call data-lookup tools (stock prices, economic indicators, etc.)
during reasoning. Answers include a consensus view plus outlier perspectives.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

from news_agent.config import Settings
from news_agent.graph import GraphStore
from news_agent.llm import LLM, extract_json
from news_agent.tools import ToolResult, execute_tool, get_tool_descriptions

logger = logging.getLogger(__name__)

# Max tool-use rounds before forcing a final answer
_MAX_TOOL_ROUNDS = 4


async def answer_question(
    question: str,
    llm: LLM,
    graph: GraphStore,
    settings: Settings,
    since: datetime | None = None,
) -> str:
    """End-to-end Q&A: parse question → graph context → ReAct tool loop → answer."""
    # Step 1: Extract entities/keywords from the question
    entities = await _extract_question_entities(question, llm)
    logger.debug("Extracted question entities: %s", entities)

    # Step 2: Gather context from graph
    graph_context = _gather_context(entities, graph, since)

    # Step 3: ReAct loop — let LLM call tools to get live data
    return await _react_loop(question, graph_context, llm, settings)


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
        recent = graph.get_recent_articles(since, limit=20)
        if recent:
            lines = [f"- {a['title']} ({a['source_name']})" for a in recent]
            sections.append(
                "No specific graph matches found. Here are recent articles:\n"
                + "\n".join(lines)
            )

    return "\n\n".join(sections) if sections else "No relevant data found in the knowledge graph."


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"TOOL:\s*(\w+)\(([^)]*)\)", re.IGNORECASE)


async def _react_loop(
    question: str,
    graph_context: str,
    llm: LLM,
    settings: Settings,
) -> str:
    """ReAct-style loop: LLM can think, call tools, observe results, then answer."""
    tool_descriptions = get_tool_descriptions()

    prompt = f"""You are a financial market analyst with access to live data tools and a news knowledge graph.

USER QUESTION: {question}

CONTEXT FROM NEWS KNOWLEDGE GRAPH:
{graph_context}

{tool_descriptions}

INSTRUCTIONS:
- First, think about what data you need to answer the question.
- If you need live data (prices, rates, indicators), call the appropriate tool using EXACTLY this syntax: TOOL: tool_name(args)
- You can call ONE tool at a time. After each tool call, you will see the result and can call another or give your final answer.
- When you have enough information, write your FINAL ANSWER starting with "ANSWER:"
- Structure the answer as:
  **Consensus View**: What most sources agree on, citing specific data.
  **Outlier / Contrarian Views**: Minority perspectives or contradictory data points.
  **Key Drivers**: 2-4 main factors.
  **Watch List**: What to monitor next.
- Be specific — use real numbers from tool results and articles from the context.

Begin your analysis:"""

    # Accumulate conversation for the ReAct loop
    full_prompt = prompt
    tool_results: list[ToolResult] = []

    for round_num in range(_MAX_TOOL_ROUNDS):
        raw = await llm.generate(full_prompt, max_tokens=1500)

        # Check if LLM produced a final answer
        answer_idx = raw.find("ANSWER:")
        if answer_idx != -1:
            return raw[answer_idx + len("ANSWER:"):].strip()

        # Check for tool call
        match = _TOOL_CALL_RE.search(raw)
        if not match:
            # No tool call and no ANSWER: tag — treat the whole output as the answer
            return raw.strip()

        tool_name = match.group(1).lower()
        tool_args = match.group(2).strip()

        logger.debug("Tool call [round %d]: %s(%s)", round_num + 1, tool_name, tool_args)
        result = await execute_tool(tool_name, tool_args, settings)
        tool_results.append(result)

        # Append the LLM's reasoning + tool result to the prompt
        # Only keep text up to (and including) the tool call
        reasoning = raw[:match.end()].strip()
        full_prompt += f"\n{reasoning}\n\nTOOL RESULT:\n{result.output}\n\nContinue your analysis. Call another tool or write your ANSWER:"

    # Max rounds reached — ask for final answer
    full_prompt += "\n\nYou have used all available tool calls. Write your ANSWER now:"
    raw = await llm.generate(full_prompt, max_tokens=1500)
    answer_idx = raw.find("ANSWER:")
    if answer_idx != -1:
        return raw[answer_idx + len("ANSWER:"):].strip()
    return raw.strip()
