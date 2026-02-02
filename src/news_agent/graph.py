"""SQLite-backed property graph for financial news knowledge.

Tables:
    articles     — ingested articles (deduplicated by URL)
    entities     — named entities (ticker, commodity, institution, …)
    events       — market events extracted from articles
    edges        — typed relationships between any two nodes

Every row carries a `created_at` timestamp so stale data can be pruned
with `prune_before()`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from news_agent.models import Article, Entity, EntityType, Event, EventType, Relationship

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("news_graph.db")


class GraphStore:
    """Lightweight graph backed by a single SQLite file."""

    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS articles (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT UNIQUE NOT NULL,
                title       TEXT NOT NULL,
                summary     TEXT DEFAULT '',
                source_name TEXT DEFAULT '',
                published_at TEXT,
                tickers     TEXT DEFAULT '[]',
                sentiment   REAL,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS entities (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                aliases     TEXT DEFAULT '[]',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(name, entity_type)
            );

            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                event_type  TEXT NOT NULL DEFAULT 'other',
                timestamp   TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS edges (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,   -- 'article', 'entity', 'event'
                source_id   INTEGER NOT NULL,
                target_type TEXT NOT NULL,
                target_id   INTEGER NOT NULL,
                relation    TEXT NOT NULL,    -- MENTIONS, REPORTS_ON, AFFECTS, RELATED_TO, CAUSED_BY
                direction   TEXT DEFAULT '',  -- bullish / bearish / neutral
                weight      REAL DEFAULT 1.0,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_type, source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_type, target_id);
            CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def upsert_article(self, article: Article) -> int:
        """Insert or ignore an article. Returns its row id."""
        if not article.url:
            # Generate a pseudo-URL from title hash for articles without URLs
            article.url = f"urn:title:{hash(article.title)}"
        cur = self._conn.execute(
            """INSERT OR IGNORE INTO articles
               (url, title, summary, source_name, published_at, tickers, sentiment)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                article.url,
                article.title,
                article.summary,
                article.source_name,
                article.published_at.isoformat() if article.published_at else None,
                json.dumps(article.tickers),
                article.raw_sentiment,
            ),
        )
        self._conn.commit()
        if cur.lastrowid and cur.rowcount > 0:
            return cur.lastrowid
        # Already existed — fetch id
        row = self._conn.execute(
            "SELECT id FROM articles WHERE url = ?", (article.url,)
        ).fetchone()
        return row["id"] if row else -1

    def upsert_entity(self, entity: Entity) -> int:
        """Insert or update an entity. Returns its row id."""
        cur = self._conn.execute(
            """INSERT INTO entities (name, entity_type, aliases)
               VALUES (?, ?, ?)
               ON CONFLICT(name, entity_type) DO UPDATE SET
                   aliases = excluded.aliases""",
            (
                entity.name.upper(),
                entity.entity_type.value,
                json.dumps(entity.aliases),
            ),
        )
        self._conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        row = self._conn.execute(
            "SELECT id FROM entities WHERE name = ? AND entity_type = ?",
            (entity.name.upper(), entity.entity_type.value),
        ).fetchone()
        return row["id"] if row else -1

    def insert_event(self, event: Event) -> int:
        """Insert an event. Returns its row id."""
        cur = self._conn.execute(
            """INSERT INTO events (description, event_type, timestamp)
               VALUES (?, ?, ?)""",
            (
                event.description,
                event.event_type.value,
                event.timestamp.isoformat() if event.timestamp else None,
            ),
        )
        self._conn.commit()
        return cur.lastrowid or -1

    def insert_edge(
        self,
        source_type: str,
        source_id: int,
        target_type: str,
        target_id: int,
        relation: str,
        direction: str = "",
        weight: float = 1.0,
    ) -> None:
        self._conn.execute(
            """INSERT INTO edges
               (source_type, source_id, target_type, target_id, relation, direction, weight)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (source_type, source_id, target_type, target_id, relation, direction, weight),
        )
        self._conn.commit()

    def ingest_extraction(
        self, article: Article, entities: list[Entity], events: list[Event], relationships: list[Relationship]
    ) -> None:
        """Full ingest: article + its extracted entities, events, and relationships."""
        art_id = self.upsert_article(article)

        # Build name→(type, id) lookup for wiring edges
        node_ids: dict[str, tuple[str, int]] = {}
        node_ids[article.url] = ("article", art_id)

        for ent in entities:
            eid = self.upsert_entity(ent)
            node_ids[ent.name.upper()] = ("entity", eid)
            # Auto-create MENTIONS edge between article and entity
            self.insert_edge("article", art_id, "entity", eid, "MENTIONS")

        for evt in events:
            evid = self.insert_event(evt)
            node_ids[evt.description[:100]] = ("event", evid)
            # Auto-create REPORTS_ON edge
            self.insert_edge("article", art_id, "event", evid, "REPORTS_ON")

        # Wire up extracted relationships
        for rel in relationships:
            src_key = rel.source.upper()
            tgt_key = rel.target.upper()
            src = node_ids.get(src_key)
            tgt = node_ids.get(tgt_key)
            if src and tgt:
                self.insert_edge(
                    src[0], src[1], tgt[0], tgt[1],
                    rel.relation, rel.direction, rel.weight,
                )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def find_entities(self, name: str) -> list[dict]:
        """Find entities matching a name (case-insensitive, partial match)."""
        rows = self._conn.execute(
            """SELECT id, name, entity_type, aliases FROM entities
               WHERE name LIKE ? OR aliases LIKE ?
               ORDER BY created_at DESC""",
            (f"%{name.upper()}%", f"%{name.upper()}%"),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_related_articles(
        self, entity_name: str, since: datetime | None = None, limit: int = 20
    ) -> list[dict]:
        """Get articles that MENTION a given entity, optionally filtered by time."""
        since_str = (since or datetime.utcnow() - timedelta(days=2)).isoformat()
        rows = self._conn.execute(
            """SELECT DISTINCT a.id, a.title, a.summary, a.url, a.source_name,
                      a.published_at, a.sentiment
               FROM articles a
               JOIN edges e ON e.source_type = 'article' AND e.source_id = a.id
                           AND e.target_type = 'entity' AND e.relation = 'MENTIONS'
               JOIN entities ent ON ent.id = e.target_id
               WHERE ent.name LIKE ?
                 AND a.published_at >= ?
               ORDER BY a.published_at DESC
               LIMIT ?""",
            (f"%{entity_name.upper()}%", since_str, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_events_affecting(
        self, entity_name: str, since: datetime | None = None, limit: int = 15
    ) -> list[dict]:
        """Get events that AFFECT a given entity."""
        since_str = (since or datetime.utcnow() - timedelta(days=2)).isoformat()
        rows = self._conn.execute(
            """SELECT ev.id, ev.description, ev.event_type, ev.timestamp,
                      e.direction, e.weight
               FROM events ev
               JOIN edges e ON e.source_type = 'event' AND e.source_id = ev.id
                           AND e.target_type = 'entity' AND e.relation = 'AFFECTS'
               JOIN entities ent ON ent.id = e.target_id
               WHERE ent.name LIKE ?
                 AND ev.created_at >= ?
               ORDER BY ev.timestamp DESC NULLS LAST
               LIMIT ?""",
            (f"%{entity_name.upper()}%", since_str, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_subgraph_for_entity(
        self, entity_name: str, since: datetime | None = None
    ) -> dict:
        """Retrieve the full subgraph around an entity — articles, events, related entities."""
        articles = self.get_related_articles(entity_name, since)
        events = self.get_events_affecting(entity_name, since)

        # Also find entities related to this entity
        related_entities = self._conn.execute(
            """SELECT DISTINCT ent2.name, ent2.entity_type, e.relation, e.direction
               FROM entities ent1
               JOIN edges e ON (e.source_type = 'entity' AND e.source_id = ent1.id)
                            OR (e.target_type = 'entity' AND e.target_id = ent1.id)
               JOIN entities ent2 ON (
                   (e.target_type = 'entity' AND e.target_id = ent2.id AND ent2.id != ent1.id)
                   OR (e.source_type = 'entity' AND e.source_id = ent2.id AND ent2.id != ent1.id)
               )
               WHERE ent1.name LIKE ?""",
            (f"%{entity_name.upper()}%",),
        ).fetchall()

        return {
            "entity": entity_name,
            "articles": articles,
            "events": events,
            "related_entities": [dict(r) for r in related_entities],
        }

    def get_recent_articles(self, since: datetime | None = None, limit: int = 50) -> list[dict]:
        """Get all recent articles."""
        since_str = (since or datetime.utcnow() - timedelta(days=1)).isoformat()
        rows = self._conn.execute(
            """SELECT id, title, summary, url, source_name, published_at, sentiment
               FROM articles WHERE published_at >= ?
               ORDER BY published_at DESC LIMIT ?""",
            (since_str, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Return counts of all node/edge types."""
        counts = {}
        for table in ("articles", "entities", "events", "edges"):
            row = self._conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
            counts[table] = row["cnt"]
        return counts

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def prune_before(self, cutoff: datetime) -> int:
        """Delete all data older than *cutoff*. Returns total rows removed."""
        cutoff_str = cutoff.isoformat()
        total = 0
        for table in ("articles", "entities", "events", "edges"):
            cur = self._conn.execute(
                f"DELETE FROM {table} WHERE created_at < ?", (cutoff_str,)
            )
            total += cur.rowcount
        self._conn.commit()
        logger.info("Pruned %d rows older than %s", total, cutoff_str)
        return total

    def prune_older_than_days(self, days: int = 7) -> int:
        return self.prune_before(datetime.utcnow() - timedelta(days=days))

    def close(self) -> None:
        self._conn.close()
