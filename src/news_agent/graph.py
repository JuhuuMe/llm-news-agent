"""SQLite-backed property graph with LSTM-style memory management.

Tables:
    articles     — ingested articles (deduplicated by URL)
    entities     — named entities (ticker, commodity, institution, …)
    events       — market events extracted from articles
    edges        — typed relationships between any two nodes

Memory model (inspired by LSTM forget/update gates):
    Every node carries a `relevance` score (0.0–1.0) that determines
    whether the node survives the next pruning cycle.

    - DECAY:  Each cycle, relevance *= decay_factor (forget gate)
    - BOOST:  Re-referenced nodes get a relevance boost (input gate)
    - CONNECT: Highly-connected nodes decay slower (structural importance)
    - PRUNE:  Nodes below a threshold are removed (output gate)

    This means:
    - A one-off article about a minor event decays quickly and gets pruned
    - "The Fed" entity persists because it keeps being referenced
    - A rate decision event stays as long as its downstream effects matter
    - Old articles about resolved topics quietly disappear
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from news_agent.models import Article, Entity, EntityType, Event, EventType, Relationship

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("news_graph.db")

# Memory hyperparameters
DECAY_FACTOR = 0.85         # Per-cycle decay (like LSTM forget gate bias)
BOOST_ON_REFERENCE = 0.3    # Relevance boost when node is re-referenced
CONNECTIVITY_BONUS = 0.02   # Per-edge bonus applied before decay check
PRUNE_THRESHOLD = 0.05      # Nodes below this relevance get removed
INITIAL_RELEVANCE = 1.0     # New nodes start at full relevance


class GraphStore:
    """Lightweight graph with LSTM-style memory management."""

    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        self._migrate_add_memory_columns()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS articles (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                url             TEXT UNIQUE NOT NULL,
                title           TEXT NOT NULL,
                summary         TEXT DEFAULT '',
                source_name     TEXT DEFAULT '',
                published_at    TEXT,
                tickers         TEXT DEFAULT '[]',
                sentiment       REAL,
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                relevance       REAL NOT NULL DEFAULT 1.0,
                last_referenced TEXT NOT NULL DEFAULT (datetime('now')),
                reference_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS entities (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL,
                entity_type     TEXT NOT NULL,
                aliases         TEXT DEFAULT '[]',
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                relevance       REAL NOT NULL DEFAULT 1.0,
                last_referenced TEXT NOT NULL DEFAULT (datetime('now')),
                reference_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(name, entity_type)
            );

            CREATE TABLE IF NOT EXISTS events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                description     TEXT NOT NULL,
                event_type      TEXT NOT NULL DEFAULT 'other',
                timestamp       TEXT,
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                relevance       REAL NOT NULL DEFAULT 1.0,
                last_referenced TEXT NOT NULL DEFAULT (datetime('now')),
                reference_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS edges (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                source_id   INTEGER NOT NULL,
                target_type TEXT NOT NULL,
                target_id   INTEGER NOT NULL,
                relation    TEXT NOT NULL,
                direction   TEXT DEFAULT '',
                weight      REAL DEFAULT 1.0,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_type, source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_type, target_id);
            CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);
            CREATE INDEX IF NOT EXISTS idx_articles_relevance ON articles(relevance);
            CREATE INDEX IF NOT EXISTS idx_entities_relevance ON entities(relevance);
            CREATE INDEX IF NOT EXISTS idx_events_relevance ON events(relevance);
        """)
        self._conn.commit()

    def _migrate_add_memory_columns(self) -> None:
        """Add relevance/reference columns if upgrading from old schema."""
        for table in ("articles", "entities", "events"):
            cols = {
                row[1]
                for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if "relevance" not in cols:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN relevance REAL NOT NULL DEFAULT 1.0"
                )
            if "last_referenced" not in cols:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN last_referenced TEXT NOT NULL DEFAULT (datetime('now'))"
                )
            if "reference_count" not in cols:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN reference_count INTEGER NOT NULL DEFAULT 0"
                )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def upsert_article(self, article: Article) -> int:
        if not article.url:
            article.url = f"urn:title:{hash(article.title)}"
        cur = self._conn.execute(
            """INSERT INTO articles
               (url, title, summary, source_name, published_at, tickers, sentiment,
                relevance, last_referenced, reference_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 0)
               ON CONFLICT(url) DO UPDATE SET
                   relevance = MIN(1.0, relevance + ?),
                   last_referenced = datetime('now'),
                   reference_count = reference_count + 1""",
            (
                article.url,
                article.title,
                article.summary,
                article.source_name,
                article.published_at.isoformat() if article.published_at else None,
                json.dumps(article.tickers),
                article.raw_sentiment,
                INITIAL_RELEVANCE,
                BOOST_ON_REFERENCE,
            ),
        )
        self._conn.commit()
        if cur.lastrowid and cur.rowcount > 0:
            return cur.lastrowid
        row = self._conn.execute(
            "SELECT id FROM articles WHERE url = ?", (article.url,)
        ).fetchone()
        return row["id"] if row else -1

    def upsert_entity(self, entity: Entity) -> int:
        cur = self._conn.execute(
            """INSERT INTO entities (name, entity_type, aliases,
                   relevance, last_referenced, reference_count)
               VALUES (?, ?, ?, ?, datetime('now'), 0)
               ON CONFLICT(name, entity_type) DO UPDATE SET
                   aliases = excluded.aliases,
                   relevance = MIN(1.0, relevance + ?),
                   last_referenced = datetime('now'),
                   reference_count = reference_count + 1""",
            (
                entity.name.upper(),
                entity.entity_type.value,
                json.dumps(entity.aliases),
                INITIAL_RELEVANCE,
                BOOST_ON_REFERENCE,
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
        cur = self._conn.execute(
            """INSERT INTO events (description, event_type, timestamp,
                   relevance, last_referenced, reference_count)
               VALUES (?, ?, ?, ?, datetime('now'), 0)""",
            (
                event.description,
                event.event_type.value,
                event.timestamp.isoformat() if event.timestamp else None,
                INITIAL_RELEVANCE,
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
        """Full ingest: article + extracted entities, events, and relationships."""
        art_id = self.upsert_article(article)
        node_ids: dict[str, tuple[str, int]] = {}
        node_ids[article.url] = ("article", art_id)

        for ent in entities:
            eid = self.upsert_entity(ent)
            node_ids[ent.name.upper()] = ("entity", eid)
            self.insert_edge("article", art_id, "entity", eid, "MENTIONS")

        for evt in events:
            evid = self.insert_event(evt)
            node_ids[evt.description[:100]] = ("event", evid)
            self.insert_edge("article", art_id, "event", evid, "REPORTS_ON")

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
    # LSTM-style memory management
    # ------------------------------------------------------------------
    def memory_cycle(self) -> dict:
        """Run one memory management cycle: decay → connectivity bonus → prune.

        Returns stats about what happened.
        """
        stats = {"decayed": 0, "boosted": 0, "pruned": 0}

        # Step 1: DECAY — reduce relevance of all nodes (forget gate)
        for table in ("articles", "entities", "events"):
            cur = self._conn.execute(
                f"UPDATE {table} SET relevance = relevance * ?",
                (DECAY_FACTOR,),
            )
            stats["decayed"] += cur.rowcount

        # Step 2: CONNECTIVITY BONUS — highly-connected nodes decay slower
        # Count edges per entity and give a bonus proportional to connectivity
        entity_edges = self._conn.execute("""
            SELECT target_id, COUNT(*) as edge_count
            FROM edges WHERE target_type = 'entity'
            GROUP BY target_id
            UNION ALL
            SELECT source_id, COUNT(*) as edge_count
            FROM edges WHERE source_type = 'entity'
            GROUP BY source_id
        """).fetchall()

        # Aggregate edge counts per entity
        entity_edge_counts: dict[int, int] = {}
        for row in entity_edges:
            eid = row["target_id"]
            entity_edge_counts[eid] = entity_edge_counts.get(eid, 0) + row["edge_count"]

        for eid, count in entity_edge_counts.items():
            # Bonus scales with log of connectivity (diminishing returns)
            bonus = CONNECTIVITY_BONUS * math.log1p(count)
            self._conn.execute(
                "UPDATE entities SET relevance = MIN(1.0, relevance + ?) WHERE id = ?",
                (bonus, eid),
            )
            stats["boosted"] += 1

        # Same for events
        event_edges = self._conn.execute("""
            SELECT source_id, COUNT(*) as edge_count
            FROM edges WHERE source_type = 'event'
            GROUP BY source_id
        """).fetchall()
        for row in event_edges:
            bonus = CONNECTIVITY_BONUS * math.log1p(row["edge_count"])
            self._conn.execute(
                "UPDATE events SET relevance = MIN(1.0, relevance + ?) WHERE id = ?",
                (bonus, row["source_id"]),
            )
            stats["boosted"] += 1

        self._conn.commit()

        # Step 3: PRUNE — remove nodes below threshold (output gate)
        stats["pruned"] = self._prune_low_relevance()

        logger.info(
            "Memory cycle: decayed=%d boosted=%d pruned=%d",
            stats["decayed"], stats["boosted"], stats["pruned"],
        )
        return stats

    def _prune_low_relevance(self) -> int:
        """Remove nodes with relevance below threshold and their orphaned edges."""
        total = 0

        # Collect IDs to prune per table
        for table, node_type in [("articles", "article"), ("entities", "entity"), ("events", "event")]:
            rows = self._conn.execute(
                f"SELECT id FROM {table} WHERE relevance < ?",
                (PRUNE_THRESHOLD,),
            ).fetchall()

            ids_to_remove = [r["id"] for r in rows]
            if not ids_to_remove:
                continue

            placeholders = ",".join("?" * len(ids_to_remove))

            # Remove edges connected to these nodes
            self._conn.execute(
                f"DELETE FROM edges WHERE (source_type = ? AND source_id IN ({placeholders}))"
                f" OR (target_type = ? AND target_id IN ({placeholders}))",
                [node_type] + ids_to_remove + [node_type] + ids_to_remove,
            )

            # Remove the nodes themselves
            cur = self._conn.execute(
                f"DELETE FROM {table} WHERE id IN ({placeholders})",
                ids_to_remove,
            )
            total += cur.rowcount

        self._conn.commit()
        return total

    def boost_relevance(self, node_type: str, node_id: int, boost: float = BOOST_ON_REFERENCE) -> None:
        """Manually boost a node's relevance (e.g., when user queries about it)."""
        table = {"article": "articles", "entity": "entities", "event": "events"}.get(node_type)
        if table:
            self._conn.execute(
                f"UPDATE {table} SET relevance = MIN(1.0, relevance + ?), "
                f"last_referenced = datetime('now'), reference_count = reference_count + 1 "
                f"WHERE id = ?",
                (boost, node_id),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def find_entities(self, name: str) -> list[dict]:
        rows = self._conn.execute(
            """SELECT id, name, entity_type, aliases, relevance, reference_count
               FROM entities
               WHERE (name LIKE ? OR aliases LIKE ?) AND relevance >= ?
               ORDER BY relevance DESC, created_at DESC""",
            (f"%{name.upper()}%", f"%{name.upper()}%", PRUNE_THRESHOLD),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_related_articles(
        self, entity_name: str, since: datetime | None = None, limit: int = 20
    ) -> list[dict]:
        since_str = (since or datetime.utcnow() - timedelta(days=7)).isoformat()
        rows = self._conn.execute(
            """SELECT DISTINCT a.id, a.title, a.summary, a.url, a.source_name,
                      a.published_at, a.sentiment, a.relevance
               FROM articles a
               JOIN edges e ON e.source_type = 'article' AND e.source_id = a.id
                           AND e.target_type = 'entity' AND e.relation = 'MENTIONS'
               JOIN entities ent ON ent.id = e.target_id
               WHERE ent.name LIKE ?
                 AND a.relevance >= ?
                 AND (a.published_at >= ? OR a.published_at IS NULL)
               ORDER BY a.relevance DESC, a.published_at DESC
               LIMIT ?""",
            (f"%{entity_name.upper()}%", PRUNE_THRESHOLD, since_str, limit),
        ).fetchall()

        # Boost queried entities (user interest = relevance signal)
        entities = self._conn.execute(
            "SELECT id FROM entities WHERE name LIKE ?",
            (f"%{entity_name.upper()}%",),
        ).fetchall()
        for ent in entities:
            self.boost_relevance("entity", ent["id"], boost=0.1)

        return [dict(r) for r in rows]

    def get_events_affecting(
        self, entity_name: str, since: datetime | None = None, limit: int = 15
    ) -> list[dict]:
        since_str = (since or datetime.utcnow() - timedelta(days=7)).isoformat()
        rows = self._conn.execute(
            """SELECT ev.id, ev.description, ev.event_type, ev.timestamp,
                      e.direction, e.weight, ev.relevance
               FROM events ev
               JOIN edges e ON e.source_type = 'event' AND e.source_id = ev.id
                           AND e.target_type = 'entity' AND e.relation = 'AFFECTS'
               JOIN entities ent ON ent.id = e.target_id
               WHERE ent.name LIKE ?
                 AND ev.relevance >= ?
                 AND (ev.created_at >= ? OR ev.created_at IS NULL)
               ORDER BY ev.relevance DESC, ev.timestamp DESC NULLS LAST
               LIMIT ?""",
            (f"%{entity_name.upper()}%", PRUNE_THRESHOLD, since_str, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_subgraph_for_entity(
        self, entity_name: str, since: datetime | None = None
    ) -> dict:
        articles = self.get_related_articles(entity_name, since)
        events = self.get_events_affecting(entity_name, since)

        related_entities = self._conn.execute(
            """SELECT DISTINCT ent2.name, ent2.entity_type, e.relation, e.direction,
                      ent2.relevance
               FROM entities ent1
               JOIN edges e ON (e.source_type = 'entity' AND e.source_id = ent1.id)
                            OR (e.target_type = 'entity' AND e.target_id = ent1.id)
               JOIN entities ent2 ON (
                   (e.target_type = 'entity' AND e.target_id = ent2.id AND ent2.id != ent1.id)
                   OR (e.source_type = 'entity' AND e.source_id = ent2.id AND ent2.id != ent1.id)
               )
               WHERE ent1.name LIKE ? AND ent2.relevance >= ?""",
            (f"%{entity_name.upper()}%", PRUNE_THRESHOLD),
        ).fetchall()

        return {
            "entity": entity_name,
            "articles": articles,
            "events": events,
            "related_entities": [dict(r) for r in related_entities],
        }

    def get_recent_articles(self, since: datetime | None = None, limit: int = 50) -> list[dict]:
        since_str = (since or datetime.utcnow() - timedelta(days=1)).isoformat()
        rows = self._conn.execute(
            """SELECT id, title, summary, url, source_name, published_at, sentiment, relevance
               FROM articles
               WHERE (published_at >= ? OR published_at IS NULL) AND relevance >= ?
               ORDER BY relevance DESC, published_at DESC LIMIT ?""",
            (since_str, PRUNE_THRESHOLD, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        counts = {}
        for table in ("articles", "entities", "events", "edges"):
            row = self._conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
            counts[table] = row["cnt"]
        # Add memory stats
        for table in ("articles", "entities", "events"):
            row = self._conn.execute(
                f"SELECT AVG(relevance) as avg_rel, MIN(relevance) as min_rel, "
                f"MAX(relevance) as max_rel FROM {table}"
            ).fetchone()
            if row["avg_rel"] is not None:
                counts[f"{table}_avg_relevance"] = round(row["avg_rel"], 3)
                counts[f"{table}_min_relevance"] = round(row["min_rel"], 3)
        return counts

    # ------------------------------------------------------------------
    # Legacy compatibility — redirect to memory_cycle
    # ------------------------------------------------------------------
    def prune_before(self, cutoff: datetime) -> int:
        """Legacy: now delegates to memory_cycle for intelligent pruning."""
        return self.memory_cycle().get("pruned", 0)

    def prune_older_than_days(self, days: int = 7) -> int:
        """Legacy: now runs a memory cycle instead of blunt TTL."""
        return self.memory_cycle().get("pruned", 0)

    def close(self) -> None:
        self._conn.close()
