import json
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

from memvault.core.models import MemoryItem, MemoryQuery, MemoryType
from memvault.storage.base import StorageBackend

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    tags TEXT NOT NULL,
    metadata TEXT NOT NULL,
    source TEXT,
    embedding TEXT,
    importance REAL NOT NULL,
    confidence REAL NOT NULL,
    access_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT,
    expires_at TEXT,
    deleted_at TEXT
);
"""


class PostgresStorage(StorageBackend):
    """
    PostgreSQL-backed storage adapter.

    Connection string format:
        postgresql://user:password@host:port/dbname
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(SCHEMA)
        self._conn.commit()

    def _serialize(self, item: MemoryItem) -> tuple:
        return (
            item.id,
            item.agent_id,
            item.user_id,
            item.namespace,
            item.type.value,
            item.content,
            item.summary,
            json.dumps(item.tags),
            json.dumps(item.metadata),
            item.source,
            json.dumps(item.embedding) if item.embedding is not None else None,
            item.importance,
            item.confidence,
            item.access_count,
            item.created_at.isoformat(),
            item.updated_at.isoformat(),
            item.last_accessed_at.isoformat() if item.last_accessed_at else None,
            item.expires_at.isoformat() if item.expires_at else None,
            item.deleted_at.isoformat() if item.deleted_at else None,
        )

    @staticmethod
    def _deserialize(row: tuple) -> MemoryItem:
        return MemoryItem(
            id=row[0],
            agent_id=row[1],
            user_id=row[2],
            namespace=row[3],
            type=MemoryType(row[4]),
            content=row[5],
            summary=row[6],
            tags=json.loads(row[7]),
            metadata=json.loads(row[8]),
            source=row[9],
            embedding=json.loads(row[10]) if row[10] is not None else None,
            importance=row[11],
            confidence=row[12],
            access_count=row[13],
            created_at=datetime.fromisoformat(row[14]),
            updated_at=datetime.fromisoformat(row[15]),
            last_accessed_at=datetime.fromisoformat(row[16]) if row[16] else None,
            expires_at=datetime.fromisoformat(row[17]) if row[17] else None,
            deleted_at=datetime.fromisoformat(row[18]) if row[18] else None,
        )

    def insert(self, item: MemoryItem) -> MemoryItem:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                )
                """,
                self._serialize(item),
            )
        self._conn.commit()
        return item

    def get(self, item_id: str) -> MemoryItem | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM memories WHERE id = %s AND deleted_at IS NULL",
                (item_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._deserialize(row)

    def update(self, item: MemoryItem) -> MemoryItem:
        item.updated_at = datetime.now(timezone.utc)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memories SET
                    agent_id=%s, user_id=%s, namespace=%s, type=%s,
                    content=%s, summary=%s, tags=%s, metadata=%s,
                    source=%s, embedding=%s, importance=%s, confidence=%s,
                    access_count=%s, created_at=%s, updated_at=%s,
                    last_accessed_at=%s, expires_at=%s, deleted_at=%s
                WHERE id=%s
                """,
                self._serialize(item)[1:] + (item.id,),
            )
        self._conn.commit()
        return item

    def delete(self, item_id: str, hard: bool = False) -> bool:
        with self._conn.cursor() as cur:
            if hard:
                cur.execute(
                    "DELETE FROM memories WHERE id = %s",
                    (item_id,),
                )
            else:
                cur.execute(
                    """
                    UPDATE memories SET deleted_at = %s
                    WHERE id = %s AND deleted_at IS NULL
                    """,
                    (datetime.now(timezone.utc).isoformat(), item_id),
                )
            deleted = cur.rowcount > 0
        self._conn.commit()
        return deleted

    def search(self, query: MemoryQuery) -> list[MemoryItem]:
        sql = """
            SELECT * FROM memories
            WHERE deleted_at IS NULL
              AND user_id = %s
              AND namespace = %s
              AND content ILIKE %s
        """
        params: list = [query.user_id, query.namespace, f"%{query.text}%"]

        if query.agent_id is not None:
            sql += " AND agent_id = %s"
            params.append(query.agent_id)

        if query.types is not None:
            placeholders = ",".join("%s" for _ in query.types)
            sql += f" AND type IN ({placeholders})"
            params.extend(t.value for t in query.types)

        sql += " LIMIT %s"
        params.append(query.top_k)

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [self._deserialize(row) for row in rows]

    def list_recent(
        self,
        user_id: str,
        agent_id: str | None = None,
        namespace: str = "default",
        limit: int = 20,
    ) -> list[MemoryItem]:
        sql = """
            SELECT * FROM memories
            WHERE deleted_at IS NULL
              AND user_id = %s
              AND namespace = %s
        """
        params: list = [user_id, namespace]

        if agent_id is not None:
            sql += " AND agent_id = %s"
            params.append(agent_id)

        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [self._deserialize(row) for row in rows]

    def list_namespaces(self,
                        user_id: str,
                        agent_id: str | None = None
                        ) -> list[str]:
        sql = """
              SELECT DISTINCT namespace FROM memories
              WHERE deleted_at IS NULL
                 AND user_id = %s
            """
        params: list = [user_id]

        if agent_id is not None:
            sql += " AND agent_id = %s"
            params.append(agent_id)

        sql += " ORDER BY namespace"

        with self._conn.cursor() as curr:
            curr.execute(sql, params)
            rows = curr.fetchall()

        return [row[0] for row in rows]

    