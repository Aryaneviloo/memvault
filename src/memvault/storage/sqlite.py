import json
import sqlite3
from datetime import datetime, timezone

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


class SQLiteStorage(StorageBackend):
    """SQLit-backend storage adapter"""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(SCHEMA)
        self._conn.commit()

    @staticmethod
    def _serialize(item: MemoryItem) -> tuple:
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
        self._conn.execute(
            """
            INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            self._serialize(item),
        )
        self._conn.commit()
        return item

    def get(self, item_id: str) -> MemoryItem | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ? AND deleted_at IS NULL",
            (item_id,),
        ).fetchone()
        if row is None:
            return None
        return self._deserialize(row)

    def update(self, item: MemoryItem) -> MemoryItem:
        item.updated_at = datetime.now(timezone.utc)
        self._conn.execute(
            """
            UPDATE memories SET
                agent_id=?, user_id=?, namespace=?, type=?, content=?, summary=?,
                tags=?, metadata=?, source=?, embedding=?, importance=?, confidence=?,
                access_count=?, created_at=?, updated_at=?, last_accessed_at=?,
                expires_at=?, deleted_at=?
            WHERE id=?
            """,
            self._serialize(item)[1:] + (item.id,),
        )
        self._conn.commit()
        return item

    def delete(self, item_id: str, hard: bool = False) -> bool:
        if hard:
            cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (item_id,))
        else:
            cursor = self._conn.execute(
                "UPDATE memories SET deleted_at = ? WHERE id = ? AND deleted_at is NULL",
                (datetime.now(timezone.utc).isoformat(), item_id),
            )

        self._conn.commit()
        return cursor.rowcount > 0

    def search(self, query: MemoryQuery) -> list[MemoryItem]:
        sql = """
            SELECT * FROM memories
            WHERE deleted_at is NULL
              AND user_id = ?
              AND namespace = ?
              AND content LIKE ?
        """

        params: list = [query.user_id, query.namespace, f"%{query.text}%"]

        if query.agent_id is not None:
            sql += "AND agent_id = ? "
            params.append(query.agent_id)

        if query.types is not None:
            placeholders = ",".join("?" for _ in query.types)
            sql += f" AND type IN ({placeholders})"
            params.extend(t.value for t in query.types)

        sql += " LIMIT ?"
        params.append(query.top_k)

        rows = self._conn.execute(sql, params).fetchall()
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
              WHERE deleted_at is NULL
                AND user_id = ?
                AND namespace = ?
        """

        params: list = [user_id, namespace]

        if agent_id is not None:
            sql += "AND agent_id = ? "
            params.append(agent_id)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._deserialize(row) for row in rows]

    def list_namespaces(self, user_id, agent_id: str | None = None) -> list[str]:
        sql = """
                SELECT DISTINCT namespace FROM memories
                WHERE deleted_at IS NULL
                    AND user_id = ?
            """

        params: list = [user_id]

        if agent_id is not None:
            sql += " AND agent_id = ?"
            params.append(agent_id)

        sql += " ORDER by namespace"

        rows = self._conn.execute(sql, params).fetchall()
        return [row[0] for row in rows]

    