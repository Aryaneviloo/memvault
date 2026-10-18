from abc import ABC, abstractmethod

from memvault.core.models import MemoryItem, MemoryQuery
from memvault.embeddings.base import BaseEmbedder


class StorageBackend(ABC):
    """
    Abstract interface that every storage backend must implement.

    This defines *what* operations are possible on memory storage,
    without saying *how* they're implemented (SQLite, Postgres, etc.).
    """

    @abstractmethod
    def insert(self, item: MemoryItem) -> MemoryItem:
        """Store a new memory item and return it."""
        raise NotImplementedError

    @abstractmethod
    def get(self, item_id: str) -> MemoryItem | None:
        """Fetch a memory item by its ID. Return None if not found."""
        raise NotImplementedError

    @abstractmethod
    def update(self, item: MemoryItem) -> MemoryItem:
        """Persist changes to an existing memory item and return it."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, item_id: str, hard: bool = False) -> bool:
        """
        Delete a memory item.

        If hard=False (default), perform a soft delete (set deleted_at).
        If hard=True, permanently remove the record.
        Return True if something was deleted, False if not found.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query: MemoryQuery) -> list[MemoryItem]:
        """Return memory items matching the given query, ranked by relevance."""
        raise NotImplementedError

    @abstractmethod
    def list_recent(
        self,
        user_id: str,
        agent_id: str | None = None,
        namespace: str = "default",
        limit: int = 20,
    ) -> list[MemoryItem]:
        """Return the most recently created/updated memories for a scope."""
        raise NotImplementedError


    @abstractmethod
    def list_namespaces(
        self,
        user_id: str,
        agent_id: str | None = None,

    ) -> list[str]:
        """Return the distinct namespaces that contain live memories"""
        raise NotImplementedError


class EmbeddingStorageWrapper:
    """
    Wraps any Storage Backend to automactically generate embeddings on insert and update
    Keeps embedding logic out of storage backend

    Usage:
    store = EmbeddingStorageWrapper(
    backend = SQLiteStorage("memories,db),
    embedder = LocalEmbedder()
    )
    store.insert(item)

    """

    def __init__(self, backend: StorageBackend, embedder: BaseEmbedder) -> None:
        self._backend = backend
        self._embedder = embedder

    def insert(self, item: MemoryItem) -> MemoryItem:
        if item.embedding is None:
            item.embedding = self._embedder.embed(item.content)
        return self._backend.insert(item)

    def update(self, item: MemoryItem) -> MemoryItem:
        item.embedding = self._embedder.embed(item.content)
        return self._backend.update(item)

    def get(self, item_id: str) -> MemoryItem | None:
        return self._backend.get(item_id)

    def delete(self, item_id: str, hard: bool = False) -> bool:
        return self._backend.delete(item_id, hard=hard)

    def search(self, query: MemoryQuery) -> list[MemoryItem]:
        return self._backend.search(query)

    def list_recent(
        self,
        user_id: str,
        agent_id: str | None = None,
        namespace: str = "default",
        limit: int = 20,
    ) -> list[MemoryItem]:

        return self._backend.list_recent(
            user_id=user_id,
            agent_id=agent_id,
            namespace=namespace,
            limit=limit,
        )

    def list_namespaces(
            self,
            user_id: str,
            agent_id: str | None = None,

    ) -> list[str]:
        return self._backend.list_namespaces(
            user_id=user_id,
            agent_id=agent_id,
        )