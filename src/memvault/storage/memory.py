from datetime import datetime, timezone

from memvault.core.models import MemoryItem, MemoryQuery
from memvault.storage.base import StorageBackend


class InMemoryStorage(StorageBackend):
    """
    Simple dict based storage backend wihtouot actual db
    Useful for tests and local experiment
    """

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    def insert(self, item: MemoryItem) -> MemoryItem:
        self._items[item.id] = item
        return item

    def get(self, item_id: str) -> MemoryItem | None:
        item = self._items.get(item_id)
        if item is None or item.deleted_at is not None:
            return None
        return item

    def update(self, item: MemoryItem) -> MemoryItem:
        item.updated_at = datetime.now(timezone.utc)
        self._items[item.id] = item
        return item

    def delete(self, item_id: str, hard: bool = False) -> bool:
        item = self._items.get(item_id)
        if item is None:
            return False
        if hard:
            del self._items[item_id]
        else:
            item.deleted_at = datetime.now(timezone.utc)
        return True

    def search(self, query: MemoryQuery) -> list[MemoryItem]:
        results = [
            item
            for item in self._items.values()
            if item.deleted_at is None
            and item.user_id == query.user_id
            and item.namespace == query.namespace
            and (query.agent_id is None or item.agent_id == query.agent_id)
            and (query.types is None or item.type in query.types)
            and query.text.lower() in item.content.lower()
        ]
        return results[: query.top_k]

    def list_recent(
        self,
        user_id: str,
        agent_id: str | None = None,
        namespace: str = "default",
        limit: int = 20,
    ) -> list[MemoryItem]:
        results = [
            item
            for item in self._items.values()
            if item.deleted_at is None
            and item.user_id == user_id
            and item.namespace == namespace
            and (agent_id is None or item.agent_id == agent_id)
        ]
        results.sort(key=lambda i: i.created_at, reverse=True)
        return results[:limit]

    def list_namespaces(self, 
                        user_id: str,
                        agent_id: str | None = None
                        ) -> list[str]:

        namespaces = {
            item.namespace
            for item in self._items.values()
            if item.deleted_at is None
            and item.user_id == user_id
            and( agent_id is None or item.agent_id == agent_id)
        }

        return sorted(namespaces)
